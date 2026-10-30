"""Bounded retention for automatically collected messages and completed jobs."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

try:
    from ...services.injection.query_gate import is_placeholder_only
except ImportError:
    from services.injection.query_gate import is_placeholder_only


@dataclass(frozen=True)
class RetentionPolicy:
    """Retention limits for automatic messages; explicit knowledge is exempt."""

    noise_days: float = 7.0
    low_value_days: float = 90.0
    recent_days: float = 30.0
    max_memories: int = 50000
    max_per_scope: int = 20000
    job_days: float = 7.0

    @classmethod
    def from_settings(cls, settings):
        """Read numeric limits, treating missing upgrade fields as defaults.

        Args:
            settings: Eviction_Settings mapping.

        Returns:
            A policy with positive limits.
        """
        defaults = cls()
        values = {}
        for field in cls.__dataclass_fields__:
            default = getattr(defaults, field)
            value = settings.get(field)
            try:
                values[field] = type(default)(value) if value is not None else default
            except (TypeError, ValueError, OverflowError):
                values[field] = default
            if not math.isfinite(values[field]) or values[field] <= 0:
                values[field] = default
        values["low_value_days"] = max(values["recent_days"], values["low_value_days"])
        return cls(**values)


def retention_candidates(
    connection,
    policy: RetentionPolicy,
    *,
    now: float,
    limit: int = 100,
    scan_placeholders: bool = True,
):
    """Select stale low-value messages and the lowest-ranked overflow per scope.

    Selection and deletion run in the same writer transaction. Temporary tables
    contain IDs and scores only, never text or vectors. Explicit knowledge,
    recent/recently recalled messages, shared grants and evidence are protected.

    Args:
        connection: Writer-owned SQLite connection.
        policy: Retention ages and automatic-memory budgets.
        now: Current Unix time.
        limit: Maximum number of IDs returned in one short transaction.
        scan_placeholders: Refresh placeholder IDs once at the start of a sweep.

    Returns:
        Candidate rows containing memory identity, revision and canonical scope.
    """
    tables = {
        row[0]: {
            item[1] for item in connection.execute(f'PRAGMA table_info("{row[0]}")')
        }
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        if not row[0].startswith(("fts_", "character_experiences_fts"))
    }
    connection.execute(
        "CREATE TEMP TABLE IF NOT EXISTS retention_keep(id INTEGER PRIMARY KEY)"
    )
    connection.execute("DELETE FROM retention_keep")
    # Direct evidence links are authoritative regardless of knowledge status.
    for table, columns in tables.items():
        for column in columns & {
            "source_memory_id",
            "origin_memory_id",
            "reply_memory_id",
            "response_memory_id",
        }:
            connection.execute(
                f'INSERT OR IGNORE INTO retention_keep SELECT "{column}" FROM "{table}" WHERE "{column}">0'
            )
    if "shared_memory_grants" in tables:
        connection.execute(
            "INSERT OR IGNORE INTO retention_keep SELECT memory_id FROM shared_memory_grants WHERE status='active'"
        )
    if "memory_feedback" in tables:
        connection.execute(
            "INSERT OR IGNORE INTO retention_keep SELECT memory_id FROM memory_feedback WHERE feedback IN ('useful','relevant','positive')"
        )
    # Structured evidence can reference memory IDs inside ObjectRefs or lists.
    for table, columns in tables.items():
        evidence_columns = columns & {
            "evidence",
            "evidence_ids",
            "memory_ids",
            "source_memory_ids",
            "evidence_refs_json",
            "evidence_bindings_json",
            "source_ref",
            "provenance",
        }
        if table == "memories":
            continue
        for column in evidence_columns:
            for (raw,) in connection.execute(
                f'SELECT "{column}" FROM "{table}" WHERE "{column}" IS NOT NULL AND "{column}" NOT IN (\'{{}}\',\'[]\',\'\')'
            ):
                try:
                    stack = [
                        (
                            json.loads(raw),
                            column
                            in {"memory_ids", "evidence_ids", "source_memory_ids"},
                        )
                    ]
                except (TypeError, ValueError):
                    continue
                keep = []
                while stack:
                    value, memory_list = stack.pop()
                    if isinstance(value, dict):
                        memory_ref = (
                            value.get("kind") == "memory"
                            or value.get("object_type") == "memory"
                            or value.get("aggregate_kind") == "memory"
                        )
                        for key, item in value.items():
                            is_id = key in {
                                "memory_id",
                                "source_memory_id",
                                "origin_memory_id",
                            } or (
                                memory_ref
                                and key in {"id", "object_id", "aggregate_id"}
                            )
                            if is_id:
                                try:
                                    keep.append((int(item),))
                                except (TypeError, ValueError):
                                    pass
                            elif isinstance(item, (dict, list)):
                                stack.append(
                                    (
                                        item,
                                        key
                                        in {
                                            "memory_ids",
                                            "evidence_ids",
                                            "source_memory_ids",
                                        },
                                    )
                                )
                    elif isinstance(value, list):
                        stack.extend((item, memory_list) for item in value)
                    elif memory_list:
                        try:
                            keep.append((int(value),))
                        except (TypeError, ValueError):
                            pass
                connection.executemany(
                    "INSERT OR IGNORE INTO retention_keep VALUES (?)", keep
                )

    day = 86400
    connection.execute(
        "CREATE TEMP TABLE IF NOT EXISTS retention_empty(id INTEGER PRIMARY KEY,version INTEGER)"
    )
    if scan_placeholders:
        connection.execute("DELETE FROM retention_empty")
        cursor = connection.execute(
            """SELECT id,version,content FROM memories
                WHERE source IN ('chat','live','noise','identity_quarantine')"""
        )
        connection.executemany(
            "INSERT INTO retention_empty VALUES (?,?)",
            (
                (mid, version)
                for mid, version, content in cursor
                if is_placeholder_only(content)
            ),
        )
    connection.execute("""CREATE TEMP TABLE IF NOT EXISTS retention_pool(
        id INTEGER PRIMARY KEY, version INTEGER, bot_id TEXT, session_id TEXT,
        visibility TEXT, group_id TEXT, score REAL, stale INTEGER, eligible INTEGER)""")
    connection.execute("DELETE FROM retention_pool")
    connection.execute(
        """INSERT INTO retention_pool
           SELECT m.id, COALESCE(m.version,1), m.bot_id, m.session_id, m.visibility, m.group_id,
               CASE WHEN m.memory_type IN ('noise','archived','evicted','deleted') OR m.source IN ('noise','identity_quarantine') THEN -10
                    ELSE COALESCE(m.importance,1) * 4 + CASE WHEN m.source='core' THEN 2 ELSE 0 END
                         + MIN(COALESCE(m.access_count,0),20) * 0.5
                         + MAX(0, 1 - (? - MAX(m.timestamp,COALESCE(m.last_accessed,0))) / (? * 86400.0)) * 3 END,
               CASE WHEN m.memory_type IN ('noise','archived','evicted','deleted') OR m.source IN ('noise','identity_quarantine')
                    THEN m.timestamp < ?
                    ELSE m.source!='core' AND m.timestamp < ? AND COALESCE(m.access_count,0)<3 AND COALESCE(m.importance,1)<1.5 END,
               CASE WHEN m.memory_type IN ('noise','archived','evicted','deleted') OR m.source IN ('noise','identity_quarantine')
                    THEN m.timestamp < ?
                    ELSE m.timestamp < ? AND COALESCE(m.last_accessed,0) < ? END
             FROM memories m
            WHERE m.source IN ('chat','live','core','noise','identity_quarantine')
              AND COALESCE(m.sender_id,'') NOT IN ('bot_remember','compat_livingmemory')
              AND m.memory_type IN ('message','noise','archived','evicted','deleted')
              AND m.resolution_state='resolved' AND m.version IS NOT NULL
              AND m.visibility IN ('group','private')
              AND COALESCE(m.bot_id,'')!='' AND COALESCE(m.session_id,'')!=''
              AND NOT EXISTS(SELECT 1 FROM retention_keep k WHERE k.id=m.id)""",
        (
            now,
            policy.low_value_days,
            now - policy.noise_days * day,
            now - policy.low_value_days * day,
            now - policy.noise_days * day,
            now - policy.recent_days * day,
            now - 7 * day,
        ),
    )
    # Recheck the captured revision so an edited placeholder cannot be deleted.
    connection.execute("""UPDATE retention_pool SET stale=1,eligible=1,score=-20
        WHERE EXISTS(SELECT 1 FROM retention_empty e WHERE e.id=retention_pool.id AND e.version=retention_pool.version)""")
    # Rank before filtering recent items so those items count against budgets.
    return connection.execute(
        """WITH ranked AS (
             SELECT *, ROW_NUMBER() OVER (
                 PARTITION BY bot_id,session_id,visibility ORDER BY eligible,score DESC,id DESC) scope_rank,
                 ROW_NUMBER() OVER (ORDER BY eligible,score DESC,id DESC) total_rank
               FROM retention_pool)
           SELECT id,version,bot_id,session_id,visibility,group_id
             FROM ranked WHERE eligible AND (stale OR scope_rank>? OR total_rank>?)
            ORDER BY stale DESC,score,id LIMIT ?""",
        (policy.max_per_scope, policy.max_memories, max(1, int(limit))),
    ).fetchall()


def prune_job_history(connection, *, before: float, limit: int = 1000) -> int:
    """Delete old terminal runs, preserving each slot's latest generation.

    Args:
        connection: Writer-owned connection.
        before: Retention cutoff timestamp.
        limit: Maximum number of rows to remove per transaction.

    Returns:
        Number of deleted historical runs.
    """
    return connection.execute(
        """DELETE FROM background_job_runs WHERE rowid IN (
             SELECT r.rowid FROM background_job_runs r
              WHERE r.status IN ('succeeded','failed','cancelled') AND r.updated_at<?
                AND EXISTS(SELECT 1 FROM background_job_runs newer
                    WHERE newer.request_id=r.request_id AND newer.schedule_slot=r.schedule_slot
                      AND newer.cursor_generation>r.cursor_generation)
              ORDER BY r.updated_at LIMIT ?)""",
        (before, limit),
    ).rowcount
