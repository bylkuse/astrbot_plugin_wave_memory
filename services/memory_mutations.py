"""Scoped Memory WebUI mutations through the writer-owned transaction and outbox."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

try:
    from ..domain.scope import RuntimeScope, scope_to_dict
    from ..engine.db.memory_repo import MemoryRepo, MemoryRevisionConflict
    from ..engine.db.outbox_repo import OutboxRepository
    from ..engine.db.scoped_tag_projection import rebuild_memory_effective_tags
except ImportError:  # pragma: no cover - focused tests import top-level packages
    from domain.scope import RuntimeScope, scope_to_dict
    from engine.db.memory_repo import MemoryRepo, MemoryRevisionConflict
    from engine.db.outbox_repo import OutboxRepository
    from engine.db.scoped_tag_projection import rebuild_memory_effective_tags


_UNSET = object()


@dataclass(frozen=True)
class MemoryMutationTarget:
    memory_id: int
    revision: int

    def to_dict(self) -> dict[str, int]:
        return {"memory_id": int(self.memory_id), "revision": int(self.revision)}


@dataclass(frozen=True)
class MemoryTagCorrectionTarget:
    correction_id: str
    revision: int
    status: str


@dataclass(frozen=True)
class MemoryMutationResult:
    operation_id: str
    targets: tuple[MemoryMutationTarget, ...]
    corrections: tuple[MemoryTagCorrectionTarget, ...] = ()

    @property
    def revision(self) -> int | None:
        return self.targets[0].revision if len(self.targets) == 1 else None

    @property
    def correction(self) -> MemoryTagCorrectionTarget | None:
        return self.corrections[0] if len(self.corrections) == 1 else None


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _scope_tuple(scope: RuntimeScope) -> tuple[str, str, str]:
    if not isinstance(scope, RuntimeScope) or scope.session is None or scope.visibility != "group":
        raise ValueError("a canonical group RuntimeScope is required")
    return scope.bot_id, scope.session.id, scope.visibility


def _normalize_tag_names(values: Sequence[Any]) -> tuple[str, ...]:
    names: list[str] = []
    seen: set[str] = set()
    for value in values:
        name = str(value or "").strip()
        key = name.casefold()
        if not name or key in seen:
            continue
        if len(name) > 200:
            raise ValueError("tag names must not exceed 200 characters")
        seen.add(key)
        names.append(name)
    return tuple(names)


def _automatic_tags(connection, scope: RuntimeScope, memory_id: int) -> list[dict[str, Any]]:
    rows = connection.execute(
        """SELECT t.id, t.name, t.tag_type, mt.position, mt.relevance
             FROM scoped_memory_tags mt
             JOIN scoped_tags t
               ON t.id=mt.tag_id AND t.bot_id=mt.bot_id
              AND t.session_id=mt.session_id AND t.visibility=mt.visibility
            WHERE mt.memory_id=? AND mt.bot_id=? AND mt.session_id=? AND mt.visibility=?
            ORDER BY mt.position, t.name""",
        (int(memory_id), *_scope_tuple(scope)),
    ).fetchall()
    return [
        {
            "id": int(row[0]),
            "name": str(row[1]),
            "tag_type": str(row[2] or "keyword"),
            "position": int(row[3] or 0),
            "relevance": float(row[4] if row[4] is not None else 1.0),
            "source": "automatic",
        }
        for row in rows
    ]


def _latest_active_correction(connection, scope: RuntimeScope, memory_id: int):
    return connection.execute(
        """SELECT correction_id, operation, requested_tags_json, before_tags_json,
                  after_tags_json, correction_revision, status, created_at, reason
             FROM scoped_memory_tag_corrections
            WHERE memory_id=? AND bot_id=? AND session_id=? AND visibility=? AND status='active'
            ORDER BY created_at DESC, rowid DESC LIMIT 1""",
        (int(memory_id), *_scope_tuple(scope)),
    ).fetchone()


def read_memory_tag_state(connection, *, scope: RuntimeScope, memory_id: int) -> dict[str, Any]:
    """Read automatic baseline, effective tags and the active manual override."""
    automatic = _automatic_tags(connection, scope, memory_id)
    correction = _latest_active_correction(connection, scope, memory_id)
    if correction is None:
        return {"automatic": automatic, "effective": list(automatic), "manual": None}
    names = tuple(json.loads(str(correction[4])))
    scoped = _scope_tuple(scope)
    tag_rows: dict[str, tuple[Any, ...]] = {}
    if names:
        placeholders = ",".join("?" for _ in names)
        rows = connection.execute(
            f"""SELECT id, name, tag_type FROM scoped_tags
                  WHERE bot_id=? AND session_id=? AND visibility=?
                    AND name IN ({placeholders})""",
            (*scoped, *names),
        ).fetchall()
        tag_rows = {str(row[1]).casefold(): row for row in rows}
    effective = []
    for position, name in enumerate(names, 1):
        row = tag_rows.get(str(name).casefold())
        effective.append(
            {
                "id": None if row is None else int(row[0]),
                "name": str(name),
                "tag_type": "custom" if row is None else str(row[2] or "custom"),
                "position": position,
                "relevance": 1.0,
                "source": "manual",
            }
        )
    return {
        "automatic": automatic,
        "effective": effective,
        "manual": {
            "correction_id": str(correction[0]),
            "operation": str(correction[1]),
            "requested_tags": json.loads(str(correction[2])),
            "before": json.loads(str(correction[3])),
            "tags": list(names),
            "revision": int(correction[5]),
            "status": str(correction[6]),
            "created_at": float(correction[7]),
            "reason": str(correction[8]),
        },
    }


def _preview_tag_change(
    current: Sequence[str], operation: str, requested: Sequence[Any]
) -> dict[str, Any]:
    normalized_operation = str(operation or "").strip().lower()
    if normalized_operation not in {"add", "remove", "replace"}:
        raise ValueError("operation must be add, remove, or replace")
    names = _normalize_tag_names(requested)
    if normalized_operation in {"add", "remove"} and not names:
        raise ValueError("at least one tag is required")
    current_names = _normalize_tag_names(current)
    if normalized_operation == "replace":
        after = names
    elif normalized_operation == "add":
        existing = {name.casefold() for name in current_names}
        after = (*current_names, *(name for name in names if name.casefold() not in existing))
    else:
        removed = {name.casefold() for name in names}
        after = tuple(name for name in current_names if name.casefold() not in removed)
    return {
        "operation": normalized_operation,
        "requested": list(names),
        "before": list(current_names),
        "after": list(after),
        "changed": tuple(current_names) != tuple(after),
    }


def _bump_memory_revision(
    connection, *, scope: RuntimeScope, memory_id: int, expected_revision: int
) -> dict[str, int]:
    bot_id, session_id, visibility = _scope_tuple(scope)
    cursor = connection.execute(
        """UPDATE memories SET version=version+1
             WHERE id=? AND version=? AND bot_id=? AND session_id=? AND visibility=?
               AND group_id=? AND resolution_state='resolved'
               AND COALESCE(quarantine, 0)=0""",
        (
            int(memory_id),
            int(expected_revision),
            bot_id,
            session_id,
            visibility,
            scope.session.conversation_id,
        ),
    )
    if int(cursor.rowcount or 0) != 1:
        raise MemoryRevisionConflict()
    return {
        "memory_id": int(memory_id),
        "previous_revision": int(expected_revision),
        "revision": int(expected_revision) + 1,
    }


def _ensure_manual_tags(connection, *, scope: RuntimeScope, names: Sequence[str], now: float) -> None:
    bot_id, session_id, visibility = _scope_tuple(scope)
    metadata = _canonical_json({"producer": "webui.memory_tag_correction", "manual": True})
    for name in names:
        connection.execute(
            """INSERT INTO scoped_tags (
                   bot_id, session_id, visibility, name, tag_type, description,
                   confidence, metadata, created_at, updated_at
               ) VALUES (?, ?, ?, ?, 'custom', '', 1.0, ?, ?, ?)
               ON CONFLICT(bot_id, session_id, visibility, name) DO NOTHING""",
            (bot_id, session_id, visibility, str(name), metadata, now, now),
        )


class MemoryMutationGateway:
    """Narrow WebUI gateway for exact Scope/ObjectRef revision mutations.

    ``ProductionWriteGateway`` remains the process wiring point. This facade only uses
    its writer-owned coordinator and appends operation/outbox rows in the same SQLite
    transaction as the canonical memory mutation.
    """

    def __init__(self, write_gateway: Any, *, clock: Any | None = None) -> None:
        coordinator = getattr(write_gateway, "coordinator", None)
        if coordinator is None:
            raise ValueError("write gateway coordinator is required")
        self._coordinator = coordinator
        self._clock = clock
        consumers = getattr(write_gateway, "_consumers", None)
        if isinstance(consumers, Mapping):
            self._consumer_names = tuple(sorted(str(name) for name in consumers))
        else:
            self._consumer_names = tuple(
                sorted(str(name) for name in getattr(coordinator, "_consumer_names", ()))
            )

    def _now(self) -> float:
        if self._clock is not None and callable(getattr(self._clock, "now", None)):
            return float(self._clock.now())
        return time.time()

    async def _commit(
        self,
        *,
        scope: RuntimeScope | None,
        command_type: str,
        actor: str,
        event_type: str,
        request_shape: Mapping[str, Any],
        mutate,
    ) -> MemoryMutationResult:
        request_hash = _digest(request_shape)
        idempotency_key = f"{command_type}:{request_hash}"
        operation_id = uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"wave-memory:{command_type}:{request_hash}",
        ).hex
        now = self._now()

        def transaction(connection):
            existing = connection.execute(
                "SELECT request_hash, status, result_json FROM write_operations WHERE idempotency_key=?",
                (idempotency_key,),
            ).fetchone()
            if existing is not None:
                if str(existing[0]) != request_hash:
                    raise ValueError("memory_mutation_idempotency_conflict")
                if existing[1] != "committed" or not existing[2]:
                    raise ValueError("memory_mutation_incomplete")
                payload = json.loads(str(existing[2]))
                return MemoryMutationResult(
                    operation_id=str(payload["operation_id"]),
                    targets=tuple(
                        MemoryMutationTarget(
                            memory_id=int(item["aggregate_id"]),
                            revision=int(item["aggregate_version"]),
                        )
                        for item in payload.get("entities", ())
                        if item.get("aggregate_kind") == "memory"
                    ),
                    corrections=tuple(
                        MemoryTagCorrectionTarget(
                            correction_id=str(item["aggregate_id"]),
                            revision=int(item["aggregate_version"]),
                            status=str(item.get("status") or "active"),
                        )
                        for item in payload.get("entities", ())
                        if item.get("aggregate_kind") == "memory_tag_correction"
                    ),
                )

            write_sequence = OutboxRepository.next_write_sequence(connection)
            connection.execute(
                """INSERT INTO write_operations(
                       operation_id, idempotency_key, request_hash, command_type, scope_json,
                       status, write_sequence, created_at)
                   VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)""",
                (
                    operation_id,
                    idempotency_key,
                    request_hash,
                    command_type,
                    _canonical_json(scope_to_dict(scope) if scope else {"kind": "system_maintenance"}),
                    write_sequence,
                    now,
                ),
            )
            records = tuple(mutate(connection, operation_id))
            entities = []
            effects = []
            for index, record in enumerate(records):
                scope_payload = record.get("scope") or scope_to_dict(scope)
                memory_id = int(record["memory_id"])
                revision = int(record["revision"])
                event_id = uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"wave-memory:{operation_id}:{index}",
                ).hex
                connection.execute(
                    """INSERT INTO domain_outbox(
                           event_id, operation_id, aggregate_kind, aggregate_id,
                           aggregate_version, event_type, payload_version, payload_json, created_at)
                       VALUES (?, ?, 'memory', ?, ?, ?, 1, ?, ?)""",
                    (
                        event_id,
                        operation_id,
                        str(memory_id),
                        revision,
                        event_type,
                        _canonical_json(
                            {
                                "memory_id": memory_id,
                                "previous_revision": int(record["previous_revision"]),
                                "scope": scope_payload,
                                **dict(record.get("event_payload") or {}),
                            }
                        ),
                        now,
                    ),
                )
                OutboxRepository.add_deliveries(
                    connection,
                    event_id,
                    self._consumer_names,
                    now,
                )
                entities.append(
                    {
                        "aggregate_kind": "memory",
                        "aggregate_id": str(memory_id),
                        "aggregate_version": revision,
                        "change_type": event_type.removeprefix("memory."),
                    }
                )
                if record.get("correction_id"):
                    entities.append(
                        {
                            "aggregate_kind": "memory_tag_correction",
                            "aggregate_id": str(record["correction_id"]),
                            "aggregate_version": int(record.get("correction_revision") or 1),
                            "change_type": str(record.get("correction_status") or "active"),
                            "status": str(record.get("correction_status") or "active"),
                        }
                    )
                effects.append(
                    {
                        "event_id": event_id,
                        "event_type": event_type,
                        "aggregate_kind": "memory",
                        "aggregate_id": str(memory_id),
                        "aggregate_version": revision,
                    }
                )
            result_payload = {
                "operation_id": operation_id,
                "committed_at": now,
                "write_sequence": write_sequence,
                "entities": entities,
                "effects": effects,
                "warnings": [],
            }
            connection.execute(
                """UPDATE write_operations
                      SET status='committed', result_json=?, committed_at=?
                    WHERE operation_id=?""",
                (_canonical_json(result_payload), now, operation_id),
            )
            return MemoryMutationResult(
                operation_id=operation_id,
                targets=tuple(
                    MemoryMutationTarget(
                        memory_id=int(record["memory_id"]),
                        revision=int(record["revision"]),
                    )
                    for record in records
                ),
                corrections=tuple(
                    MemoryTagCorrectionTarget(
                        correction_id=str(record["correction_id"]),
                        revision=int(record.get("correction_revision") or 1),
                        status=str(record.get("correction_status") or "active"),
                    )
                    for record in records
                    if record.get("correction_id")
                ),
            )

        return await self._coordinator.transaction(transaction, actor=actor)

    async def update_memory(
        self,
        *,
        scope: RuntimeScope,
        target: MemoryMutationTarget,
        content: Any = _UNSET,
        importance: Any = _UNSET,
    ) -> MemoryMutationResult:
        fields: dict[str, Any] = {}
        repo_fields: dict[str, Any] = {}
        if content is not _UNSET:
            fields["content"] = str(content or "")
            repo_fields["content"] = content
        if importance is not _UNSET:
            fields["importance"] = importance
            repo_fields["importance"] = importance
        if not fields:
            raise ValueError("at least one mutable memory field is required")
        request_shape = {
            "scope": scope_to_dict(scope),
            "target": target.to_dict(),
            "fields": fields,
        }
        return await self._commit(
            scope=scope,
            command_type="memory.webui.update.v1",
            actor="webui.memory.update",
            event_type="memory.updated",
            request_shape=request_shape,
            mutate=lambda connection, _operation_id: (
                MemoryRepo.update_scoped_memory(
                    connection,
                    scope=scope,
                    memory_id=target.memory_id,
                    expected_revision=target.revision,
                    **repo_fields,
                ),
            ),
        )

    async def update_memory_vector(
        self,
        *,
        scope: RuntimeScope,
        target: MemoryMutationTarget,
        vector: Any,
    ) -> MemoryMutationResult:
        vector_array = None if vector is None else [float(value) for value in vector]
        request_shape = {
            "scope": scope_to_dict(scope),
            "target": target.to_dict(),
            "vector_sha256": _digest(vector_array),
        }
        return await self._commit(
            scope=scope,
            command_type="memory.webui.reembed.v1",
            actor="durable.memory.reembed",
            event_type="memory.reembedded",
            request_shape=request_shape,
            mutate=lambda connection, _operation_id: (
                MemoryRepo.update_scoped_memory(
                    connection,
                    scope=scope,
                    memory_id=target.memory_id,
                    expected_revision=target.revision,
                    vector=vector_array,
                ),
            ),
        )

    async def correct_memory_tags(
        self,
        *,
        scope: RuntimeScope,
        target: MemoryMutationTarget,
        operation: str,
        tags: Sequence[Any],
        reason: str,
    ) -> MemoryMutationResult:
        normalized_reason = str(reason or "").strip()
        if not normalized_reason:
            raise ValueError("a correction reason is required")
        if len(normalized_reason) > 1000:
            raise ValueError("correction reason must not exceed 1000 characters")
        correction_id = uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"wave-memory:tag-correction:{_digest({'scope': scope_to_dict(scope), 'target': target.to_dict(), 'operation': operation, 'tags': list(tags), 'reason': normalized_reason})}",
        ).hex
        request_shape = {
            "scope": scope_to_dict(scope),
            "target": target.to_dict(),
            "operation": str(operation or "").strip().lower(),
            "tags": list(tags),
            "reason": normalized_reason,
            "correction_id": correction_id,
        }

        def mutate(connection, operation_id):
            state = read_memory_tag_state(connection, scope=scope, memory_id=target.memory_id)
            current_names = [item["name"] for item in state["effective"]]
            preview = _preview_tag_change(current_names, operation, tags)
            if not preview["changed"]:
                raise ValueError("tag correction would not change the effective tags")
            now = self._now()
            _ensure_manual_tags(connection, scope=scope, names=preview["after"], now=now)
            record = _bump_memory_revision(
                connection,
                scope=scope,
                memory_id=target.memory_id,
                expected_revision=target.revision,
            )
            bot_id, session_id, visibility = _scope_tuple(scope)
            connection.execute(
                """INSERT INTO scoped_memory_tag_corrections (
                       correction_id, operation_id, bot_id, session_id, visibility,
                       memory_id, operation, requested_tags_json, before_tags_json,
                       after_tags_json, memory_revision_before, memory_revision_after,
                       status, correction_revision, actor, reason, created_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', 1, ?, ?, ?)""",
                (
                    correction_id,
                    operation_id,
                    bot_id,
                    session_id,
                    visibility,
                    int(target.memory_id),
                    preview["operation"],
                    _canonical_json(preview["requested"]),
                    _canonical_json(preview["before"]),
                    _canonical_json(preview["after"]),
                    int(target.revision),
                    int(record["revision"]),
                    "webui.memory.tags.correct",
                    normalized_reason,
                    now,
                ),
            )
            projection_revision = rebuild_memory_effective_tags(
                connection,
                scope=scope,
                memory_id=target.memory_id,
                now=now,
            )
            record.update(
                {
                    "correction_id": correction_id,
                    "correction_revision": 1,
                    "correction_status": "active",
                    "event_payload": {
                        "correction_id": correction_id,
                        "operation": preview["operation"],
                        "requested_tags": preview["requested"],
                        "before_tags": preview["before"],
                        "after_tags": preview["after"],
                        "reason": normalized_reason,
                        "projection_revision": projection_revision,
                        "projection_status": "ready",
                    },
                }
            )
            return (record,)

        return await self._commit(
            scope=scope,
            command_type="memory.webui.tags.correct.v1",
            actor="webui.memory.tags.correct",
            event_type="memory.tags_corrected",
            request_shape=request_shape,
            mutate=mutate,
        )

    async def undo_memory_tag_correction(
        self,
        *,
        scope: RuntimeScope,
        target: MemoryMutationTarget,
        correction_id: str,
        correction_revision: int,
        reason: str,
    ) -> MemoryMutationResult:
        normalized_id = str(correction_id or "").strip()
        normalized_reason = str(reason or "").strip()
        if not normalized_id or not normalized_reason:
            raise ValueError("correction ref and undo reason are required")
        request_shape = {
            "scope": scope_to_dict(scope),
            "target": target.to_dict(),
            "correction_id": normalized_id,
            "correction_revision": int(correction_revision),
            "reason": normalized_reason,
        }

        def mutate(connection, operation_id):
            bot_id, session_id, visibility = _scope_tuple(scope)
            row = connection.execute(
                """SELECT correction_revision, status, before_tags_json, after_tags_json
                     FROM scoped_memory_tag_corrections
                    WHERE correction_id=? AND memory_id=? AND bot_id=?
                      AND session_id=? AND visibility=?""",
                (
                    normalized_id,
                    int(target.memory_id),
                    bot_id,
                    session_id,
                    visibility,
                ),
            ).fetchone()
            if row is None or str(row[1]) != "active" or int(row[0]) != int(correction_revision):
                raise MemoryRevisionConflict()
            record = _bump_memory_revision(
                connection,
                scope=scope,
                memory_id=target.memory_id,
                expected_revision=target.revision,
            )
            now = self._now()
            next_revision = int(correction_revision) + 1
            cursor = connection.execute(
                """UPDATE scoped_memory_tag_corrections
                      SET status='undone', correction_revision=?, undone_at=?,
                          undone_by_operation_id=?
                    WHERE correction_id=? AND correction_revision=? AND status='active'""",
                (
                    next_revision,
                    now,
                    operation_id,
                    normalized_id,
                    int(correction_revision),
                ),
            )
            if int(cursor.rowcount or 0) != 1:
                raise MemoryRevisionConflict()
            projection_revision = rebuild_memory_effective_tags(
                connection,
                scope=scope,
                memory_id=target.memory_id,
                now=now,
            )
            record.update(
                {
                    "correction_id": normalized_id,
                    "correction_revision": next_revision,
                    "correction_status": "undone",
                    "event_payload": {
                        "correction_id": normalized_id,
                        "before_tags": json.loads(str(row[3])),
                        "after_tags": json.loads(str(row[2])),
                        "reason": normalized_reason,
                        "undone_correction_revision": int(correction_revision),
                        "projection_revision": projection_revision,
                        "projection_status": "ready",
                    },
                }
            )
            return (record,)

        return await self._commit(
            scope=scope,
            command_type="memory.webui.tags.undo.v1",
            actor="webui.memory.tags.undo",
            event_type="memory.tags_correction_undone",
            request_shape=request_shape,
            mutate=mutate,
        )

    async def delete_memories(
        self,
        *,
        scope: RuntimeScope,
        targets: Sequence[MemoryMutationTarget],
    ) -> MemoryMutationResult:
        normalized = tuple(
            dict.fromkeys(
                MemoryMutationTarget(int(target.memory_id), int(target.revision))
                for target in targets
            )
        )
        request_shape = {
            "scope": scope_to_dict(scope),
            "targets": [target.to_dict() for target in normalized],
        }
        return await self._commit(
            scope=scope,
            command_type="memory.webui.delete.v1",
            actor="webui.memory.delete",
            event_type="memory.deleted",
            request_shape=request_shape,
            mutate=lambda connection, _operation_id: MemoryRepo.delete_scoped_memories(
                connection,
                scope=scope,
                expected_revisions={
                    target.memory_id: target.revision for target in normalized
                },
            ),
        )

    async def prune_automatic_memories(self, *, policy, limit: int = 100, scan_placeholders: bool = True) -> MemoryMutationResult:
        """Select and delete retention candidates atomically with outbox events.

        Args:
            policy: Automatic-memory ages and capacities.
            limit: Maximum deletions per writer transaction.
            scan_placeholders: Refresh empty-media IDs at the start of the sweep.

        Returns:
            Deleted memory identities for diagnostics.
        """
        try:
            from ..domain.scope import SessionRef
            from ..engine.db.retention_repo import retention_candidates
        except ImportError:
            from domain.scope import SessionRef
            from engine.db.retention_repo import retention_candidates

        now = self._now()

        def mutate(connection, _operation_id):
            rows = retention_candidates(connection, policy, now=now, limit=limit,
                                        scan_placeholders=scan_placeholders)
            groups = {}
            for memory_id, revision, bot_id, session_id, visibility, group_id in rows:
                parts = session_id.split(":", 2)
                if len(parts) != 3 or parts[1] != visibility or parts[2] != group_id:
                    continue
                key = (bot_id, session_id, visibility, group_id, parts[0])
                groups.setdefault(key, {})[memory_id] = revision
            records = []
            for (bot_id, session_id, visibility, group_id, platform_id), targets in groups.items():
                scope = RuntimeScope(
                    bot_id=bot_id, visibility=visibility,
                    session=SessionRef(id=session_id, platform_id=platform_id,
                                       kind=visibility, conversation_id=group_id),
                )
                deleted = MemoryRepo.delete_scoped_memories(
                    connection, scope=scope, expected_revisions=targets, allow_quarantine=True,
                )
                records.extend({**record, "scope": scope_to_dict(scope)} for record in deleted)
            return records

        return await self._commit(
            scope=None, command_type="memory.retention.delete.v1", actor="memory.retention",
            event_type="memory.deleted", request_shape={"batch_id": uuid.uuid4().hex, "limit": limit},
            mutate=mutate,
        )


__all__ = [
    "MemoryMutationGateway",
    "MemoryMutationResult",
    "MemoryMutationTarget",
    "MemoryTagCorrectionTarget",
    "read_memory_tag_state",
]
