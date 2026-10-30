"""Transaction-scoped durable job request and run repository."""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


class JobRepositoryError(RuntimeError):
    reason_code = "job_repository_error"


class JobRequestConflictError(JobRepositoryError):
    reason_code = "job_request_conflict"


class JobRunConflictError(JobRepositoryError):
    reason_code = "job_run_conflict"


class JobLeaseLostError(JobRepositoryError):
    reason_code = "job_lease_lost"


class JobStateError(JobRepositoryError):
    reason_code = "job_state_invalid"


@dataclass(frozen=True)
class JobRequest:
    request_id: str
    idempotency_key: str
    kind: str
    scope: Any
    payload: Any
    created_at: float


@dataclass(frozen=True)
class JobRun:
    run_id: str
    request_id: str
    schedule_slot: str
    cursor_generation: int
    status: str
    cursor: Any
    attempt: int
    lease_owner: str | None
    lease_until: float | None
    progress: Any
    result: Any | None
    error_code: str | None
    error_message: str | None
    created_at: float
    updated_at: float


class JobRepository:
    """Pure SQLite repository; callers own transaction and commit boundaries."""

    PENDING = "pending"
    RUNNING = "running"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLED = "cancelled"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TERMINAL_STATUSES = frozenset({CANCELLED, SUCCEEDED, FAILED})

    _REQUEST_COLUMNS = "request_id, idempotency_key, kind, scope_json, payload_json, created_at"
    _RUN_COLUMNS = (
        "run_id, request_id, schedule_slot, cursor_generation, status, cursor_json, attempt, "
        "lease_owner, lease_until, progress_json, result_json, error_code, error_message, "
        "created_at, updated_at"
    )

    @staticmethod
    def migrate(connection: sqlite3.Connection) -> None:
        """Create the Stage 3 tables/index without committing the caller's transaction."""

        statements = (
            """CREATE TABLE IF NOT EXISTS job_requests (
                   request_id TEXT PRIMARY KEY,
                   idempotency_key TEXT NOT NULL UNIQUE,
                   kind TEXT NOT NULL,
                   scope_json TEXT NOT NULL,
                   payload_json TEXT NOT NULL,
                   created_at REAL NOT NULL
               )""",
            """CREATE TABLE IF NOT EXISTS background_job_runs (
                   run_id TEXT PRIMARY KEY,
                   request_id TEXT NOT NULL REFERENCES job_requests(request_id) ON DELETE CASCADE,
                   schedule_slot TEXT NOT NULL,
                   cursor_generation INTEGER NOT NULL DEFAULT 0,
                   status TEXT NOT NULL,
                   cursor_json TEXT NOT NULL DEFAULT '{}',
                   attempt INTEGER NOT NULL DEFAULT 0,
                   lease_owner TEXT,
                   lease_until REAL,
                   progress_json TEXT NOT NULL DEFAULT '{}',
                   result_json TEXT,
                   error_code TEXT,
                   error_message TEXT,
                   created_at REAL NOT NULL,
                   updated_at REAL NOT NULL,
                   UNIQUE(request_id, schedule_slot, cursor_generation)
               )""",
            """CREATE INDEX IF NOT EXISTS idx_background_job_runs_ready
                   ON background_job_runs(status, lease_until, updated_at)""",
            """CREATE INDEX IF NOT EXISTS idx_background_job_runs_request
                   ON background_job_runs(request_id, created_at)""",
            """CREATE INDEX IF NOT EXISTS idx_background_job_runs_retention
                   ON background_job_runs(status, updated_at)""",
        )
        for statement in statements:
            connection.execute(statement)

    @staticmethod
    def _text(value: Any, field: str) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError(f"{field} must not be empty")
        return text

    @staticmethod
    def _json_ready(value: Any) -> Any:
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            return to_dict()
        if isinstance(value, Mapping):
            return dict(value)
        return value

    @classmethod
    def _dump(cls, value: Any) -> str:
        return json.dumps(
            cls._json_ready(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @staticmethod
    def _load(value: str | None) -> Any | None:
        return None if value is None else json.loads(value)

    @classmethod
    def _request_from_row(cls, row: Any) -> JobRequest:
        return JobRequest(
            request_id=str(row[0]),
            idempotency_key=str(row[1]),
            kind=str(row[2]),
            scope=cls._load(row[3]),
            payload=cls._load(row[4]),
            created_at=float(row[5]),
        )

    @classmethod
    def _run_from_row(cls, row: Any) -> JobRun:
        return JobRun(
            run_id=str(row[0]),
            request_id=str(row[1]),
            schedule_slot=str(row[2]),
            cursor_generation=int(row[3]),
            status=str(row[4]),
            cursor=cls._load(row[5]),
            attempt=int(row[6]),
            lease_owner=None if row[7] is None else str(row[7]),
            lease_until=None if row[8] is None else float(row[8]),
            progress=cls._load(row[9]),
            result=cls._load(row[10]),
            error_code=None if row[11] is None else str(row[11]),
            error_message=None if row[12] is None else str(row[12]),
            created_at=float(row[13]),
            updated_at=float(row[14]),
        )

    @classmethod
    def get_request(
        cls, connection: sqlite3.Connection, request_id: str
    ) -> JobRequest | None:
        row = connection.execute(
            f"SELECT {cls._REQUEST_COLUMNS} FROM job_requests WHERE request_id=?",
            (str(request_id),),
        ).fetchone()
        return None if row is None else cls._request_from_row(row)

    @classmethod
    def get_request_by_idempotency_key(
        cls, connection: sqlite3.Connection, idempotency_key: str
    ) -> JobRequest | None:
        row = connection.execute(
            f"SELECT {cls._REQUEST_COLUMNS} FROM job_requests WHERE idempotency_key=?",
            (str(idempotency_key),),
        ).fetchone()
        return None if row is None else cls._request_from_row(row)

    @classmethod
    def create_request(
        cls,
        connection: sqlite3.Connection,
        *,
        idempotency_key: str,
        kind: str,
        scope: Any,
        payload: Any,
        created_at: float,
        request_id: str | None = None,
    ) -> JobRequest:
        """Create an idempotent request, or return the identical existing request."""

        key = cls._text(idempotency_key, "idempotency_key")
        request_kind = cls._text(kind, "kind")
        scope_json = cls._dump(scope)
        payload_json = cls._dump(payload)
        identifier = cls._text(
            request_id or uuid.uuid5(uuid.NAMESPACE_URL, f"wave-memory:job-request:{key}").hex,
            "request_id",
        )
        connection.execute(
            """INSERT INTO job_requests(
                   request_id, idempotency_key, kind, scope_json, payload_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT DO NOTHING""",
            (identifier, key, request_kind, scope_json, payload_json, float(created_at)),
        )
        row = connection.execute(
            f"SELECT {cls._REQUEST_COLUMNS} FROM job_requests WHERE idempotency_key=?",
            (key,),
        ).fetchone()
        if row is None:
            conflicting = cls.get_request(connection, identifier)
            if conflicting is not None:
                raise JobRequestConflictError(
                    f"request id {identifier!r} already belongs to idempotency key "
                    f"{conflicting.idempotency_key!r}"
                )
            raise JobRepositoryError("job request insert did not produce a row")
        existing = cls._request_from_row(row)
        if (
            existing.kind != request_kind
            or cls._dump(existing.scope) != scope_json
            or cls._dump(existing.payload) != payload_json
        ):
            raise JobRequestConflictError(
                f"idempotency key {key!r} already belongs to a different job request"
            )
        return existing

    ensure_request = create_request
    get_or_create_request = create_request
    register_request = create_request

    @classmethod
    def get_run(cls, connection: sqlite3.Connection, run_id: str) -> JobRun | None:
        row = connection.execute(
            f"SELECT {cls._RUN_COLUMNS} FROM background_job_runs WHERE run_id=?",
            (str(run_id),),
        ).fetchone()
        return None if row is None else cls._run_from_row(row)

    @classmethod
    def schedule_run(
        cls,
        connection: sqlite3.Connection,
        *,
        request_id: str,
        schedule_slot: str,
        cursor_generation: int = 0,
        cursor: Any | None = None,
        created_at: float,
        run_id: str | None = None,
        reschedule_terminal: bool = False,
    ) -> JobRun:
        """Idempotently schedule one run for a request/slot/generation tuple.

        ``cursor`` is mutable checkpoint state and therefore never participates in
        schedule identity. Maintenance callers may request a new generation only
        after the latest run for the slot has reached a terminal state.
        """

        request_identifier = cls._text(request_id, "request_id")
        slot = cls._text(schedule_slot, "schedule_slot")
        generation = int(cursor_generation)
        if generation < 0:
            raise ValueError("cursor_generation must be non-negative")
        if cls.get_request(connection, request_identifier) is None:
            raise ValueError(f"unknown job request: {request_identifier}")

        if reschedule_terminal:
            latest_row = connection.execute(
                f"SELECT {cls._RUN_COLUMNS} FROM background_job_runs "
                "WHERE request_id=? AND schedule_slot=? AND cursor_generation>=? "
                "ORDER BY cursor_generation DESC LIMIT 1",
                (request_identifier, slot, generation),
            ).fetchone()
            if latest_row is not None:
                latest = cls._run_from_row(latest_row)
                if latest.status not in cls.TERMINAL_STATUSES:
                    return latest
                generation = latest.cursor_generation + 1

        cursor_json = cls._dump({} if cursor is None else cursor)
        identifier = cls._text(
            run_id
            or uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"wave-memory:job-run:{request_identifier}:{slot}:{generation}",
            ).hex,
            "run_id",
        )
        timestamp = float(created_at)
        connection.execute(
            """INSERT INTO background_job_runs(
                   run_id, request_id, schedule_slot, cursor_generation, status,
                   cursor_json, progress_json, created_at, updated_at)
               VALUES (?, ?, ?, ?, 'pending', ?, '{}', ?, ?)
               ON CONFLICT DO NOTHING""",
            (identifier, request_identifier, slot, generation, cursor_json, timestamp, timestamp),
        )
        row = connection.execute(
            f"SELECT {cls._RUN_COLUMNS} FROM background_job_runs "
            "WHERE request_id=? AND schedule_slot=? AND cursor_generation=?",
            (request_identifier, slot, generation),
        ).fetchone()
        if row is None:
            conflicting = cls.get_run(connection, identifier)
            if conflicting is not None:
                raise JobRunConflictError(
                    f"run id {identifier!r} already belongs to request/slot/generation "
                    f"{conflicting.request_id!r}/{conflicting.schedule_slot!r}/"
                    f"{conflicting.cursor_generation}"
                )
            raise JobRepositoryError("job run insert did not produce a row")
        return cls._run_from_row(row)

    create_run = schedule_run
    schedule = schedule_run

    @classmethod
    def _claim_row(
        cls,
        connection: sqlite3.Connection,
        row: Any,
        *,
        now: float,
        lease_owner: str,
        lease_seconds: float,
    ) -> JobRun | None:
        run_id = str(row[0])
        owner = cls._text(lease_owner, "lease_owner")
        timestamp = float(now)
        lease_until = timestamp + max(0.001, float(lease_seconds))
        changed = connection.execute(
            """UPDATE background_job_runs
               SET status='running', attempt=attempt+1, lease_owner=?, lease_until=?,
                   error_code=NULL, error_message=NULL, updated_at=?
               WHERE run_id=? AND (
                   status='pending'
                   OR (status='running' AND COALESCE(lease_until, 0)<=?)
               )""",
            (owner, lease_until, timestamp, run_id, timestamp),
        ).rowcount
        return None if changed != 1 else cls.get_run(connection, run_id)

    @classmethod
    def claim_next(
        cls,
        connection: sqlite3.Connection,
        *,
        now: float,
        lease_owner: str,
        lease_seconds: float = 30.0,
        kinds: Iterable[str] | None = None,
        excluded_run_ids: Iterable[str] = (),
    ) -> JobRun | None:
        """Claim pending work or atomically take over a run whose lease expired."""

        params: list[Any] = [float(now)]
        where = [
            "(r.status='pending' OR (r.status='running' AND COALESCE(r.lease_until, 0)<=?))"
        ]
        normalized_kinds = tuple(dict.fromkeys(str(kind) for kind in (kinds or ())))
        if normalized_kinds:
            placeholders = ",".join("?" for _ in normalized_kinds)
            where.append(f"q.kind IN ({placeholders})")
            params.extend(normalized_kinds)
        excluded = tuple(dict.fromkeys(str(run_id) for run_id in excluded_run_ids))
        if excluded:
            placeholders = ",".join("?" for _ in excluded)
            where.append(f"r.run_id NOT IN ({placeholders})")
            params.extend(excluded)
        rows = connection.execute(
            f"""SELECT r.run_id
                  FROM background_job_runs r
                  JOIN job_requests q ON q.request_id=r.request_id
                 WHERE {' AND '.join(where)}
                 ORDER BY r.created_at, r.run_id LIMIT 1""",
            tuple(params),
        ).fetchall()
        for row in rows:
            claimed = cls._claim_row(
                connection,
                row,
                now=float(now),
                lease_owner=lease_owner,
                lease_seconds=lease_seconds,
            )
            if claimed is not None:
                return claimed
        return None

    claim = claim_next

    @classmethod
    def claim_run(
        cls,
        connection: sqlite3.Connection,
        run_id: str,
        *,
        now: float,
        lease_owner: str,
        lease_seconds: float = 30.0,
    ) -> JobRun | None:
        row = connection.execute(
            "SELECT run_id FROM background_job_runs WHERE run_id=?",
            (str(run_id),),
        ).fetchone()
        if row is None:
            return None
        return cls._claim_row(
            connection,
            row,
            now=float(now),
            lease_owner=lease_owner,
            lease_seconds=lease_seconds,
        )

    @classmethod
    def _require_active_lease(
        cls,
        connection: sqlite3.Connection,
        *,
        run_id: str,
        lease_owner: str,
        now: float,
        statuses: tuple[str, ...] = (RUNNING,),
    ) -> JobRun:
        run = cls.get_run(connection, run_id)
        if (
            run is None
            or run.status not in statuses
            or run.lease_owner != str(lease_owner)
            or run.lease_until is None
            or run.lease_until <= float(now)
        ):
            raise JobLeaseLostError(f"active lease not held for job run {run_id!r}")
        return run

    @classmethod
    def renew_lease(
        cls,
        connection: sqlite3.Connection,
        run_id: str,
        *,
        lease_owner: str,
        now: float,
        lease_seconds: float = 30.0,
    ) -> JobRun:
        cls._require_active_lease(
            connection, run_id=run_id, lease_owner=lease_owner, now=now
        )
        timestamp = float(now)
        connection.execute(
            """UPDATE background_job_runs SET lease_until=?, updated_at=?
               WHERE run_id=? AND lease_owner=? AND status='running'""",
            (timestamp + max(0.001, float(lease_seconds)), timestamp, str(run_id), str(lease_owner)),
        )
        run = cls.get_run(connection, run_id)
        assert run is not None
        return run

    @classmethod
    def update_progress(
        cls,
        connection: sqlite3.Connection,
        run_id: str,
        *,
        lease_owner: str,
        now: float,
        progress: Any | None = None,
        cursor: Any | None = None,
        lease_seconds: float = 30.0,
    ) -> JobRun:
        """Persist progress/cursor while the caller still owns a live lease."""

        run = cls._require_active_lease(
            connection, run_id=run_id, lease_owner=lease_owner, now=now
        )
        progress_json = cls._dump(run.progress if progress is None else progress)
        cursor_json = cls._dump(run.cursor if cursor is None else cursor)
        connection.execute(
            """UPDATE background_job_runs
               SET progress_json=?, cursor_json=?, lease_until=?, updated_at=?
               WHERE run_id=? AND lease_owner=? AND status='running'""",
            (
                progress_json,
                cursor_json,
                float(now) + max(1.0, float(lease_seconds)),
                float(now),
                str(run_id),
                str(lease_owner),
            ),
        )
        updated = cls.get_run(connection, run_id)
        assert updated is not None
        return updated

    save_progress = update_progress
    checkpoint = update_progress

    @classmethod
    def request_cancel(
        cls, connection: sqlite3.Connection, run_id: str, *, now: float
    ) -> JobRun | None:
        """Cancel queued work immediately or signal a leased worker to stop."""

        run = cls.get_run(connection, run_id)
        if run is None or run.status in cls.TERMINAL_STATUSES:
            return run
        timestamp = float(now)
        if run.status == cls.PENDING:
            connection.execute(
                """UPDATE background_job_runs
                   SET status='cancelled', lease_owner=NULL, lease_until=NULL, updated_at=?
                   WHERE run_id=? AND status='pending'""",
                (timestamp, str(run_id)),
            )
        elif run.status == cls.RUNNING:
            connection.execute(
                """UPDATE background_job_runs SET status='cancel_requested', updated_at=?
                   WHERE run_id=? AND status='running'""",
                (timestamp, str(run_id)),
            )
        return cls.get_run(connection, run_id)

    cancel = request_cancel
    cancel_run = request_cancel

    @classmethod
    def mark_cancelled(
        cls,
        connection: sqlite3.Connection,
        run_id: str,
        *,
        now: float,
        lease_owner: str | None = None,
    ) -> JobRun:
        run = cls.get_run(connection, run_id)
        if run is None:
            raise JobStateError(f"unknown job run: {run_id}")
        if run.status == cls.CANCELLED:
            return run
        if run.status == cls.CANCEL_REQUESTED:
            if lease_owner is None or run.lease_owner != str(lease_owner):
                raise JobLeaseLostError(f"lease not held for cancelled job run {run_id!r}")
        elif run.status != cls.PENDING:
            raise JobStateError(f"job run {run_id!r} cannot be cancelled from {run.status!r}")
        connection.execute(
            """UPDATE background_job_runs
               SET status='cancelled', lease_owner=NULL, lease_until=NULL, updated_at=?
               WHERE run_id=?""",
            (float(now), str(run_id)),
        )
        updated = cls.get_run(connection, run_id)
        assert updated is not None
        return updated

    @classmethod
    def cancellation_requested(cls, connection: sqlite3.Connection, run_id: str) -> bool:
        row = connection.execute(
            "SELECT status FROM background_job_runs WHERE run_id=?",
            (str(run_id),),
        ).fetchone()
        return row is not None and str(row[0]) == cls.CANCEL_REQUESTED

    @classmethod
    def mark_succeeded(
        cls,
        connection: sqlite3.Connection,
        run_id: str,
        *,
        lease_owner: str,
        now: float,
        result: Any | None = None,
        progress: Any | None = None,
        cursor: Any | None = None,
    ) -> JobRun:
        run = cls._require_active_lease(
            connection, run_id=run_id, lease_owner=lease_owner, now=now
        )
        connection.execute(
            """UPDATE background_job_runs
               SET status='succeeded', cursor_json=?, progress_json=?, result_json=?,
                   error_code=NULL, error_message=NULL, lease_owner=NULL, lease_until=NULL,
                   updated_at=?
               WHERE run_id=? AND lease_owner=? AND status='running'""",
            (
                cls._dump(run.cursor if cursor is None else cursor),
                cls._dump(run.progress if progress is None else progress),
                None if result is None else cls._dump(result),
                float(now),
                str(run_id),
                str(lease_owner),
            ),
        )
        updated = cls.get_run(connection, run_id)
        assert updated is not None
        return updated

    succeed = mark_succeeded
    complete = mark_succeeded

    @classmethod
    def mark_failed(
        cls,
        connection: sqlite3.Connection,
        run_id: str,
        *,
        lease_owner: str,
        now: float,
        error_code: str,
        error_message: str,
        progress: Any | None = None,
        cursor: Any | None = None,
    ) -> JobRun:
        run = cls._require_active_lease(
            connection,
            run_id=run_id,
            lease_owner=lease_owner,
            now=now,
            statuses=(cls.RUNNING, cls.CANCEL_REQUESTED),
        )
        connection.execute(
            """UPDATE background_job_runs
               SET status='failed', cursor_json=?, progress_json=?, result_json=NULL,
                   error_code=?, error_message=?, lease_owner=NULL, lease_until=NULL,
                   updated_at=?
               WHERE run_id=? AND lease_owner=? AND status IN ('running','cancel_requested')""",
            (
                cls._dump(run.cursor if cursor is None else cursor),
                cls._dump(run.progress if progress is None else progress),
                cls._text(error_code, "error_code"),
                str(error_message)[:4000],
                float(now),
                str(run_id),
                str(lease_owner),
            ),
        )
        updated = cls.get_run(connection, run_id)
        assert updated is not None
        return updated

    fail = mark_failed

    @classmethod
    def release_for_retry(
        cls,
        connection: sqlite3.Connection,
        run_id: str,
        *,
        lease_owner: str,
        now: float,
        reason: str = "worker_stopped",
    ) -> JobRun:
        """Release an owned live lease so graceful shutdown can replay immediately."""
        run = cls.get_run(connection, run_id)
        if run is None:
            raise JobStateError(f"unknown job run: {run_id}")
        if run.lease_owner != str(lease_owner) or run.status not in {
            cls.RUNNING,
            cls.CANCEL_REQUESTED,
        }:
            raise JobLeaseLostError(run_id)
        next_status = cls.CANCELLED if run.status == cls.CANCEL_REQUESTED else cls.PENDING
        connection.execute(
            """UPDATE background_job_runs
               SET status=?, lease_owner=NULL, lease_until=NULL,
                   error_code=?, error_message=?, updated_at=?
               WHERE run_id=? AND lease_owner=?""",
            (
                next_status,
                str(reason),
                "worker stopped before completion; run released for replay",
                float(now),
                str(run_id),
                str(lease_owner),
            ),
        )
        updated = cls.get_run(connection, run_id)
        assert updated is not None
        return updated

    @classmethod
    def recover_expired_leases(
        cls, connection: sqlite3.Connection, *, now: float
    ) -> int:
        """Requeue crashed workers and finish expired cancellation requests."""

        timestamp = float(now)
        cancelled = connection.execute(
            """UPDATE background_job_runs
               SET status='cancelled', lease_owner=NULL, lease_until=NULL, updated_at=?
               WHERE status='cancel_requested' AND COALESCE(lease_until, 0)<=?""",
            (timestamp, timestamp),
        ).rowcount
        requeued = connection.execute(
            """UPDATE background_job_runs
               SET status='pending', lease_owner=NULL, lease_until=NULL,
                   error_code='lease_expired', error_message='worker lease expired; run requeued',
                   updated_at=?
               WHERE status='running' AND COALESCE(lease_until, 0)<=?""",
            (timestamp, timestamp),
        ).rowcount
        return int(cancelled) + int(requeued)

    recover_leases = recover_expired_leases
    recover_expired = recover_expired_leases

    @classmethod
    def list_runs(
        cls,
        connection: sqlite3.Connection,
        *,
        request_id: str | None = None,
        statuses: Iterable[str] | None = None,
        limit: int = 100,
    ) -> tuple[JobRun, ...]:
        where: list[str] = []
        params: list[Any] = []
        if request_id is not None:
            where.append("request_id=?")
            params.append(str(request_id))
        normalized_statuses = tuple(dict.fromkeys(str(status) for status in (statuses or ())))
        if normalized_statuses:
            placeholders = ",".join("?" for _ in normalized_statuses)
            where.append(f"status IN ({placeholders})")
            params.extend(normalized_statuses)
        predicate = "" if not where else " WHERE " + " AND ".join(where)
        params.append(max(1, min(int(limit), 1000)))
        rows = connection.execute(
            f"SELECT {cls._RUN_COLUMNS} FROM background_job_runs{predicate} "
            "ORDER BY created_at DESC, run_id DESC LIMIT ?",
            tuple(params),
        ).fetchall()
        return tuple(cls._run_from_row(row) for row in rows)

    @classmethod
    def next_claimable_at(cls, connection: sqlite3.Connection) -> float | None:
        row = connection.execute(
            """SELECT MIN(CASE WHEN status='running' THEN lease_until ELSE updated_at END)
                 FROM background_job_runs
                WHERE status IN ('pending','running')"""
        ).fetchone()
        return None if row is None or row[0] is None else float(row[0])


DurableJobRepository = JobRepository

__all__ = [
    "DurableJobRepository",
    "JobLeaseLostError",
    "JobRepository",
    "JobRepositoryError",
    "JobRequest",
    "JobRequestConflictError",
    "JobRun",
    "JobRunConflictError",
    "JobStateError",
]
