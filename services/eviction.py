"""Periodic, bounded retention for automatic memories and maintenance history."""

from __future__ import annotations

import asyncio
import time

from astrbot.api import logger

from ..engine.database import WaveMemoryDB
from ..engine.db.retention_repo import RetentionPolicy, prune_job_history
from ..engine.vector_index import VectorIndex
from .memory_mutations import MemoryMutationGateway


class EvictionService:
    """Trim stale automatic memories while protecting knowledge and evidence."""

    def __init__(
        self,
        db: WaveMemoryDB,
        memory_index: VectorIndex,
        eviction_interval_hours: float = 6.0,
        write_gateway=None,
        retention_settings=None,
    ):
        self.db = db
        self.memory_index = memory_index
        self.interval = max(0.25, eviction_interval_hours) * 3600
        self.write_gateway = write_gateway
        self._task: asyncio.Task | None = None
        self._running = False
        self._stats: dict = {}
        # Keep delivered history for seven days for diagnosis and idempotent replay.
        self.outbox_retention_days = 7.0
        self.policy = RetentionPolicy.from_settings(retention_settings or {})
        self._prune_lock = asyncio.Lock()

    def start(self, supervisor=None):
        self._running = True
        if supervisor is None:
            self._task = asyncio.create_task(self._loop())
        else:
            self._task = supervisor.start(
                "wave-memory:eviction", self._loop(), owner="eviction"
            )
        logger.info(
            "[WaveMemory] Retention started (history=%dd, noise=%dd, low-value=%dd, automatic=%d, per-scope=%d)",
            int(self.outbox_retention_days),
            self.policy.noise_days,
            self.policy.low_value_days,
            self.policy.max_memories,
            self.policy.max_per_scope,
        )

    def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()

    async def _loop(self):
        """Start after startup settles, then apply the configured interval."""
        await asyncio.sleep(60)
        while self._running:
            try:
                await self.prune_outbox_history()
                memories = await self.prune_memories()
                jobs = await self.prune_jobs()
                delay = (
                    60
                    if memories["memories"] >= 2000 or jobs["job_runs"] >= 100000
                    else self.interval
                )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"[EvictionService] Error: {e}")
                delay = min(self.interval, 600)
            await asyncio.sleep(delay)

    async def prune_outbox_history(self) -> dict | None:
        """Prune delivered outbox history through the writer coordinator.

        Returns:
            Cleanup counters, or None when no coordinator is available.
        """
        coordinator = getattr(self.write_gateway, "coordinator", None)
        prune = getattr(coordinator, "prune_outbox_history", None)
        if not callable(prune):
            return None
        result = await prune(retention_days=self.outbox_retention_days)
        if any(
            result.get(key)
            for key in ("events", "deliveries", "operations", "projections")
        ):
            logger.info("[EvictionService] outbox history retention: %s", result)
        self._stats["outbox_pruned"] = self._stats.get("outbox_pruned", 0) + int(
            result.get("events", 0)
        )
        return result

    @property
    def stats(self) -> dict:
        return dict(self._stats)

    async def prune_memories(self) -> dict:
        """Trim at most 2,000 memories, yielding between short writer transactions.

        Returns:
            Number of deleted automatic memories and completed batches.
        """
        async with self._prune_lock:
            mutations = MemoryMutationGateway(self.write_gateway)
            result = {"memories": 0, "batches": 0}
            for batch_index in range(20):
                batch = await mutations.prune_automatic_memories(
                    policy=self.policy,
                    limit=100,
                    scan_placeholders=batch_index == 0,
                )
                deleted = len(batch.targets)
                result["memories"] += deleted
                result["batches"] += 1
                if deleted == 0:
                    break
                # Outbox delivery keeps HNSW and other derived views recoverable.
                await asyncio.sleep(0.1)
            self._stats["last_memory_prune"] = result
            self._stats["memories_pruned"] = (
                self._stats.get("memories_pruned", 0) + result["memories"]
            )
            if result["memories"]:
                logger.info("[EvictionService] automatic memory retention: %s", result)
            return result

    async def prune_jobs(self) -> dict:
        """Remove terminal job history without deleting active runs or slot cursors.

        Returns:
            Deleted history count and number of batches.
        """
        before = time.time() - self.policy.job_days * 86400
        result = {"job_runs": 0, "batches": 0}
        for _ in range(100):
            count = await self.write_gateway.coordinator.transaction(
                lambda connection: prune_job_history(
                    connection, before=before, limit=1000
                )
            )
            result["job_runs"] += count
            result["batches"] += 1
            if count == 0:
                break
            await asyncio.sleep(0.1)
        self._stats["last_job_prune"] = result
        if result["job_runs"]:
            logger.info("[EvictionService] terminal job retention: %s", result)
        return result
