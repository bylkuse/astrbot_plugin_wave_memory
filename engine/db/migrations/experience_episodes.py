"""Create episode evidence storage during normal database initialization."""

from __future__ import annotations

from ..connection import ConnectionManager


def ensure_experience_episode_schema(cm: ConnectionManager) -> None:
    """Create the episode table and upgrade earlier nullable cursor columns.

    Args:
        cm: Database connection manager owning the startup migration.
    """
    with cm.migration_transaction() as connection:
        connection.execute("""CREATE TABLE IF NOT EXISTS experience_episodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bot_id TEXT NOT NULL, group_id TEXT NOT NULL, user_id TEXT,
            episode_type TEXT NOT NULL, trigger_text TEXT, bot_inner_thought TEXT,
            bot_action TEXT, bot_reply TEXT, user_reaction TEXT, outcome TEXT,
            source_memory_ids TEXT DEFAULT '[]', emotional_weight REAL DEFAULT 0,
            created_at REAL NOT NULL, idempotency_key TEXT, updated_at REAL)""")
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(experience_episodes)")
        }
        for name, kind in (("idempotency_key", "TEXT"), ("updated_at", "REAL")):
            if name not in columns:
                connection.execute(
                    f"ALTER TABLE experience_episodes ADD COLUMN {name} {kind}"
                )
        connection.execute("""CREATE INDEX IF NOT EXISTS idx_experience_bot_time
            ON experience_episodes(bot_id, created_at)""")
        connection.execute("""CREATE INDEX IF NOT EXISTS idx_experience_scene_time
            ON experience_episodes(bot_id, group_id, created_at DESC)""")
        connection.execute("""CREATE UNIQUE INDEX IF NOT EXISTS idx_experience_episode_idempotency
            ON experience_episodes(bot_id, group_id, idempotency_key)
            WHERE idempotency_key IS NOT NULL""")
