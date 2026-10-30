import asyncio
import logging
import sys
import types
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

if "astrbot.api" not in sys.modules:
    astrbot_module = types.ModuleType("astrbot")
    api_module = types.ModuleType("astrbot.api")
    api_module.logger = logging.getLogger("astrbot-test")
    sys.modules.setdefault("astrbot", astrbot_module)
    sys.modules["astrbot.api"] = api_module


class FakeDB:
    def __init__(self):
        self.added = []
        self.recent_duplicate_id = None
        self.duplicate_checks = []

    def add_memory(self, **kwargs):
        self.added.append(kwargs)
        return len(self.added)

    def find_recent_duplicate_memory(self, *, scope, normalized_content, since_ts):
        self.duplicate_checks.append({
            "scope": scope,
            "normalized_content": normalized_content,
            "since_ts": since_ts,
        })
        return self.recent_duplicate_id


class FakeIndex:
    def __init__(self):
        self.added = []
        self.saved = False
        self.count = 0

    def add(self, ids, vectors):
        self.added.append({"ids": ids, "vectors": vectors})
        self.count += len(ids)

    def save(self):
        self.saved = True


class FakeEmbedding:
    async def get_embeddings(self, texts):
        return [np.ones(3, dtype=np.float32) * (idx + 1) for idx, _ in enumerate(texts)]


class HangingEmbedding:
    async def get_embeddings(self, texts):
        await asyncio.sleep(1)
        return [np.ones(3, dtype=np.float32) for _ in texts]


class FlakyEmbedding:
    def __init__(self):
        self.calls = 0

    async def get_embeddings(self, texts):
        self.calls += 1
        if self.calls == 1:
            await asyncio.sleep(1)
            return [np.ones(3, dtype=np.float32) for _ in texts]
        return [np.ones(3, dtype=np.float32) for _ in texts]


class BlockingQualityRepository:
    def __init__(self):
        self.calls = 0

    def record(self, proposal, decision):
        self.calls += 1
        raise TimeoutError()


class FakeWriteGateway:
    def __init__(self):
        self.calls = []

    async def append_memory(self, **kwargs):
        self.calls.append(dict(kwargs))
        return len(self.calls)


def _group_scope(*, bot_id="bot-a", group_id="group-1"):
    from astrbot_plugin_wave_memory.domain.scope import RuntimeScope, SessionRef

    return RuntimeScope(
        bot_id=bot_id,
        visibility="group",
        session=SessionRef(
            id=f"qq:group:{group_id}",
            platform_id="qq",
            kind="group",
            conversation_id=group_id,
        ),
        subject_principal_id="qq:user:user-1",
    )


def _private_scope(*, bot_id="bot-a", user_id="user-1"):
    from astrbot_plugin_wave_memory.domain.scope import RuntimeScope, SessionRef

    conversation_id = f"user:{user_id}"
    return RuntimeScope(
        bot_id=bot_id,
        visibility="private",
        session=SessionRef(
            id=f"qq:private:{conversation_id}",
            platform_id="qq",
            kind="private",
            conversation_id=conversation_id,
        ),
        subject_principal_id=f"qq:user:{user_id}",
    )


class MessageWriterDedupTest(unittest.TestCase):
    def _writer(self, db=None, *, window_seconds=300):
        from astrbot_plugin_wave_memory.services.message_writer import MemoryDedupPolicy, MessageWriter

        return MessageWriter(
            db=db or FakeDB(),
            memory_index=FakeIndex(),
            embedding_service=FakeEmbedding(),
            bot_keywords=set(),
            noise_max_length=1,
            dedup_policy=MemoryDedupPolicy(window_seconds=window_seconds),
        )

    def test_media_placeholders_skip_queue_and_database_but_keep_caption(self):
        db = FakeDB()
        writer = self._writer(db)
        async def run():
            await writer.enqueue({"scope": _group_scope(), "content": "[图片]"})
            self.assertTrue(writer._queue.empty())
            await writer._process_batch([
                {"scope": _group_scope(), "content": "[At:123] [图片]", "source": "core"},
                {"scope": _group_scope(), "content": "时雨: [图片] 时雨: [图片]"},
                {"scope": _group_scope(), "content": "我的新键盘是红色的 [图片]", "sender_id": "u1"},
            ])
        asyncio.run(run())
        self.assertEqual([row["content"] for row in db.added], ["我的新键盘是红色的"])
        self.assertEqual(writer.stats["placeholder_skipped"], 3)

    def test_generated_error_is_not_stored_but_user_discussion_is(self):
        db = FakeDB()
        writer = self._writer(db)
        asyncio.run(writer._process_batch([
            {"scope": _group_scope(), "sender_id": "bot", "content": "LLM 响应错误: RateLimitError"},
            {"scope": _group_scope(), "sender_id": "u1", "content": "我在研究 RateLimitError 的处理方式"},
        ]))
        self.assertEqual(len(db.added), 1)
        self.assertEqual(db.added[0]["sender_id"], "u1")
        self.assertEqual(writer.stats["system_error_skipped"], 1)

    def test_auto_agent_and_compat_writes_share_content_dedup_in_same_batch(self):
        db = FakeDB()
        writer = self._writer(db)

        asyncio.run(writer._process_batch([
            {
                "scope": _group_scope(),
                "group_id": "group-1",
                "sender_id": "user-1",
                "sender_name": "用户",
                "content": "用户喜欢苹果派",
                "timestamp": 1000.0,
                "source": "core",
                "importance": 1.0,
            },
            {
                "scope": _group_scope(),
                "group_id": "group-1",
                "sender_id": "bot_remember",
                "sender_name": "主动记忆",
                "content": "  用户喜欢苹果派  ",
                "timestamp": 1001.0,
                "source": "core",
                "importance": 1.5,
            },
            {
                "scope": _group_scope(),
                "group_id": "group-1",
                "sender_id": "compat_livingmemory",
                "sender_name": "LivingMemory兼容",
                "content": "用户喜欢苹果派",
                "timestamp": 1002.0,
                "source": "compat_livingmemory",
                "importance": 0.7,
            },
        ]))

        self.assertEqual(len(db.added), 1)
        self.assertEqual(db.added[0]["content"], "用户喜欢苹果派")
        self.assertEqual(db.added[0]["importance"], 1.5)
        self.assertEqual(db.added[0]["scope"].bot_id, "bot-a")
        self.assertEqual(db.added[0]["provenance"]["schema"], "memory-origin/v1")
        self.assertEqual(db.added[0]["origin_metadata"]["origin_kind"], "message_writer:core")
        self.assertEqual(writer.stats.get("duplicate"), 2)

    def test_recent_duplicate_is_skipped_without_modifying_existing_memory(self):
        db = FakeDB()
        db.recent_duplicate_id = 42
        writer = self._writer(db, window_seconds=120)

        asyncio.run(writer._process_batch([
            {
                "scope": _group_scope(),
                "group_id": "group-1",
                "sender_id": "bot_remember",
                "sender_name": "主动记忆",
                "content": "用户喜欢苹果派",
                "timestamp": 2000.0,
                "source": "core",
                "importance": 1.5,
            }
        ]))

        self.assertEqual(db.added, [])
        self.assertEqual(db.duplicate_checks[0]["scope"].bot_id, "bot-a")
        self.assertEqual(db.duplicate_checks[0]["scope"].session.id, "qq:group:group-1")
        self.assertEqual(db.duplicate_checks[0]["normalized_content"], "用户喜欢苹果派")
        self.assertEqual(db.duplicate_checks[0]["since_ts"], 1880.0)
        self.assertEqual(writer.stats.get("duplicate"), 1)

    def test_scope_payload_projects_real_group_and_prevents_cross_bot_batch_dedup(self):
        db = FakeDB()
        writer = self._writer(db)

        asyncio.run(writer._process_batch([
            {
                "scope": _group_scope(bot_id="bot-a"),
                "group_id": "group-1",
                "sender_id": "user-1",
                "sender_name": "用户",
                "content": "两个 Bot 不应互相去重",
                "timestamp": 3000.0,
                "source": "core",
            },
            {
                "scope": _group_scope(bot_id="bot-b"),
                "group_id": "group-1",
                "sender_id": "user-1",
                "sender_name": "用户",
                "content": "两个 Bot 不应互相去重",
                "timestamp": 3001.0,
                "source": "core",
            },
        ]))

        self.assertEqual(len(db.added), 2)
        self.assertEqual({item["group_id"] for item in db.added}, {"group-1"})

    def test_gateway_idempotency_hint_is_bound_to_runtime_scope(self):
        from astrbot_plugin_wave_memory.services.message_writer import MessageWriter

        gateway = FakeWriteGateway()
        writer = MessageWriter(
            db=FakeDB(),
            memory_index=FakeIndex(),
            embedding_service=FakeEmbedding(),
            bot_keywords=set(),
            noise_max_length=1,
            write_gateway=gateway,
        )
        base = {
            "group_id": "group-1",
            "sender_id": "user-1",
            "sender_name": "用户",
            "content": "同一平台事件号",
            "timestamp": 3100.0,
            "source": "core",
            "event_id": "event-42",
        }

        asyncio.run(writer._persist_memory(
            {**base, "scope": _group_scope(bot_id="bot-a")},
            vector=np.ones(3), importance=1.0, source="core",
        ))
        asyncio.run(writer._persist_memory(
            {**base, "scope": _group_scope(bot_id="bot-b")},
            vector=np.ones(3), importance=1.0, source="core",
        ))

        self.assertEqual(len(gateway.calls), 2)
        hints = [call["idempotency_hint"] for call in gateway.calls]
        self.assertNotEqual(hints[0], hints[1])
        self.assertIn("bot-a:group:qq:group:group-1:event-42", hints[0])
        self.assertIn("bot-b:group:qq:group:group-1:event-42", hints[1])

    def test_message_ingress_does_not_wait_for_quality_audit_repository(self):
        from astrbot_plugin_wave_memory.services.message_writer import MessageWriter
        from astrbot_plugin_wave_memory.services.quality_gate import QualityGate

        db = FakeDB()
        repository = BlockingQualityRepository()
        writer = MessageWriter(
            db=db,
            memory_index=FakeIndex(),
            embedding_service=FakeEmbedding(),
            bot_keywords=set(),
            noise_max_length=1,
            quality_gate=QualityGate(repository=repository),
        )

        asyncio.run(writer._process_batch([{
            "scope": _group_scope(),
            "group_id": "group-1",
            "sender_id": "user-1",
            "content": "质量审计不能阻塞基础消息",
            "timestamp": 3200.0,
            "source": "core",
        }]))

        self.assertEqual(repository.calls, 0)
        self.assertEqual(len(db.added), 1)
        self.assertEqual(db.added[0]["provenance"]["quality"]["outcome"], "allow")

    def test_embedding_timeout_persists_scoped_message_without_vector(self):
        from astrbot_plugin_wave_memory.services.message_writer import MessageWriter

        db = FakeDB()
        requested = []
        writer = MessageWriter(
            db=db,
            memory_index=FakeIndex(),
            embedding_service=HangingEmbedding(),
            bot_keywords=set(),
            noise_max_length=1,
            embedding_timeout=0.01,
            embedding_retry_attempts=2,
            embedding_retry_backoff_seconds=0.0,
            on_vector_backfill_requested=lambda: requested.append("queued"),
        )

        asyncio.run(writer._process_batch([{
            "scope": _group_scope(),
            "group_id": "group-1",
            "sender_id": "user-1",
            "content": "向量服务故障时也必须先记住原始消息",
            "timestamp": 3300.0,
            "source": "core",
        }]))

        self.assertEqual(len(db.added), 1)
        self.assertIsNone(db.added[0]["vector"])
        self.assertEqual(writer.stats["embedding_timeouts"], 2)
        self.assertEqual(writer.stats["embedding_retries"], 1)
        self.assertEqual(writer.stats["embedding_terminal_timeouts"], 1)
        self.assertEqual(writer.stats["consecutive_failures"], 0)
        self.assertEqual(requested, ["queued"])

    def test_embedding_timeout_recovers_within_retry_budget(self):
        from astrbot_plugin_wave_memory.services.message_writer import MessageWriter

        db = FakeDB()
        embedding = FlakyEmbedding()
        writer = MessageWriter(
            db=db,
            memory_index=FakeIndex(),
            embedding_service=embedding,
            bot_keywords=set(),
            noise_max_length=1,
            embedding_timeout=0.05,
            embedding_retry_attempts=2,
            embedding_retry_backoff_seconds=0.0,
        )

        asyncio.run(writer._process_batch([{
            "scope": _group_scope(),
            "group_id": "group-1",
            "sender_id": "user-1",
            "content": "短暂超时后应恢复向量",
            "timestamp": 3301.0,
            "source": "core",
        }]))

        self.assertEqual(embedding.calls, 2)
        self.assertEqual(len(db.added), 1)
        self.assertIsNotNone(db.added[0]["vector"])
        self.assertEqual(writer.stats["embedding_timeouts"], 1)
        self.assertEqual(writer.stats["embedding_retries"], 1)
        self.assertEqual(writer.stats["embedding_recovered_after_retry"], 1)
        self.assertEqual(writer.stats["embedding_terminal_timeouts"], 0)

    def test_private_scope_uses_canonical_conversation_and_visibility_dedup_key(self):
        db = FakeDB()
        writer = self._writer(db)
        asyncio.run(writer._process_batch([
            {
                "scope": _private_scope(),
                "group_id": "user:user-1",
                "sender_id": "user-1",
                "content": "私聊消息应进入基础写入链路",
                "timestamp": 3400.0,
                "source": "chat",
            },
            {
                "scope": _private_scope(bot_id="bot-b"),
                "group_id": "user:user-1",
                "sender_id": "user-1",
                "content": "私聊消息应进入基础写入链路",
                "timestamp": 3401.0,
                "source": "chat",
            },
        ]))
        assert len(db.added) == 2
        assert {item["scope"].visibility for item in db.added} == {"private"}
        assert {item["group_id"] for item in db.added} == {"user:user-1"}

    def test_scope_payload_rejects_mismatched_or_non_group_legacy_projection(self):
        from astrbot_plugin_wave_memory.services.message_writer import MessageScopeError
        from astrbot_plugin_wave_memory.domain.scope import RuntimeScope

        writer = self._writer()
        with self.assertRaises(MessageScopeError) as missing:
            asyncio.run(writer._process_batch([{
                "group_id": "group-1",
                "content": "缺少作用域必须拒绝",
            }]))
        self.assertEqual(missing.exception.code, "scope_required")

        with self.assertRaises(MessageScopeError) as mismatch:
            asyncio.run(writer._process_batch([{
                "scope": _group_scope(),
                "group_id": "other-group",
                "content": "作用域不一致",
            }]))
        self.assertEqual(mismatch.exception.code, "scope_session_mismatch")

        private_scope = RuntimeScope(
            bot_id="bot-a",
            visibility="bot_private",
            session=None,
            subject_principal_id=None,
        )
        with self.assertRaises(MessageScopeError) as unsupported:
            asyncio.run(writer._process_batch([{
                "scope": private_scope,
                "content": "不能伪造成群",
            }]))
        self.assertEqual(unsupported.exception.code, "legacy_writer_scope_visibility_unsupported")


if __name__ == "__main__":
    unittest.main()
