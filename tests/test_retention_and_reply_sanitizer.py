"""Regression coverage for annotation delivery and bounded memory retention."""

import json
import sqlite3
from types import SimpleNamespace

import pytest
from reply_components import Image, MessageChain, Node, Plain
from reply_components import reply_components as reply_components

from domain.scope import RuntimeScope, SessionRef
from engine.db.job_repo import JobRepository, JobStateError
from engine.db.outbox_repo import OutboxRepository
from engine.db.retention_repo import (
    RetentionPolicy,
    prune_job_history,
    retention_candidates,
)
from services.memory_mutations import MemoryMutationGateway, MemoryMutationTarget
from services.reply_sanitizer import (
    ImpressionFilter,
    sanitize_chain,
    sanitize_stream,
    strip_impression_marks,
)


@pytest.mark.parametrize(
    "tag",
    [
        "<<impression:喜欢钻研技术 | impact: 3>>",
        "[IMPRESSION：喜欢钻研技术]",
        "<impression>喜欢钻研技术</impression>",
        "【Impression:喜欢钻研技术】",
        "<impression:喜欢钻研技术>",
        "[impression:]",
    ],
)
def test_every_stream_boundary_hides_annotation(tag):
    text = "你好。" + tag + "再见。"
    for split in range(len(text) + 1):
        parser = ImpressionFilter()
        delivered = (
            parser.feed(text[:split]) + parser.feed(text[split:]) + parser.finish()
        )
        assert delivered == "你好。再见。", (tag, split, delivered)
    parser = ImpressionFilter()
    delivered = "".join(parser.feed(ch) for ch in text) + parser.finish()
    assert delivered == "你好。再见。"


def test_incomplete_annotation_and_normal_comparison():
    assert strip_impression_marks("你好。<<impression:这是内部的文字")[0] == "你好。"
    assert (
        strip_impression_marks("a < b，正常文本 [hello]")[0]
        == "a < b，正常文本 [hello]"
    )
    assert strip_impression_marks("你好。\n[Impression:] ")[0] == "你好。"
    assert strip_impression_marks("你好。\n<<impression:内部文字>>")[0] == "你好。"


@pytest.mark.usefixtures("reply_components")
def test_split_components_and_nested_forwarded_messages():
    image = Image.fromURL("https://example.com/image.png")
    node = Node(name="test", uin="1", content=[Plain("转发[Impression:内部文字]")])
    chain = [Plain("正文<<imp"), Plain("ression:内部文字>>"), image, node]
    marks = sanitize_chain(chain)
    assert chain[0].text == "正文"
    assert chain[1].text == ""
    assert chain[2] is image
    assert node.content[0].text == "转发"
    assert marks == ["内部文字", "内部文字"]


@pytest.mark.asyncio
@pytest.mark.usefixtures("reply_components")
async def test_stream_consumes_empty_initial_chain_and_preserves_chunks():
    chunks = [
        MessageChain(chain=[Plain(x)])
        for x in ["正文<", "<im", "pression：内部文字", ">", ">尾声"]
    ]

    async def source():
        for chunk in chunks:
            yield chunk

    delivered = [
        chunk
        async for chunk in sanitize_stream(
            source(), SimpleNamespace(), SimpleNamespace()
        )
    ]
    assert (
        "".join(comp.text for chunk in delivered for comp in chunk.chain) == "正文尾声"
    )


def connection():
    db = sqlite3.connect(":memory:")
    db.execute("PRAGMA foreign_keys=ON")
    db.executescript("""
        CREATE TABLE memories(id INTEGER PRIMARY KEY, group_id TEXT, bot_id TEXT, sender_id TEXT,
          session_id TEXT, visibility TEXT, resolution_state TEXT DEFAULT 'resolved',
          quarantine INTEGER DEFAULT 0, version INTEGER DEFAULT 1, timestamp REAL,
          last_accessed REAL, importance REAL DEFAULT 1, access_count INTEGER DEFAULT 0,
          memory_type TEXT DEFAULT 'message', source TEXT DEFAULT 'chat', content TEXT);
        CREATE TABLE scoped_facts(source_memory_id INTEGER);
        CREATE TABLE scoped_soul_timeline(evidence TEXT);
        CREATE TABLE scoped_memory_tags(memory_id INTEGER);
        CREATE TABLE scoped_memory_effective_tags(memory_id INTEGER);
        CREATE TABLE tag_extraction_status(memory_id INTEGER PRIMARY KEY);
    """)
    OutboxRepository.migrate(db)
    db.commit()
    return db


def test_retention_settings_are_read_by_policy():
    settings = {
        "noise_days": 3.0,
        "low_value_days": 60.0,
        "recent_days": 20.0,
        "max_memories": 40000,
        "max_per_scope": 10000,
        "job_days": 5.0,
    }
    policy = RetentionPolicy.from_settings(settings)
    assert {name: getattr(policy, name) for name in settings} == settings


NOW = 200 * 86400


def add(
    db,
    mid,
    *,
    age=100,
    source="chat",
    importance=1,
    accesses=0,
    last_access=None,
    scope="a",
    visibility="group",
    memory_type="message",
):
    db.execute(
        """INSERT INTO memories(id,group_id,bot_id,session_id,visibility,timestamp,last_accessed,source,importance,access_count,content,memory_type)
        VALUES (?,?, 'bot-alpha',?,?,?,?,?,?,?,'test',?)""",
        (
            mid,
            scope,
            f"qq:{visibility}:{scope}",
            visibility,
            NOW - age * 86400,
            last_access,
            source,
            importance,
            accesses,
            memory_type,
        ),
    )


def test_retention_protects_explicit_core_evidence_recent_and_used():
    db = connection()
    for mid in range(1, 11):
        add(db, mid)
    db.execute("UPDATE memories SET source='explicit' WHERE id=2")
    db.execute("UPDATE memories SET source='core' WHERE id=3")
    db.execute("INSERT INTO scoped_facts VALUES(4)")
    db.execute(
        "INSERT INTO scoped_soul_timeline VALUES(?)", ('[{"kind":"memory","id":"5"}]',)
    )
    db.execute("UPDATE memories SET timestamp=? WHERE id=6", (NOW - 86400,))
    db.execute("UPDATE memories SET last_accessed=? WHERE id=7", (NOW - 86400,))
    db.execute("UPDATE memories SET importance=2 WHERE id=8")
    db.execute("UPDATE memories SET access_count=10 WHERE id=9")
    ids = [row[0] for row in retention_candidates(db, RetentionPolicy(), now=NOW)]
    assert set(ids) == {1, 10}
    db.close()


def test_capacity_favors_valuable_memories_and_isolates_scopes():
    db = connection()
    for mid in range(1, 7):
        add(db, mid, age=40, scope="a" if mid < 5 else "b", importance=1 + mid / 10)
    policy = RetentionPolicy(max_per_scope=3, max_memories=5)
    ids = [row[0] for row in retention_candidates(db, policy, now=NOW)]
    assert ids == [1]
    add(db, 7, age=8, source="noise", memory_type="noise")
    assert 7 in [row[0] for row in retention_candidates(db, policy, now=NOW)]
    db.close()


def test_placeholder_sweep_deletes_recent_empty_media_but_keeps_real_caption():
    db = connection()
    add(db, 1, age=1)
    add(db, 2, age=1)
    db.execute("UPDATE memories SET content='[图片]' WHERE id=1")
    db.execute("UPDATE memories SET content='这是我买的红色键盘 [图片]' WHERE id=2")
    assert [r[0] for r in retention_candidates(db, RetentionPolicy(), now=NOW)] == [1]
    db.execute("UPDATE memories SET content='编辑为有意义的文字',version=2 WHERE id=1")
    assert (
        retention_candidates(db, RetentionPolicy(), now=NOW, scan_placeholders=False)
        == []
    )
    db.close()


def test_automatic_core_counts_against_quota_but_agent_memory_is_kept():
    db = connection()
    for mid in range(1, 5):
        add(db, mid, age=40, source="core")
    db.execute("UPDATE memories SET sender_id='bot_remember' WHERE id=1")
    policy = RetentionPolicy(max_per_scope=2)
    assert [row[0] for row in retention_candidates(db, policy, now=NOW)] == [2]
    db.close()


def test_episode_source_ids_are_protected_from_retention():
    db = connection()
    db.execute("CREATE TABLE experience_episodes(source_memory_ids TEXT)")
    add(db, 1)
    db.execute("INSERT INTO experience_episodes VALUES ('[1]')")
    assert retention_candidates(db, RetentionPolicy(), now=NOW) == []
    db.close()


class Coordinator:
    def __init__(self, db):
        self.db = db
        self._consumer_names = ("memory_index", "fts_cjk")

    async def transaction(self, callback, **kwargs):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            result = callback(self.db)
            self.db.commit()
            return result
        except BaseException:
            self.db.rollback()
            raise


@pytest.mark.asyncio
async def test_retention_deletes_relations_and_commits_scoped_outbox():
    db = connection()
    add(db, 1)
    add(db, 2, scope="private", visibility="private")
    db.execute("INSERT INTO scoped_memory_effective_tags VALUES(1)")
    db.execute("INSERT INTO tag_extraction_status VALUES(1)")
    db.commit()
    gateway = MemoryMutationGateway(
        SimpleNamespace(coordinator=Coordinator(db)),
        clock=SimpleNamespace(now=lambda: NOW),
    )
    result = await gateway.prune_automatic_memories(policy=RetentionPolicy(), limit=10)
    assert len(result.targets) == 2
    assert db.execute("SELECT count(*) FROM memories").fetchone()[0] == 0
    assert (
        db.execute("SELECT count(*) FROM scoped_memory_effective_tags").fetchone()[0]
        == 0
    )
    assert db.execute("SELECT count(*) FROM tag_extraction_status").fetchone()[0] == 0
    assert db.execute("SELECT count(*) FROM outbox_deliveries").fetchone()[0] == 4
    assert {
        json.loads(row[0])["scope"]["visibility"]
        for row in db.execute("SELECT payload_json FROM domain_outbox")
    } == {"group", "private"}
    db.close()


@pytest.mark.asyncio
async def test_existing_manual_delete_callback_executes():
    db = connection()
    add(db, 1)
    db.commit()
    gateway = MemoryMutationGateway(SimpleNamespace(coordinator=Coordinator(db)))
    scope = RuntimeScope(
        bot_id="bot-alpha",
        visibility="group",
        session=SessionRef(
            id="qq:group:a", platform_id="qq", kind="group", conversation_id="a"
        ),
    )
    result = await gateway.delete_memories(
        scope=scope, targets=[MemoryMutationTarget(1, 1)]
    )
    assert len(result.targets) == 1
    db.close()


def test_job_retention_preserves_active_and_latest_generations():
    db = connection()
    JobRepository.migrate(db)
    request = JobRepository.create_request(
        db, idempotency_key="x", kind="test", scope={}, payload={}, created_at=1
    )
    for generation in range(5):
        run = JobRepository.schedule_run(
            db,
            request_id=request.request_id,
            schedule_slot="x",
            cursor_generation=generation,
            cursor={},
            created_at=10,
        )
        db.execute(
            "UPDATE background_job_runs SET status=? WHERE run_id=?",
            ("running" if generation == 1 else "succeeded", run.run_id),
        )
    assert prune_job_history(db, before=100) == 3
    assert db.execute(
        "SELECT cursor_generation FROM background_job_runs ORDER BY cursor_generation"
    ).fetchall() == [(1,), (4,)]
    db.close()


def test_missing_run_release_reports_job_state_error():
    db = connection()
    with pytest.raises(JobStateError, match="unknown job run"):
        JobRepository.release_for_retry(db, "missing", lease_owner="worker", now=NOW)
    db.close()
