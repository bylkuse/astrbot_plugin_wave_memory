"""Production episode setup and WebUI socket lifecycle regressions."""

import asyncio
import socket
import urllib.request

import pytest
from engine.database import WaveMemoryDB
from quart import Quart
from services.experience_episodes import fetch_recent_episodes
from webui.server import Server


def test_database_startup_creates_episode_evidence_store(tmp_path):
    db = WaveMemoryDB(str(tmp_path / "memory.db"), dimension=4)
    try:
        assert fetch_recent_episodes(db.conn, bot_id="test-profile", group_id="g") == []
        columns = {
            row[1] for row in db.conn.execute("PRAGMA table_info(experience_episodes)")
        }
        assert {"source_memory_ids", "idempotency_key", "updated_at"} <= columns
    finally:
        db.close()


@pytest.mark.asyncio
async def test_webui_socket_is_not_inherited_and_restarts_cleanly():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    app = Quart(__name__)

    @app.get("/health")
    async def health():
        return {"ok": True}

    server = Server(host="127.0.0.1", port=port)
    server.app = app

    def request():
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/health", timeout=2
        ) as response:
            return response.status

    for _ in range(2):
        thread = __import__("threading").Thread(target=server._run_thread, daemon=True)
        server._thread = thread
        thread.start()
        try:
            for _ in range(40):
                if server._sockets:
                    break
                await asyncio.sleep(0.05)
            assert server._sockets
            assert all(not sock.get_inheritable() for sock in server._sockets)
            assert await asyncio.to_thread(request) == 200
        finally:
            await server.stop()
        assert not thread.is_alive()
        assert not server._sockets
