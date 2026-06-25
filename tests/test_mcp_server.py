"""Tests for the Claude Memory MCP server."""

import json
import os
import sqlite3
import sys
import threading
from datetime import datetime, timezone
from typing import Any
from unittest.mock import patch

import pytest

# Force SQLite fallback mode for all tests
os.environ.pop("MEMORY_API_KEY", None)
os.environ.pop("CLAUDE_MEMORY_API_KEY", None)

# Add src to path so we can import without installing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from claude_memory.mcp_server import MemoryServer, SERVER_NAME, SERVER_VERSION, PROTOCOL_VERSION


@pytest.fixture
def server(tmp_path):
    """Create a MemoryServer with a temporary SQLite database."""
    db_path = str(tmp_path / "test_memory.db")
    srv = MemoryServer(sqlite_db_path=db_path)
    yield srv
    if srv.sqlite_conn:
        srv.sqlite_conn.close()


class TestSQLiteInit:
    def test_creates_database(self, tmp_path):
        db_path = str(tmp_path / "sub" / "test.db")
        srv = MemoryServer(sqlite_db_path=db_path)
        assert os.path.exists(db_path)
        # Verify tables exist
        cursor = srv.sqlite_conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memories'")
        assert cursor.fetchone() is not None
        srv.sqlite_conn.close()

    def test_creates_fts_table(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        srv = MemoryServer(sqlite_db_path=db_path)
        cursor = srv.sqlite_conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memories_fts'")
        assert cursor.fetchone() is not None
        srv.sqlite_conn.close()


class TestMemoryStore:
    def test_store_basic(self, server):
        result = server.memory_store({
            "content": "User prefers dark mode",
            "expanded_keywords": "dark mode theme preference ui",
        })
        assert "Stored memory #1" in result
        assert "facts" in result

    def test_store_with_category(self, server):
        result = server.memory_store({
            "content": "User likes Python",
            "category": "preferences",
            "expanded_keywords": "python programming language preference",
        })
        assert "preferences" in result

    def test_store_with_importance(self, server):
        result = server.memory_store({
            "content": "Critical info",
            "importance": 0.9,
            "expanded_keywords": "critical important info",
        })
        assert "0.9" in result

    def test_store_requires_content(self, server):
        with pytest.raises(ValueError, match="content is required"):
            server.memory_store({"expanded_keywords": "test"})

    def test_store_force_sensitive(self, server):
        result = server.memory_store({
            "content": "API key: sk-1234",
            "force_sensitive": True,
            "expanded_keywords": "api key secret credential",
        })
        assert "Stored memory #1" in result
        # Verify is_sensitive flag is set
        cursor = server.sqlite_conn.cursor()
        cursor.execute("SELECT is_sensitive FROM memories WHERE id = 1")
        row = cursor.fetchone()
        assert row["is_sensitive"] == 1


class TestMemoryRecall:
    def test_recall_finds_memory(self, server):
        server.memory_store({
            "content": "User works at Acme Corp",
            "expanded_keywords": "acme corp company work employer",
        })
        result = server.memory_recall({
            "context": "work",
            "expanded_query": "company employer job",
        })
        assert "Acme Corp" in result
        assert "Found 1 memories" in result

    def test_recall_no_results(self, server):
        result = server.memory_recall({
            "context": "nonexistent topic",
            "expanded_query": "nothing here at all",
        })
        assert "No memories found" in result

    def test_recall_with_category_filter(self, server):
        server.memory_store({
            "content": "User prefers vim",
            "category": "preferences",
            "expanded_keywords": "vim editor preference text",
        })
        server.memory_store({
            "content": "Project uses React",
            "category": "projects",
            "expanded_keywords": "react project frontend framework",
        })
        result = server.memory_recall({
            "context": "preferences",
            "expanded_query": "vim editor",
            "category": "preferences",
        })
        assert "vim" in result
        assert "React" not in result

    def test_recall_requires_context(self, server):
        with pytest.raises(ValueError, match="context is required"):
            server.memory_recall({"expanded_query": "test"})


class TestMemoryList:
    def test_list_empty(self, server):
        result = server.memory_list({})
        assert "No memories stored yet" in result

    def test_list_with_memories(self, server):
        server.memory_store({
            "content": "Memory one",
            "expanded_keywords": "one first test",
        })
        server.memory_store({
            "content": "Memory two",
            "expanded_keywords": "two second test",
        })
        result = server.memory_list({})
        assert "Memory one" in result
        assert "Memory two" in result
        assert "2 shown" in result

    def test_list_with_category(self, server):
        server.memory_store({
            "content": "A fact",
            "category": "facts",
            "expanded_keywords": "fact test",
        })
        server.memory_store({
            "content": "A preference",
            "category": "preferences",
            "expanded_keywords": "preference test",
        })
        result = server.memory_list({"category": "facts"})
        assert "A fact" in result
        assert "A preference" not in result

    def test_list_empty_category(self, server):
        result = server.memory_list({"category": "projects"})
        assert "No memories in category 'projects'" in result

    def test_list_respects_limit(self, server):
        for i in range(5):
            server.memory_store({
                "content": f"Memory {i}",
                "expanded_keywords": f"memory number {i}",
            })
        result = server.memory_list({"limit": 2})
        assert "2 shown" in result


class TestMemoryDelete:
    def test_delete_existing(self, server):
        server.memory_store({
            "content": "To be deleted",
            "expanded_keywords": "delete remove test",
        })
        result = server.memory_delete({"id": 1})
        assert "Deleted memory #1" in result
        assert "To be deleted" in result

    def test_delete_nonexistent(self, server):
        result = server.memory_delete({"id": 999})
        assert "not found" in result

    def test_delete_requires_id(self, server):
        with pytest.raises(ValueError, match="id is required"):
            server.memory_delete({})


class TestSecretGet:
    def test_secret_get_sensitive(self, server):
        server.memory_store({
            "content": "secret password 12345",
            "force_sensitive": True,
            "expanded_keywords": "password secret credential",
        })
        result = server.secret_get({"id": 1})
        assert "secret password 12345" in result

    def test_secret_get_not_sensitive(self, server):
        server.memory_store({
            "content": "public info",
            "expanded_keywords": "public info test",
        })
        result = server.secret_get({"id": 1})
        assert "not marked as sensitive" in result

    def test_secret_get_nonexistent(self, server):
        result = server.secret_get({"id": 999})
        assert "not found" in result

    def test_secret_get_requires_id(self, server):
        with pytest.raises(ValueError, match="id is required"):
            server.secret_get({})


class TestMCPProtocol:
    def test_handle_initialize(self, server):
        result = server.handle_initialize({})
        assert result["protocolVersion"] == PROTOCOL_VERSION
        assert result["serverInfo"]["name"] == SERVER_NAME
        assert result["serverInfo"]["version"] == SERVER_VERSION
        assert "tools" in result["capabilities"]

    def test_handle_tools_list(self, server):
        result = server.handle_tools_list({})
        tools = result["tools"]
        assert len(tools) == 6
        names = {t["name"] for t in tools}
        assert names == {"memory_store", "memory_recall", "memory_list", "memory_delete", "secret_get", "memory_count"}

    def test_handle_tools_call_store(self, server):
        result = server.handle_tools_call({
            "name": "memory_store",
            "arguments": {
                "content": "test memory",
                "expanded_keywords": "test memory keywords",
            },
        })
        assert not result.get("isError", False)
        assert "Stored memory" in result["content"][0]["text"]

    def test_handle_tools_call_unknown(self, server):
        result = server.handle_tools_call({
            "name": "nonexistent_tool",
            "arguments": {},
        })
        assert result["isError"] is True
        assert "Unknown tool" in result["content"][0]["text"]

    def test_handle_tools_call_error(self, server):
        result = server.handle_tools_call({
            "name": "memory_store",
            "arguments": {},  # missing content
        })
        assert result["isError"] is True
        assert "Error" in result["content"][0]["text"]


class TestProcessMessage:
    def test_initialize(self, server):
        response = server.process_message({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {},
        })
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 1
        assert "result" in response
        assert response["result"]["serverInfo"]["name"] == SERVER_NAME

    def test_tools_list(self, server):
        response = server.process_message({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {},
        })
        assert "result" in response
        assert len(response["result"]["tools"]) == 6

    def test_tools_call(self, server):
        response = server.process_message({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "memory_store",
                "arguments": {
                    "content": "via process_message",
                    "expanded_keywords": "process message test",
                },
            },
        })
        assert "result" in response
        assert "Stored memory" in response["result"]["content"][0]["text"]

    def test_unknown_method(self, server):
        response = server.process_message({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "unknown/method",
            "params": {},
        })
        assert "error" in response
        assert response["error"]["code"] == -32601
        assert "Method not found" in response["error"]["message"]

    def test_notification_no_id(self, server):
        response = server.process_message({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {},
        })
        assert response is None

    def test_jsonrpc_response_format(self, server):
        response = server.process_message({
            "jsonrpc": "2.0",
            "id": 5,
            "method": "initialize",
            "params": {},
        })
        # Verify it's valid JSON when serialized
        serialized = json.dumps(response)
        parsed = json.loads(serialized)
        assert parsed["jsonrpc"] == "2.0"
        assert parsed["id"] == 5


class TestMemoryCount:
    def test_count_empty(self, server):
        result = server.memory_count({})
        assert "0" in result

    def test_count_after_store(self, server):
        server.memory_store({
            "content": "test memory",
            "expanded_keywords": "test memory keywords data",
        })
        result = server.memory_count({})
        assert "1" in result
        assert "facts" in result

    def test_count_multiple_categories(self, server):
        server.memory_store({
            "content": "a fact",
            "category": "facts",
            "expanded_keywords": "fact test data words",
        })
        server.memory_store({
            "content": "a preference",
            "category": "preferences",
            "expanded_keywords": "preference test data words",
        })
        result = server.memory_count({})
        assert "facts: 1" in result
        assert "preferences: 1" in result

    def test_count_via_tools_call(self, server):
        result = server.handle_tools_call({
            "name": "memory_count",
            "arguments": {},
        })
        assert not result.get("isError", False)
        assert "0" in result["content"][0]["text"]


class TestSchemaMigration:
    def test_schema_version_set(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        srv = MemoryServer(sqlite_db_path=db_path)
        cursor = srv.sqlite_conn.cursor()
        version = cursor.execute("PRAGMA user_version").fetchone()[0]
        assert version == 2
        srv.sqlite_conn.close()

    def test_migration_idempotent(self, tmp_path):
        """Running _init_sqlite twice should not error."""
        from claude_memory.mcp_server import _init_sqlite
        db_path = str(tmp_path / "test.db")
        conn1, _ = _init_sqlite(db_path)
        conn1.close()
        conn2, _ = _init_sqlite(db_path)
        version = conn2.execute("PRAGMA user_version").fetchone()[0]
        assert version == 2
        conn2.close()

    def test_server_id_column_exists(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        srv = MemoryServer(sqlite_db_path=db_path)
        cursor = srv.sqlite_conn.cursor()
        cursor.execute("PRAGMA table_info(memories)")
        columns = {row["name"] for row in cursor.fetchall()}
        assert "server_id" in columns
        srv.sqlite_conn.close()


def _server_rows(server: MemoryServer) -> int:
    assert server.sqlite_conn is not None
    return int(server.sqlite_conn.execute("SELECT COUNT(*) AS c FROM memories").fetchone()["c"])


class TestConcurrentWrites:
    """Regression tests for 'database is locked' under heavy concurrent local writes.

    The store path (MemoryServer) and the background sync writer (SyncEngine) must not
    contend on the single SQLite writer. Before the fix they used two separate connections
    to the same file; under load the second writer blew past busy_timeout and raised
    ``sqlite3.OperationalError: database is locked``. After the fix all local writes are
    serialized through one shared, lock-guarded connection, so a lock error is impossible.
    """

    def test_concurrent_stores_all_rows_land(self, tmp_path):
        """Many threads calling memory_store concurrently → every row lands, no lock error."""
        db_path = str(tmp_path / "concurrent.db")
        server = MemoryServer(sqlite_db_path=db_path)
        try:
            n_threads = 16
            per_thread = 12
            errors: list[BaseException] = []
            barrier = threading.Barrier(n_threads)

            def worker(tid: int) -> None:
                barrier.wait()  # release all threads at once to maximise contention
                for i in range(per_thread):
                    try:
                        server.memory_store({
                            "content": f"thread {tid} memory {i}",
                            "expanded_keywords": f"thread {tid} memory {i} concurrent write",
                        })
                    except BaseException as exc:  # noqa: BLE001 — capture everything for the assert
                        errors.append(exc)

            threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert errors == [], f"concurrent stores raised: {errors!r}"
            assert _server_rows(server) == n_threads * per_thread
        finally:
            if server.sqlite_conn:
                server.sqlite_conn.close()

    def test_concurrent_stores_while_sync_writer_active_no_lock(self, tmp_path):
        """Store path + background sync writer hammer the SAME file → no 'database is locked'.

        Reproduces the production failure: ``MemoryServer`` and ``SyncEngine`` both write to
        one SQLite file. We shrink ``busy_timeout`` so the structural race (two writers fighting
        the single SQLite write lock) surfaces within seconds instead of 30s. The shared-writer
        fix makes a lock error impossible regardless of busy_timeout.
        """
        from claude_memory.sync import SyncEngine

        db_path = str(tmp_path / "hybrid.db")
        server = MemoryServer(sqlite_db_path=db_path)
        # The production hybrid wiring: the sync engine SHARES the server's LocalStore
        # (one connection, one lock) — the structural fix for the cross-connection race.
        engine = SyncEngine(
            db_path=db_path,
            api_base_url="http://fake-api:8080",
            api_key="test-key",
            sync_interval=3600,  # never auto-syncs; we drive the writer by hand
            store=server.store,
        )
        assert engine._conn is server.sqlite_conn  # truly one shared connection

        # Shrink the busy timeout so that, were there still two writers, the race would
        # surface in ms not 30s. With one shared connection a lock error is impossible.
        server.sqlite_conn.execute("PRAGMA busy_timeout=50")

        now = datetime.now(timezone.utc).isoformat()

        # A write-heavy pull: many rows upserted inside the sync engine's single lock block —
        # exactly the kind of long-held writer that starved the store path.
        def big_pull(method: str, path: str, body: Any = None) -> dict[str, Any]:
            return {
                "memories": [
                    {
                        "id": 10_000 + j,
                        "content": f"server row {j}",
                        "category": "facts",
                        "tags": "",
                        "expanded_keywords": "",
                        "importance": 0.5,
                        "is_sensitive": False,
                        "created_at": now,
                        "updated_at": now,
                        "deleted_at": None,
                    }
                    for j in range(120)
                ],
                "server_time": now,
            }

        errors: list[BaseException] = []
        stop = threading.Event()

        def sync_writer() -> None:
            with patch.object(engine, "_api_request", side_effect=big_pull):
                while not stop.is_set():
                    try:
                        engine._pull_changes()
                    except BaseException as exc:  # noqa: BLE001
                        errors.append(exc)

        n_threads = 12
        per_thread = 12
        barrier = threading.Barrier(n_threads)

        def store_worker(tid: int) -> None:
            barrier.wait()
            for i in range(per_thread):
                try:
                    server.memory_store({
                        "content": f"local {tid}.{i}",
                        "expanded_keywords": f"local thread {tid} item {i} keywords here",
                    })
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

        writer = threading.Thread(target=sync_writer, daemon=True)
        writer.start()
        final_rows = 0
        try:
            store_threads = [threading.Thread(target=store_worker, args=(t,)) for t in range(n_threads)]
            for t in store_threads:
                t.start()
            for t in store_threads:
                t.join()
            stop.set()
            writer.join(timeout=5)
            final_rows = _server_rows(server)  # read while the connection is still open
        finally:
            stop.set()
            engine.stop()  # shares the server's store → does not close the connection
            if server.sqlite_conn:
                server.sqlite_conn.close()

        locked = [e for e in errors if isinstance(e, sqlite3.OperationalError) and "locked" in str(e)]
        assert locked == [], f"hit 'database is locked' under concurrency: {locked!r}"
        assert errors == [], f"concurrent writes raised: {errors!r}"
        # Every local store must have landed.
        assert final_rows >= n_threads * per_thread


class TestRecallProgressAndBounding:
    """The slow path — a remote semantic ``memory_recall`` — must be bounded and give a
    notion of progress instead of hanging silently and dropping the session."""

    def test_remote_recall_timeout_returns_clear_signal_not_raise(self, server):
        """A timed-out / unreachable remote recall returns a clear 'retry' message, never hangs/raises."""
        import claude_memory.mcp_server as m

        with patch.object(m, "_api_request", side_effect=RuntimeError("API connection error: timed out")):
            text = server._recall_remote({"context": "x", "expanded_query": "a b c d e"})

        assert "recall" in text.lower()
        # Mentions the bound and that the caller should retry — a clear working/not-done signal.
        assert "retry" in text.lower() or "again" in text.lower()
        assert str(m.RECALL_TIMEOUT) in text

    def test_remote_recall_success_formats_rows(self, server):
        """A successful remote recall still formats results normally."""
        import claude_memory.mcp_server as m

        payload = {"memories": [
            {"id": 7, "category": "facts", "importance": 0.8, "content": "hello",
             "tags": "t", "created_at": "2026-01-01T00:00:00+00:00"},
        ]}
        with patch.object(m, "_api_request", return_value=payload):
            text = server._recall_remote({"context": "x", "expanded_query": "a b c d e"})

        assert "Found 1 memories" in text
        assert "hello" in text

    def test_remote_recall_uses_bounded_timeout(self, server):
        """The remote recall passes the bounded RECALL_TIMEOUT to the HTTP layer."""
        import claude_memory.mcp_server as m

        with patch.object(m, "_api_request", return_value={"memories": []}) as mock_api:
            server._recall_remote({"context": "x", "expanded_query": "a b c d e"})

        _, kwargs = mock_api.call_args
        assert kwargs.get("timeout") == m.RECALL_TIMEOUT

    def test_api_request_accepts_timeout_kwarg(self):
        """Module-level _api_request must accept an explicit timeout without breaking callers."""
        import inspect
        import claude_memory.mcp_server as m

        sig = inspect.signature(m._api_request)
        assert "timeout" in sig.parameters

    def test_progress_notification_emitted_for_recall_with_token(self, server):
        """When the client supplies a progressToken, a 'working' progress notification is emitted."""
        sent: list[dict[str, Any]] = []
        server._notify = lambda method, params: sent.append({"method": method, "params": params})

        with patch.object(server, "memory_recall", return_value="ok"):
            server.handle_tools_call({
                "name": "memory_recall",
                "arguments": {"context": "x", "expanded_query": "a b c d e"},
                "_meta": {"progressToken": "tok-1"},
            })

        progress = [s for s in sent if s["method"] == "notifications/progress"]
        assert progress, "expected a notifications/progress for a tokened recall"
        assert progress[0]["params"]["progressToken"] == "tok-1"

    def test_no_progress_notification_without_token(self, server):
        """No token → no progress notification (don't spam clients that didn't opt in)."""
        sent: list[dict[str, Any]] = []
        server._notify = lambda method, params: sent.append({"method": method, "params": params})

        with patch.object(server, "memory_recall", return_value="ok"):
            server.handle_tools_call({
                "name": "memory_recall",
                "arguments": {"context": "x", "expanded_query": "a b c d e"},
            })

        assert [s for s in sent if s["method"] == "notifications/progress"] == []

    def test_fast_tools_do_not_emit_progress(self, server):
        """Only the slow recall path signals progress; a store with a token stays quiet."""
        sent: list[dict[str, Any]] = []
        server._notify = lambda method, params: sent.append({"method": method, "params": params})

        server.handle_tools_call({
            "name": "memory_store",
            "arguments": {"content": "x", "expanded_keywords": "a b c d e"},
            "_meta": {"progressToken": "tok-2"},
        })

        assert [s for s in sent if s["method"] == "notifications/progress"] == []


# ─── FastMCP Postgres-backed tools (api/app.py) wire through the shared helpers ──
#
# app.memory_recall / app.memory_store are the FastMCP (Postgres) tools — distinct
# from the local SQLite MemoryServer above. They must use the SAME shared
# _fused_recall / schedule_embedding helpers as the REST endpoints so the two recall
# paths and two store paths cannot drift (S8 item (f)).


@pytest.fixture
def fastmcp_app():
    """Reload api.app with a mocked asyncpg pool + a fixed MCP user context.

    Returns (app_mod, conn). app_mod.memory_recall / app_mod.memory_store are the
    FastMCP tool coroutines; in this FastMCP version the decorator returns the original
    function, so they can be awaited directly.
    """
    import importlib
    from unittest.mock import AsyncMock, MagicMock

    conn = AsyncMock()
    pool = MagicMock()
    acm = MagicMock()
    acm.__aenter__ = AsyncMock(return_value=conn)
    acm.__aexit__ = AsyncMock(return_value=False)
    pool.acquire.return_value = acm

    with patch.dict(os.environ, {"DATABASE_URL": "postgresql://test", "API_KEY": "k", "API_KEYS": ""}):
        import claude_memory.api.database as db_mod
        import claude_memory.api.app as app_mod

        importlib.reload(db_mod)
        importlib.reload(app_mod)
        db_mod.pool = pool
        app_mod._current_user.set("mcpuser")
        yield app_mod, conn


class TestFastMCPToolsShareHelpers:
    @pytest.mark.asyncio
    async def test_memory_recall_delegates_to_shared_fused_recall(self, fastmcp_app):
        """(f) FastMCP memory_recall retrieves via the shared _fused_recall helper."""
        from datetime import datetime, timezone

        app_mod, conn = fastmcp_app
        now = datetime.now(timezone.utc)
        captured: dict[str, Any] = {}

        async def fake_fused(conn_arg, **kwargs):
            captured.update(kwargs)
            return [{
                "id": 1, "content": "mcp helper row", "category": "facts", "tags": "",
                "importance": 0.5, "is_sensitive": False, "rank": 0.5,
                "owner": "mcpuser", "created_at": now, "updated_at": now, "shared_by": None,
            }]

        with patch.object(app_mod, "_fused_recall", side_effect=fake_fused) as mock_fused:
            out = await app_mod.memory_recall(context="hello mcp", limit=9)

        data = json.loads(out)
        assert data["memories"][0]["content"] == "mcp helper row"
        mock_fused.assert_called_once()
        assert captured["query_text"] == "hello mcp"
        assert captured["limit"] == 9
        assert captured["user_id"] == "mcpuser"

    @pytest.mark.asyncio
    async def test_memory_store_schedules_embed_on_write(self, fastmcp_app):
        """(e/f) FastMCP memory_store calls the shared schedule_embedding after INSERT."""
        app_mod, conn = fastmcp_app
        conn.fetchrow.return_value = {"id": 77}

        with patch.object(app_mod, "schedule_embedding") as mock_sched:
            out = await app_mod.memory_store(content="store via mcp", category="facts")

        assert json.loads(out)["id"] == 77
        mock_sched.assert_called_once()
        args, kwargs = mock_sched.call_args
        assert args[1] == 77
        assert args[2] == "store via mcp"
        assert kwargs["is_sensitive"] is False

    @pytest.mark.asyncio
    async def test_memory_store_sensitive_chunk_not_embedded(self, fastmcp_app):
        """(c) A sensitive chunk passes is_sensitive=True to schedule_embedding."""
        app_mod, conn = fastmcp_app
        conn.fetchrow.return_value = {"id": 88}

        with patch.object(app_mod, "schedule_embedding") as mock_sched, \
             patch.object(app_mod, "_detect_sensitive", return_value=True), \
             patch.object(app_mod, "is_vault_configured", return_value=False):
            await app_mod.memory_store(content="api key sk-xyz", category="facts")

        mock_sched.assert_called_once()
        assert mock_sched.call_args.kwargs["is_sensitive"] is True
