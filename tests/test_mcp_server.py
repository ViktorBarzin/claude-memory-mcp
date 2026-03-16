"""Tests for the Claude Memory MCP server."""

import json
import os
import sys

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
