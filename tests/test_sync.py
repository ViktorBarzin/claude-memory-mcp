"""Tests for the SyncEngine (local SQLite cache + remote API sync)."""

import json
import os
import sys
import urllib.error
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

import pytest

# Force SQLite-only mode for test imports
os.environ.pop("MEMORY_API_KEY", None)
os.environ.pop("CLAUDE_MEMORY_API_KEY", None)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from claude_memory.mcp_server import _init_sqlite
from claude_memory.sync import SyncEngine


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test_sync.db")


@pytest.fixture
def sqlite_conn(db_path):
    """Create a SQLite database with the standard schema."""
    conn, _ = _init_sqlite(db_path)
    yield conn
    conn.close()


@pytest.fixture
def engine(db_path, sqlite_conn):
    """Create a SyncEngine with mocked API."""
    eng = SyncEngine(
        db_path=db_path,
        api_base_url="http://fake-api:8080",
        api_key="test-key",
        sync_interval=3600,  # Don't auto-sync in tests
    )
    yield eng
    eng._conn.close()


class TestSyncEngineInit:
    def test_creates_pending_ops_table(self, engine):
        cursor = engine._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='pending_ops'"
        )
        assert cursor.fetchone() is not None

    def test_creates_sync_meta_table(self, engine):
        cursor = engine._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='sync_meta'"
        )
        assert cursor.fetchone() is not None

    def test_adds_server_id_column(self, engine):
        cursor = engine._conn.execute("PRAGMA table_info(memories)")
        columns = {row["name"] for row in cursor.fetchall()}
        assert "server_id" in columns

    def test_server_id_unique_index(self, engine):
        cursor = engine._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_memories_server_id'"
        )
        assert cursor.fetchone() is not None


class TestEnqueueOps:
    def test_enqueue_store(self, engine):
        engine.enqueue_store(
            local_id=1,
            content="test memory",
            category="facts",
            tags="test",
            expanded_keywords="test memory keywords",
            importance=0.7,
        )
        cursor = engine._conn.execute("SELECT * FROM pending_ops")
        ops = cursor.fetchall()
        assert len(ops) == 1
        assert ops[0]["op_type"] == "store"
        payload = json.loads(ops[0]["payload"])
        assert payload["content"] == "test memory"
        assert payload["local_id"] == 1
        assert payload["importance"] == 0.7

    def test_enqueue_delete(self, engine):
        engine.enqueue_delete(server_id=42)
        cursor = engine._conn.execute("SELECT * FROM pending_ops")
        ops = cursor.fetchall()
        assert len(ops) == 1
        assert ops[0]["op_type"] == "delete"
        payload = json.loads(ops[0]["payload"])
        assert payload["server_id"] == 42

    def test_multiple_enqueues(self, engine):
        engine.enqueue_store(1, "mem1", "facts", "", "", 0.5)
        engine.enqueue_store(2, "mem2", "facts", "", "", 0.5)
        engine.enqueue_delete(10)
        cursor = engine._conn.execute("SELECT COUNT(*) as cnt FROM pending_ops")
        assert cursor.fetchone()["cnt"] == 3


class TestPushPendingOps:
    def test_push_store_clears_queue(self, engine):
        engine.enqueue_store(1, "test", "facts", "", "kw", 0.5)

        with patch.object(engine, "_api_request") as mock_api:
            mock_api.return_value = {"id": 100, "category": "facts", "importance": 0.5}
            engine._push_pending_ops()

        # Queue should be empty
        cursor = engine._conn.execute("SELECT COUNT(*) as cnt FROM pending_ops")
        assert cursor.fetchone()["cnt"] == 0

        # server_id should be set on local memory (if it exists)
        mock_api.assert_called_once()

    def test_push_store_updates_server_id(self, engine, sqlite_conn):
        # Insert a local memory first
        now = datetime.now(timezone.utc).isoformat()
        sqlite_conn.execute(
            "INSERT INTO memories (id, content, category, tags, expanded_keywords, importance, is_sensitive, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (1, "test content", "facts", "", "kw", 0.5, 0, now, now),
        )
        sqlite_conn.commit()

        engine.enqueue_store(1, "test content", "facts", "", "kw", 0.5)

        with patch.object(engine, "_api_request") as mock_api:
            mock_api.return_value = {"id": 200, "category": "facts", "importance": 0.5}
            engine._push_pending_ops()

        # Check server_id was updated
        cursor = engine._conn.execute("SELECT server_id FROM memories WHERE id = 1")
        row = cursor.fetchone()
        assert row["server_id"] == 200

    def test_push_delete_clears_queue(self, engine):
        engine.enqueue_delete(42)

        with patch.object(engine, "_api_request") as mock_api:
            mock_api.return_value = {"deleted": 42, "preview": "test"}
            engine._push_pending_ops()

        cursor = engine._conn.execute("SELECT COUNT(*) as cnt FROM pending_ops")
        assert cursor.fetchone()["cnt"] == 0

    def test_push_delete_404_still_clears(self, engine):
        """A 404 on delete means already deleted on server — should still clear queue."""
        engine.enqueue_delete(42)

        import urllib.error
        with patch.object(engine, "_api_request") as mock_api:
            mock_api.side_effect = urllib.error.HTTPError(
                url="http://fake", code=404, msg="Not Found", hdrs=None, fp=None
            )
            engine._push_pending_ops()

        cursor = engine._conn.execute("SELECT COUNT(*) as cnt FROM pending_ops")
        assert cursor.fetchone()["cnt"] == 0

    def test_push_failure_keeps_queue_returns_false(self, engine):
        """Push failure should keep the op in queue and return False (not raise)."""
        engine.enqueue_store(1, "test", "facts", "", "kw", 0.5)

        with patch.object(engine, "_api_request") as mock_api:
            mock_api.side_effect = RuntimeError("Connection refused")
            result = engine._push_pending_ops()

        assert result is False
        cursor = engine._conn.execute("SELECT COUNT(*) as cnt FROM pending_ops")
        assert cursor.fetchone()["cnt"] == 1


class TestPullChanges:
    def test_pull_inserts_new_memories(self, engine):
        now = datetime.now(timezone.utc).isoformat()
        with patch.object(engine, "_api_request") as mock_api:
            mock_api.return_value = {
                "memories": [
                    {
                        "id": 10,
                        "content": "server memory",
                        "category": "facts",
                        "tags": "tag1",
                        "expanded_keywords": "server memory keywords",
                        "importance": 0.8,
                        "is_sensitive": False,
                        "created_at": now,
                        "updated_at": now,
                        "deleted_at": None,
                    }
                ],
                "server_time": now,
            }
            engine._pull_changes()

        cursor = engine._conn.execute("SELECT * FROM memories WHERE server_id = 10")
        row = cursor.fetchone()
        assert row is not None
        assert row["content"] == "server memory"
        assert row["importance"] == 0.8

    def test_pull_updates_existing_memories(self, engine):
        now = datetime.now(timezone.utc).isoformat()
        # Insert existing memory with server_id
        engine._conn.execute(
            "INSERT INTO memories (content, category, tags, expanded_keywords, importance, is_sensitive, created_at, updated_at, server_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("old content", "facts", "", "", 0.5, 0, now, now, 10),
        )
        engine._conn.commit()

        with patch.object(engine, "_api_request") as mock_api:
            mock_api.return_value = {
                "memories": [
                    {
                        "id": 10,
                        "content": "updated content",
                        "category": "projects",
                        "tags": "",
                        "expanded_keywords": "",
                        "importance": 0.9,
                        "is_sensitive": False,
                        "created_at": now,
                        "updated_at": now,
                        "deleted_at": None,
                    }
                ],
                "server_time": now,
            }
            engine._pull_changes()

        cursor = engine._conn.execute("SELECT * FROM memories WHERE server_id = 10")
        row = cursor.fetchone()
        assert row["content"] == "updated content"
        assert row["category"] == "projects"
        assert row["importance"] == 0.9

    def test_pull_deletes_soft_deleted(self, engine):
        now = datetime.now(timezone.utc).isoformat()
        engine._conn.execute(
            "INSERT INTO memories (content, category, tags, expanded_keywords, importance, is_sensitive, created_at, updated_at, server_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("to be deleted", "facts", "", "", 0.5, 0, now, now, 20),
        )
        engine._conn.commit()

        with patch.object(engine, "_api_request") as mock_api:
            mock_api.return_value = {
                "memories": [
                    {
                        "id": 20,
                        "content": "to be deleted",
                        "category": "facts",
                        "tags": "",
                        "expanded_keywords": "",
                        "importance": 0.5,
                        "is_sensitive": False,
                        "created_at": now,
                        "updated_at": now,
                        "deleted_at": now,
                    }
                ],
                "server_time": now,
            }
            engine._pull_changes()

        cursor = engine._conn.execute("SELECT * FROM memories WHERE server_id = 20")
        assert cursor.fetchone() is None

    def test_pull_updates_last_sync_ts(self, engine):
        server_time = "2026-03-14T12:00:00+00:00"
        with patch.object(engine, "_api_request") as mock_api:
            mock_api.return_value = {
                "memories": [],
                "server_time": server_time,
            }
            engine._pull_changes()

        assert engine.last_sync_ts == server_time

    def test_pull_with_since_param(self, engine):
        engine.last_sync_ts = "2026-03-14T10:00:00+00:00"

        with patch.object(engine, "_api_request") as mock_api:
            mock_api.return_value = {"memories": [], "server_time": "2026-03-14T12:00:00+00:00"}
            engine._pull_changes()

        call_args = mock_api.call_args
        from urllib.parse import unquote
        assert "since=2026-03-14T10:00:00+00:00" in unquote(call_args[0][1])


class TestTrySyncStore:
    def test_success_returns_server_id(self, engine, sqlite_conn):
        now = datetime.now(timezone.utc).isoformat()
        sqlite_conn.execute(
            "INSERT INTO memories (id, content, category, tags, expanded_keywords, importance, is_sensitive, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (1, "test", "facts", "", "kw", 0.5, 0, now, now),
        )
        sqlite_conn.commit()

        with patch.object(engine, "_api_request") as mock_api:
            mock_api.return_value = {"id": 300, "category": "facts", "importance": 0.5}
            result = engine.try_sync_store(1, "test", "facts", "", "kw", 0.5)

        assert result == 300

    def test_failure_enqueues_op(self, engine):
        with patch.object(engine, "_api_request") as mock_api:
            mock_api.side_effect = RuntimeError("Connection refused")
            result = engine.try_sync_store(1, "test", "facts", "", "kw", 0.5)

        assert result is None
        cursor = engine._conn.execute("SELECT COUNT(*) as cnt FROM pending_ops")
        assert cursor.fetchone()["cnt"] == 1


class TestTrySyncDelete:
    def test_success_returns_true(self, engine):
        with patch.object(engine, "_api_request") as mock_api:
            mock_api.return_value = {"deleted": 42, "preview": "test"}
            result = engine.try_sync_delete(42)

        assert result is True

    def test_failure_enqueues_op(self, engine):
        with patch.object(engine, "_api_request") as mock_api:
            mock_api.side_effect = RuntimeError("Connection refused")
            result = engine.try_sync_delete(42)

        assert result is False
        cursor = engine._conn.execute("SELECT COUNT(*) as cnt FROM pending_ops")
        assert cursor.fetchone()["cnt"] == 1


class TestSyncMeta:
    def test_last_sync_ts_none_initially(self, engine):
        assert engine.last_sync_ts is None

    def test_last_sync_ts_persists(self, engine):
        engine.last_sync_ts = "2026-03-14T12:00:00+00:00"
        assert engine.last_sync_ts == "2026-03-14T12:00:00+00:00"

    def test_api_available_initially_false(self, engine):
        assert engine.api_available is False


class TestFullSyncCycle:
    def test_store_sync_push_delete_pull(self, engine, sqlite_conn):
        """Full cycle: store locally → push to API → server deletes → pull removes local."""
        now = datetime.now(timezone.utc).isoformat()

        # 1. Store locally
        sqlite_conn.execute(
            "INSERT INTO memories (id, content, category, tags, expanded_keywords, importance, is_sensitive, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (1, "cycle test", "facts", "", "cycle test kw", 0.5, 0, now, now),
        )
        sqlite_conn.commit()

        # 2. Enqueue and push store
        engine.enqueue_store(1, "cycle test", "facts", "", "cycle test kw", 0.5)

        with patch.object(engine, "_api_request") as mock_api:
            mock_api.return_value = {"id": 500, "category": "facts", "importance": 0.5}
            engine._push_pending_ops()

        # Verify server_id set
        cursor = engine._conn.execute("SELECT server_id FROM memories WHERE id = 1")
        assert cursor.fetchone()["server_id"] == 500

        # 3. Server soft-deletes → pull removes local
        with patch.object(engine, "_api_request") as mock_api:
            mock_api.return_value = {
                "memories": [
                    {
                        "id": 500,
                        "content": "cycle test",
                        "category": "facts",
                        "tags": "",
                        "expanded_keywords": "cycle test kw",
                        "importance": 0.5,
                        "is_sensitive": False,
                        "created_at": now,
                        "updated_at": now,
                        "deleted_at": now,
                    }
                ],
                "server_time": now,
            }
            engine._pull_changes()

        # Should be gone locally
        cursor = engine._conn.execute("SELECT * FROM memories WHERE server_id = 500")
        assert cursor.fetchone() is None


class TestAuthFailureHandling:
    def test_auth_flag_set_on_401(self, engine):
        """401 from _api_request should set _auth_failed flag."""
        engine.enqueue_store(1, "test", "facts", "", "kw", 0.5)

        with patch.object(engine, "_api_request") as mock_api:
            mock_api.side_effect = urllib.error.HTTPError(
                url="http://fake", code=401, msg="Unauthorized", hdrs=None, fp=None
            )
            result = engine._push_pending_ops()

        assert result is False
        assert engine._auth_failed is True

    def test_auth_flag_set_on_403(self, engine):
        engine.enqueue_store(1, "test", "facts", "", "kw", 0.5)

        with patch.object(engine, "_api_request") as mock_api:
            mock_api.side_effect = urllib.error.HTTPError(
                url="http://fake", code=403, msg="Forbidden", hdrs=None, fp=None
            )
            result = engine._push_pending_ops()

        assert result is False
        assert engine._auth_failed is True

    def test_push_aborts_on_auth_failure(self, engine):
        """On 401, push should abort immediately — no further ops attempted."""
        engine.enqueue_store(1, "test1", "facts", "", "kw", 0.5)
        engine.enqueue_store(2, "test2", "facts", "", "kw", 0.5)

        with patch.object(engine, "_api_request") as mock_api:
            mock_api.side_effect = urllib.error.HTTPError(
                url="http://fake", code=401, msg="Unauthorized", hdrs=None, fp=None
            )
            engine._push_pending_ops()

        # Both ops should still be in queue (aborted before processing second)
        cursor = engine._conn.execute("SELECT COUNT(*) as cnt FROM pending_ops")
        assert cursor.fetchone()["cnt"] == 2

    def test_try_sync_store_queues_when_auth_failed(self, engine):
        """When auth is failed, try_sync_store should queue without attempting API call."""
        engine._auth_failed = True

        result = engine.try_sync_store(1, "test", "facts", "", "kw", 0.5)

        assert result is None
        cursor = engine._conn.execute("SELECT COUNT(*) as cnt FROM pending_ops")
        assert cursor.fetchone()["cnt"] == 1

    def test_try_sync_delete_queues_when_auth_failed(self, engine):
        engine._auth_failed = True

        result = engine.try_sync_delete(42)

        assert result is False
        cursor = engine._conn.execute("SELECT COUNT(*) as cnt FROM pending_ops")
        assert cursor.fetchone()["cnt"] == 1

    def test_check_auth_clears_flag_on_success(self, engine):
        engine._auth_failed = True

        with patch.object(engine, "_api_request") as mock_api:
            mock_api.return_value = {"status": "ok", "user_id": "test"}
            result = engine._check_auth()

        assert result is True
        assert engine._auth_failed is False

    def test_check_auth_stays_failed_on_401(self, engine):
        engine._auth_failed = True

        with patch.object(engine, "_api_request") as mock_api:
            mock_api.side_effect = urllib.error.HTTPError(
                url="http://fake", code=401, msg="Unauthorized", hdrs=None, fp=None
            )
            # Also mock urlopen for /health fallback
            with patch("urllib.request.urlopen") as mock_urlopen:
                mock_urlopen.return_value.__enter__ = MagicMock()
                mock_urlopen.return_value.__exit__ = MagicMock(return_value=False)
                result = engine._check_auth()

        assert result is False
        assert engine._auth_failed is True


class TestRetryCount:
    def test_retry_count_incremented_on_failure(self, engine):
        engine.enqueue_store(1, "test", "facts", "", "kw", 0.5)

        with patch.object(engine, "_api_request") as mock_api:
            mock_api.side_effect = RuntimeError("Connection refused")
            engine._push_pending_ops()

        cursor = engine._conn.execute("SELECT retry_count FROM pending_ops WHERE id = 1")
        assert cursor.fetchone()["retry_count"] == 1

    def test_op_skipped_after_max_retries(self, engine):
        engine.enqueue_store(1, "test", "facts", "", "kw", 0.5)
        # Set retry_count to max
        engine._conn.execute("UPDATE pending_ops SET retry_count = 5 WHERE id = 1")
        engine._conn.commit()

        with patch.object(engine, "_api_request") as mock_api:
            engine._push_pending_ops()

        # Op should be deleted (skipped), API never called
        cursor = engine._conn.execute("SELECT COUNT(*) as cnt FROM pending_ops")
        assert cursor.fetchone()["cnt"] == 0
        mock_api.assert_not_called()

    def test_retry_count_persists_across_pushes(self, engine):
        engine.enqueue_store(1, "test", "facts", "", "kw", 0.5)

        with patch.object(engine, "_api_request") as mock_api:
            mock_api.side_effect = RuntimeError("fail")
            engine._push_pending_ops()
            engine._push_pending_ops()
            engine._push_pending_ops()

        cursor = engine._conn.execute("SELECT retry_count FROM pending_ops WHERE id = 1")
        assert cursor.fetchone()["retry_count"] == 3


class TestDecoupledPushPull:
    def test_pull_runs_even_when_push_fails(self, engine):
        """Pull should execute even if push fails — they're decoupled."""
        engine.enqueue_store(1, "test", "facts", "", "kw", 0.5)
        now = datetime.now(timezone.utc).isoformat()

        call_count = 0

        def mock_api(method, path, body=None):
            nonlocal call_count
            call_count += 1
            if "POST" == method:
                raise RuntimeError("Push failed")
            # GET for pull
            return {
                "memories": [{
                    "id": 99, "content": "from server", "category": "facts",
                    "tags": "", "expanded_keywords": "", "importance": 0.5,
                    "is_sensitive": False, "created_at": now, "updated_at": now,
                    "deleted_at": None,
                }],
                "server_time": now,
            }

        with patch.object(engine, "_api_request", side_effect=mock_api):
            engine._sync_once()

        # Pull should have inserted the server memory
        cursor = engine._conn.execute("SELECT * FROM memories WHERE server_id = 99")
        assert cursor.fetchone() is not None

    def test_sync_once_returns_normally_on_partial_failure(self, engine):
        """If push fails but pull succeeds, _sync_once should not raise."""
        engine.enqueue_store(1, "test", "facts", "", "kw", 0.5)

        def mock_api(method, path, body=None):
            if method == "POST":
                raise RuntimeError("Push failed")
            return {"memories": [], "server_time": "2026-03-16T12:00:00+00:00"}

        with patch.object(engine, "_api_request", side_effect=mock_api):
            # Should not raise
            engine._sync_once()


class TestFullResync:
    def test_full_resync_inserts_server_records(self, engine):
        now = datetime.now(timezone.utc).isoformat()
        with patch.object(engine, "_api_request") as mock_api:
            mock_api.return_value = {
                "memories": [
                    {"id": 1, "content": "server mem 1", "category": "facts",
                     "tags": "", "expanded_keywords": "", "importance": 0.5,
                     "is_sensitive": False, "created_at": now, "updated_at": now},
                    {"id": 2, "content": "server mem 2", "category": "projects",
                     "tags": "", "expanded_keywords": "", "importance": 0.8,
                     "is_sensitive": False, "created_at": now, "updated_at": now},
                ],
                "server_time": now,
            }
            engine._full_resync()

        cursor = engine._conn.execute("SELECT COUNT(*) as cnt FROM memories")
        assert cursor.fetchone()["cnt"] == 2

    def test_full_resync_removes_stale_local_records(self, engine):
        """Local records with server_ids not on server should be deleted."""
        now = datetime.now(timezone.utc).isoformat()
        # Insert a local record with server_id=999 (not on server)
        engine._conn.execute(
            "INSERT INTO memories (content, category, tags, expanded_keywords, importance, "
            "is_sensitive, created_at, updated_at, server_id) VALUES (?,?,?,?,?,?,?,?,?)",
            ("stale", "facts", "", "", 0.5, 0, now, now, 999),
        )
        engine._conn.commit()

        with patch.object(engine, "_api_request") as mock_api:
            mock_api.return_value = {
                "memories": [
                    {"id": 1, "content": "current", "category": "facts",
                     "tags": "", "expanded_keywords": "", "importance": 0.5,
                     "is_sensitive": False, "created_at": now, "updated_at": now},
                ],
                "server_time": now,
            }
            engine._full_resync()

        # Stale record should be gone
        cursor = engine._conn.execute("SELECT * FROM memories WHERE server_id = 999")
        assert cursor.fetchone() is None
        # Current record should exist
        cursor = engine._conn.execute("SELECT * FROM memories WHERE server_id = 1")
        assert cursor.fetchone() is not None

    def test_full_resync_deletes_orphans_after_push(self, engine):
        """Orphans (server_id IS NULL) should be cleaned up after push attempt."""
        now = datetime.now(timezone.utc).isoformat()
        engine._conn.execute(
            "INSERT INTO memories (content, category, tags, expanded_keywords, importance, "
            "is_sensitive, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
            ("orphan", "facts", "", "", 0.5, 0, now, now),
        )
        engine._conn.commit()

        with patch.object(engine, "_api_request") as mock_api:
            mock_api.return_value = {
                "memories": [],
                "server_time": now,
            }
            engine._full_resync()

        cursor = engine._conn.execute("SELECT * FROM memories WHERE server_id IS NULL")
        assert cursor.fetchone() is None

    def test_full_resync_updates_last_sync_ts(self, engine):
        server_time = "2026-03-16T15:00:00+00:00"
        with patch.object(engine, "_api_request") as mock_api:
            mock_api.return_value = {"memories": [], "server_time": server_time}
            engine._full_resync()

        assert engine.last_sync_ts == server_time

    def test_full_resync_updates_existing_records(self, engine):
        now = datetime.now(timezone.utc).isoformat()
        engine._conn.execute(
            "INSERT INTO memories (content, category, tags, expanded_keywords, importance, "
            "is_sensitive, created_at, updated_at, server_id) VALUES (?,?,?,?,?,?,?,?,?)",
            ("old content", "facts", "", "", 0.5, 0, now, now, 10),
        )
        engine._conn.commit()

        with patch.object(engine, "_api_request") as mock_api:
            mock_api.return_value = {
                "memories": [
                    {"id": 10, "content": "new content", "category": "projects",
                     "tags": "updated", "expanded_keywords": "", "importance": 0.9,
                     "is_sensitive": False, "created_at": now, "updated_at": now},
                ],
                "server_time": now,
            }
            engine._full_resync()

        cursor = engine._conn.execute("SELECT * FROM memories WHERE server_id = 10")
        row = cursor.fetchone()
        assert row["content"] == "new content"
        assert row["category"] == "projects"
        assert row["importance"] == 0.9


class TestPushOrphans:
    def test_push_orphans_skips_duplicates(self, engine):
        now = datetime.now(timezone.utc).isoformat()
        # Insert orphan with content matching server
        engine._conn.execute(
            "INSERT INTO memories (content, category, tags, expanded_keywords, importance, "
            "is_sensitive, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
            ("duplicate content", "facts", "", "", 0.5, 0, now, now),
        )
        engine._conn.commit()

        call_log = []

        def mock_api(method, path, body=None):
            call_log.append((method, path))
            return {
                "memories": [{"id": 1, "content": "duplicate content", "category": "facts",
                              "tags": "", "expanded_keywords": "", "importance": 0.5,
                              "is_sensitive": False, "created_at": now, "updated_at": now}],
                "server_time": now,
            }

        with patch.object(engine, "_api_request", side_effect=mock_api):
            engine._push_orphans()

        # Should have called GET for sync but NOT POST (duplicate skipped)
        assert all(m != "POST" for m, _ in call_log)

    def test_push_orphans_posts_unique(self, engine):
        now = datetime.now(timezone.utc).isoformat()
        engine._conn.execute(
            "INSERT INTO memories (id, content, category, tags, expanded_keywords, importance, "
            "is_sensitive, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (1, "unique content", "facts", "", "", 0.5, 0, now, now),
        )
        engine._conn.commit()

        def mock_api(method, path, body=None):
            if method == "GET":
                return {"memories": [], "server_time": now}
            if method == "POST":
                return {"id": 100, "category": "facts", "importance": 0.5}
            return {}

        with patch.object(engine, "_api_request", side_effect=mock_api):
            engine._push_orphans()

        # Orphan should now have server_id
        cursor = engine._conn.execute("SELECT server_id FROM memories WHERE id = 1")
        assert cursor.fetchone()["server_id"] == 100


class TestGetCounts:
    def test_empty_counts(self, engine):
        counts = engine.get_counts()
        assert counts["total"] == 0
        assert counts["by_category"] == {}
        assert counts["orphans_no_server_id"] == 0
        assert counts["pending_ops"] == 0
        assert counts["auth_failed"] is False

    def test_counts_with_data(self, engine):
        now = datetime.now(timezone.utc).isoformat()
        engine._conn.execute(
            "INSERT INTO memories (content, category, tags, expanded_keywords, importance, "
            "is_sensitive, created_at, updated_at, server_id) VALUES (?,?,?,?,?,?,?,?,?)",
            ("mem1", "facts", "", "", 0.5, 0, now, now, 1),
        )
        engine._conn.execute(
            "INSERT INTO memories (content, category, tags, expanded_keywords, importance, "
            "is_sensitive, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
            ("orphan", "projects", "", "", 0.5, 0, now, now),
        )
        engine.enqueue_store(99, "queued", "facts", "", "", 0.5)
        engine._conn.commit()

        counts = engine.get_counts()
        assert counts["total"] == 2
        assert counts["by_category"]["facts"] == 1
        assert counts["by_category"]["projects"] == 1
        assert counts["orphans_no_server_id"] == 1
        assert counts["pending_ops"] == 1
