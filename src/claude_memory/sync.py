"""Background sync between local SQLite cache and remote API.

Uses only stdlib — no pip install required.
"""

import json
import logging
import sqlite3
import threading
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


class SyncEngine:
    """Background sync between local SQLite cache and remote API."""

    def __init__(self, db_path: str, api_base_url: str, api_key: str, sync_interval: int = 60):
        self.db_path = db_path
        self.api_base_url = api_base_url.rstrip("/")
        self.api_key = api_key
        self.sync_interval = sync_interval

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_sync_success = False

        # Own connection for thread safety
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, timeout=30.0, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._lock = threading.Lock()

        self._init_sync_tables()

    def _init_sync_tables(self) -> None:
        """Create sync-specific tables if they don't exist."""
        with self._lock:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS pending_ops (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    op_type TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sync_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
            """)
            # Add server_id column to memories if missing
            cursor = self._conn.execute("PRAGMA table_info(memories)")
            columns = {row["name"] for row in cursor.fetchall()}
            if "server_id" not in columns:
                self._conn.execute("ALTER TABLE memories ADD COLUMN server_id INTEGER")
                self._conn.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS idx_memories_server_id ON memories(server_id)"
                )
            self._conn.commit()

    @property
    def last_sync_ts(self) -> str | None:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT value FROM sync_meta WHERE key = 'last_sync_ts'"
            )
            row = cursor.fetchone()
            return row["value"] if row else None

    @last_sync_ts.setter
    def last_sync_ts(self, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO sync_meta (key, value) VALUES ('last_sync_ts', ?)",
                (value,),
            )
            self._conn.commit()

    @property
    def api_available(self) -> bool:
        return self._last_sync_success

    def start(self) -> None:
        """Start background sync thread (non-blocking)."""
        self._thread = threading.Thread(target=self._sync_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal background thread to stop and wait."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._conn.close()

    def _sync_loop(self) -> None:
        """Periodic sync loop running in background thread."""
        while not self._stop_event.is_set():
            self._stop_event.wait(self.sync_interval)
            if self._stop_event.is_set():
                break
            try:
                self._sync_once()
                self._last_sync_success = True
            except Exception as e:
                logger.warning("Sync cycle failed: %s", e)
                self._last_sync_success = False

    def _sync_once(self) -> None:
        """Push pending ops, then pull remote changes."""
        self._push_pending_ops()
        self._pull_changes()

    def _api_request(self, method: str, path: str, body: dict | None = None) -> dict:
        """Make an HTTP request to the memory API."""
        url = f"{self.api_base_url}{path}"
        data = json.dumps(body).encode() if body else None
        req = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())

    def _push_pending_ops(self) -> None:
        """Push queued operations to the API server."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT id, op_type, payload FROM pending_ops ORDER BY id"
            )
            ops = cursor.fetchall()

        for op in ops:
            op_id = op["id"]
            op_type = op["op_type"]
            payload = json.loads(op["payload"])

            try:
                if op_type == "store":
                    result = self._api_request("POST", "/api/memories", payload)
                    server_id = result.get("id")
                    if server_id and payload.get("local_id"):
                        with self._lock:
                            self._conn.execute(
                                "UPDATE memories SET server_id = ? WHERE id = ?",
                                (server_id, payload["local_id"]),
                            )
                            self._conn.commit()
                elif op_type == "delete":
                    server_id = payload.get("server_id")
                    if server_id:
                        try:
                            self._api_request("DELETE", f"/api/memories/{server_id}")
                        except RuntimeError as e:
                            if "404" in str(e):
                                pass  # Already deleted on server
                            else:
                                raise

                # Remove from pending queue on success
                with self._lock:
                    self._conn.execute("DELETE FROM pending_ops WHERE id = ?", (op_id,))
                    self._conn.commit()

            except Exception as e:
                logger.warning("Failed to push op %d (%s): %s", op_id, op_type, e)
                raise  # Propagate to mark sync as failed

    def _pull_changes(self) -> None:
        """Pull changes from server since last sync."""
        params = ""
        ts = self.last_sync_ts
        if ts:
            params = f"?since={ts}"

        result = self._api_request("GET", f"/api/memories/sync{params}")
        memories = result.get("memories", [])
        server_time = result.get("server_time")

        with self._lock:
            for mem in memories:
                server_id = mem["id"]
                deleted_at = mem.get("deleted_at")

                if deleted_at:
                    # Remove from local cache
                    self._conn.execute(
                        "DELETE FROM memories WHERE server_id = ?", (server_id,)
                    )
                else:
                    # Upsert by server_id (server wins)
                    existing = self._conn.execute(
                        "SELECT id FROM memories WHERE server_id = ?", (server_id,)
                    ).fetchone()

                    if existing:
                        self._conn.execute(
                            """UPDATE memories SET content = ?, category = ?, tags = ?,
                               expanded_keywords = ?, importance = ?, is_sensitive = ?,
                               updated_at = ? WHERE server_id = ?""",
                            (
                                mem["content"],
                                mem["category"],
                                mem.get("tags", ""),
                                mem.get("expanded_keywords", ""),
                                mem["importance"],
                                1 if mem.get("is_sensitive") else 0,
                                mem.get("updated_at", datetime.now(timezone.utc).isoformat()),
                                server_id,
                            ),
                        )
                    else:
                        self._conn.execute(
                            """INSERT INTO memories
                               (content, category, tags, expanded_keywords, importance,
                                is_sensitive, created_at, updated_at, server_id)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            (
                                mem["content"],
                                mem["category"],
                                mem.get("tags", ""),
                                mem.get("expanded_keywords", ""),
                                mem["importance"],
                                1 if mem.get("is_sensitive") else 0,
                                mem.get("created_at", datetime.now(timezone.utc).isoformat()),
                                mem.get("updated_at", datetime.now(timezone.utc).isoformat()),
                                server_id,
                            ),
                        )
            self._conn.commit()

        if server_time:
            self.last_sync_ts = server_time

    def enqueue_store(
        self,
        local_id: int,
        content: str,
        category: str,
        tags: str,
        expanded_keywords: str,
        importance: float,
        force_sensitive: bool = False,
    ) -> None:
        """Queue a store operation for later sync."""
        payload = {
            "local_id": local_id,
            "content": content,
            "category": category,
            "tags": tags,
            "expanded_keywords": expanded_keywords,
            "importance": importance,
            "force_sensitive": force_sensitive,
        }
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                "INSERT INTO pending_ops (op_type, payload, created_at) VALUES (?, ?, ?)",
                ("store", json.dumps(payload), now),
            )
            self._conn.commit()

    def enqueue_delete(self, server_id: int) -> None:
        """Queue a delete operation for later sync."""
        payload = {"server_id": server_id}
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                "INSERT INTO pending_ops (op_type, payload, created_at) VALUES (?, ?, ?)",
                ("delete", json.dumps(payload), now),
            )
            self._conn.commit()

    def try_sync_store(
        self,
        local_id: int,
        content: str,
        category: str,
        tags: str,
        expanded_keywords: str,
        importance: float,
        force_sensitive: bool = False,
    ) -> int | None:
        """Try to sync a store immediately. Returns server_id or None if failed."""
        try:
            result = self._api_request("POST", "/api/memories", {
                "content": content,
                "category": category,
                "tags": tags,
                "expanded_keywords": expanded_keywords,
                "importance": importance,
                "force_sensitive": force_sensitive,
            })
            server_id = result.get("id")
            if server_id:
                with self._lock:
                    self._conn.execute(
                        "UPDATE memories SET server_id = ? WHERE id = ?",
                        (server_id, local_id),
                    )
                    self._conn.commit()
            return server_id
        except Exception:
            self.enqueue_store(
                local_id, content, category, tags, expanded_keywords, importance, force_sensitive
            )
            return None

    def try_sync_delete(self, server_id: int) -> bool:
        """Try to sync a delete immediately. Returns True if successful."""
        try:
            self._api_request("DELETE", f"/api/memories/{server_id}")
            return True
        except Exception:
            self.enqueue_delete(server_id)
            return False
