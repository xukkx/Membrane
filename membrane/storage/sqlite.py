import sqlite3
import json
import logging
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class SQLiteStorage:
    def __init__(self, db_path: str = "membrane.db"):
        self.db_path = db_path
        self._ensure_fts5()
        self._init_db()

    @contextmanager
    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _ensure_fts5(self):
        """Check if FTS5 is available. Fail fast if not."""
        try:
            conn = sqlite3.connect(":memory:")
            cursor = conn.cursor()
            cursor.execute("CREATE VIRTUAL TABLE fts_check USING fts5(content)")
            cursor.close()
            conn.close()
        except sqlite3.OperationalError:
            raise RuntimeError("SQLite FTS5 extension is not available. Membrane requires FTS5.")

    def _init_db(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS temp_events (
                    event_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    thread_id TEXT,
                    task_id TEXT,
                    type TEXT NOT NULL,
                    text TEXT NOT NULL,
                    search_text TEXT NOT NULL,
                    tags_json TEXT,
                    entities_json TEXT,
                    ts_epoch_ms INTEGER NOT NULL,
                    ts_iso_utc TEXT
                )
                """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_temp_events_scope
                ON temp_events (project_id, agent_id, session_id, ts_epoch_ms)
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS kb_items (
                    kb_item_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    content_json TEXT NOT NULL,
                    search_text TEXT NOT NULL,
                    provenance_json TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    status TEXT NOT NULL,
                    conflicts_json TEXT,
                    supersedes_json TEXT,
                    ts_epoch_ms INTEGER NOT NULL,
                    ts_iso_utc TEXT
                )
                """
            )

            cursor.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_kb_items_unique_id
                ON kb_items (kb_item_id)
                """
            )

            # Migration-safe reset of FTS/triggers from any previous schema.
            cursor.execute("DROP TRIGGER IF EXISTS trg_temp_events_ai")
            cursor.execute("DROP TRIGGER IF EXISTS trg_temp_events_au")
            cursor.execute("DROP TRIGGER IF EXISTS trg_temp_events_ad")
            cursor.execute("DROP TRIGGER IF EXISTS trg_kb_items_ai")
            cursor.execute("DROP TRIGGER IF EXISTS trg_kb_items_au")
            cursor.execute("DROP TRIGGER IF EXISTS trg_kb_items_ad")
            cursor.execute("DROP TABLE IF EXISTS temp_events_fts")
            cursor.execute("DROP TABLE IF EXISTS kb_items_fts")

            cursor.execute(
                """
                CREATE VIRTUAL TABLE temp_events_fts USING fts5(
                    search_text,
                    event_id UNINDEXED,
                    project_id UNINDEXED,
                    agent_id UNINDEXED,
                    session_id UNINDEXED,
                    content='temp_events',
                    content_rowid='rowid'
                )
                """
            )

            cursor.execute(
                """
                CREATE VIRTUAL TABLE kb_items_fts USING fts5(
                    search_text,
                    kb_item_id UNINDEXED,
                    project_id UNINDEXED,
                    content='kb_items',
                    content_rowid='rowid'
                )
                """
            )

            cursor.execute(
                """
                CREATE TRIGGER trg_temp_events_ai AFTER INSERT ON temp_events BEGIN
                    insert into temp_events_fts(rowid, search_text, event_id, project_id, agent_id, session_id)
                    VALUES (new.rowid, new.search_text, new.event_id, new.project_id, new.agent_id, new.session_id);
                END;
                """
            )
            cursor.execute(
                """
                CREATE TRIGGER trg_temp_events_ad AFTER DELETE ON temp_events BEGIN
                    insert into temp_events_fts(temp_events_fts, rowid, search_text, event_id, project_id, agent_id, session_id)
                    VALUES('delete', old.rowid, old.search_text, old.event_id, old.project_id, old.agent_id, old.session_id);
                END;
                """
            )
            cursor.execute(
                """
                CREATE TRIGGER trg_temp_events_au AFTER UPDATE ON temp_events BEGIN
                    insert into temp_events_fts(temp_events_fts, rowid, search_text, event_id, project_id, agent_id, session_id)
                    VALUES('delete', old.rowid, old.search_text, old.event_id, old.project_id, old.agent_id, old.session_id);
                    insert into temp_events_fts(rowid, search_text, event_id, project_id, agent_id, session_id)
                    VALUES (new.rowid, new.search_text, new.event_id, new.project_id, new.agent_id, new.session_id);
                END;
                """
            )

            cursor.execute(
                """
                CREATE TRIGGER trg_kb_items_ai AFTER INSERT ON kb_items BEGIN
                    insert into kb_items_fts(rowid, search_text, kb_item_id, project_id)
                    VALUES (new.rowid, new.search_text, new.kb_item_id, new.project_id);
                END;
                """
            )
            cursor.execute(
                """
                CREATE TRIGGER trg_kb_items_ad AFTER DELETE ON kb_items BEGIN
                    insert into kb_items_fts(kb_items_fts, rowid, search_text, kb_item_id, project_id)
                    VALUES('delete', old.rowid, old.search_text, old.kb_item_id, old.project_id);
                END;
                """
            )
            cursor.execute(
                """
                CREATE TRIGGER trg_kb_items_au AFTER UPDATE ON kb_items BEGIN
                    insert into kb_items_fts(kb_items_fts, rowid, search_text, kb_item_id, project_id)
                    VALUES('delete', old.rowid, old.search_text, old.kb_item_id, old.project_id);
                    insert into kb_items_fts(rowid, search_text, kb_item_id, project_id)
                    VALUES (new.rowid, new.search_text, new.kb_item_id, new.project_id);
                END;
                """
            )

            # Rebuild from external content tables.
            cursor.execute("insert into temp_events_fts(temp_events_fts) VALUES ('rebuild')")
            cursor.execute("insert into kb_items_fts(kb_items_fts) VALUES ('rebuild')")

            conn.commit()

    def add_event(self, event: Dict[str, Any], search_text: str):
        """Inserts an event into temp_events (FTS sync via triggers)."""
        scope = event['scope']
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO temp_events (
                    event_id, project_id, agent_id, session_id, thread_id, task_id,
                    type, text, search_text, tags_json, entities_json, ts_epoch_ms, ts_iso_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event['event_id'],
                    scope['project_id'],
                    scope['agent_id'],
                    scope['session_id'],
                    scope.get('thread_id'),
                    scope.get('task_id'),
                    event['type'],
                    event['text'],
                    search_text,
                    json.dumps(event.get('tags', [])),
                    json.dumps(event.get('entities', [])),
                    event['ts_epoch_ms'],
                    event.get('ts_iso_utc'),
                ),
            )
            conn.commit()

    def upsert_kb_item(self, item: Dict[str, Any], search_text: str):
        """Upserts a KB item into kb_items (FTS sync via triggers)."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO kb_items (
                    kb_item_id, project_id, kind, content_json, search_text,
                    provenance_json, confidence, status, conflicts_json, supersedes_json,
                    ts_epoch_ms, ts_iso_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item['kb_item_id'],
                    item['project_id'],
                    item['kind'],
                    json.dumps(item['content']),
                    search_text,
                    json.dumps(item['provenance_event_ids']),
                    item['confidence'],
                    item['status'],
                    json.dumps(item.get('conflicts_with', [])),
                    json.dumps(item.get('supersedes', [])),
                    item['ts_epoch_ms'],
                    item.get('ts_iso_utc'),
                ),
            )
            conn.commit()

    def get_kb_item(self, kb_item_id: str) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM kb_items WHERE kb_item_id = ?", (kb_item_id,))
            row = cursor.fetchone()
            if row:
                d = dict(row)
                d['content'] = json.loads(d['content_json'])
                d['provenance_event_ids'] = json.loads(d['provenance_json'])
                d['conflicts_with'] = json.loads(d['conflicts_json']) if d['conflicts_json'] else []
                d['supersedes'] = json.loads(d['supersedes_json']) if d['supersedes_json'] else []
                return d
            return None

    def get_event(self, event_id: str) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM temp_events WHERE event_id = ?", (event_id,))
            row = cursor.fetchone()
            if row:
                d = dict(row)
                d['scope'] = {
                    'project_id': d['project_id'],
                    'agent_id': d['agent_id'],
                    'session_id': d['session_id'],
                    'thread_id': d['thread_id'],
                    'task_id': d['task_id'],
                }
                d['tags'] = json.loads(d['tags_json']) if d['tags_json'] else []
                d['entities'] = json.loads(d['entities_json']) if d['entities_json'] else []
                return d
            return None

    def fetch_all_kb_items(self) -> List[Dict[str, Any]]:
        items = []
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM kb_items")
            for row in cursor.fetchall():
                d = dict(row)
                d['content'] = json.loads(d['content_json'])
                d['provenance_event_ids'] = json.loads(d['provenance_json'])
                d['conflicts_with'] = json.loads(d['conflicts_json']) if d['conflicts_json'] else []
                d['supersedes'] = json.loads(d['supersedes_json']) if d['supersedes_json'] else []
                items.append(d)
        return items

    def fetch_all_temp_events(self) -> List[Dict[str, Any]]:
        events = []
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM temp_events")
            for row in cursor.fetchall():
                d = dict(row)
                d['scope'] = {
                    'project_id': d['project_id'],
                    'agent_id': d['agent_id'],
                    'session_id': d['session_id'],
                    'thread_id': d['thread_id'],
                    'task_id': d['task_id'],
                }
                d['tags'] = json.loads(d['tags_json']) if d['tags_json'] else []
                d['entities'] = json.loads(d['entities_json']) if d['entities_json'] else []
                events.append(d)
        return events

    def search_temp_events(self, query: str, project_id: str, agent_id: str, session_id: str, limit: int = 200) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            sql = """
                SELECT bm25(temp_events_fts) AS fts_rank, rowid, * FROM temp_events_fts
                WHERE temp_events_fts MATCH ?
                AND project_id = ?
                AND agent_id = ?
                AND session_id = ?
                ORDER BY fts_rank ASC
                LIMIT ?
            """
            cursor.execute(sql, (query, project_id, agent_id, session_id, limit))
            rows = cursor.fetchall()

            results = []
            for row in rows:
                cursor.execute("SELECT * FROM temp_events WHERE event_id = ?", (row['event_id'],))
                detail_row = cursor.fetchone()
                if detail_row:
                    d = dict(detail_row)
                    d['scope'] = {
                        'project_id': d['project_id'],
                        'agent_id': d['agent_id'],
                        'session_id': d['session_id'],
                        'thread_id': d['thread_id'],
                        'task_id': d['task_id'],
                    }
                    d['tags'] = json.loads(d['tags_json']) if d['tags_json'] else []
                    d['entities'] = json.loads(d['entities_json']) if d['entities_json'] else []
                    d['fts_rank'] = row['fts_rank']
                    results.append(d)
            return results

    def search_kb_items(self, query: str, project_id: str, limit: int = 200) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            sql = """
                SELECT bm25(kb_items_fts) AS fts_rank, rowid, * FROM kb_items_fts
                WHERE kb_items_fts MATCH ?
                AND project_id = ?
                ORDER BY fts_rank ASC
                LIMIT ?
            """
            cursor.execute(sql, (query, project_id, limit))
            rows = cursor.fetchall()

            results = []
            for row in rows:
                cursor.execute("SELECT * FROM kb_items WHERE kb_item_id = ?", (row['kb_item_id'],))
                detail_row = cursor.fetchone()
                if detail_row:
                    d = dict(detail_row)
                    d['content'] = json.loads(d['content_json'])
                    d['provenance_event_ids'] = json.loads(d['provenance_json'])
                    d['conflicts_with'] = json.loads(d['conflicts_json']) if d['conflicts_json'] else []
                    d['supersedes'] = json.loads(d['supersedes_json']) if d['supersedes_json'] else []
                    d['fts_rank'] = row['fts_rank']
                    results.append(d)
            return results
