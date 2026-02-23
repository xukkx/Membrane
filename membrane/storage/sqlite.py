import sqlite3
import json
import logging
from typing import Optional, List, Dict, Any, Tuple
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

            # temp_events table
            cursor.execute("""
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
            """)

            # Index for temp retrieval
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_temp_events_scope
                ON temp_events (project_id, agent_id, session_id, ts_epoch_ms)
            """)

            # kb_items table
            cursor.execute("""
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
            """)

            # Explicit unique index for clarity/readability
            cursor.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_kb_items_unique_id
                ON kb_items (kb_item_id)
            """)

            # FTS5 for temp_events
            # Using external content tables is cleaner but manual sync is also fine.
            # I'll stick to manual sync for explicit control.
            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS temp_events_fts USING fts5(
                    search_text,
                    event_id UNINDEXED,
                    project_id UNINDEXED,
                    agent_id UNINDEXED,
                    session_id UNINDEXED
                )
            """)

            # FTS5 for kb_items
            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS kb_items_fts USING fts5(
                    search_text,
                    kb_item_id UNINDEXED,
                    project_id UNINDEXED
                )
            """)

            conn.commit()

    def add_event(self, event: Dict[str, Any], search_text: str):
        """Inserts an event into temp_events and temp_events_fts."""
        scope = event['scope']
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO temp_events (
                    event_id, project_id, agent_id, session_id, thread_id, task_id,
                    type, text, search_text, tags_json, entities_json, ts_epoch_ms, ts_iso_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
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
                event.get('ts_iso_utc')
            ))

            cursor.execute("""
                INSERT INTO temp_events_fts (rowid, search_text, event_id, project_id, agent_id, session_id)
                VALUES (last_insert_rowid(), ?, ?, ?, ?, ?)
            """, (
                search_text,
                event['event_id'],
                scope['project_id'],
                scope['agent_id'],
                scope['session_id']
            ))
            conn.commit()

    def upsert_kb_item(self, item: Dict[str, Any], search_text: str):
        """Upserts a KB item into kb_items and updates kb_items_fts."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Check if exists first? Or replace?
            # Upsert using INSERT OR REPLACE logic
            cursor.execute("""
                INSERT OR REPLACE INTO kb_items (
                    kb_item_id, project_id, kind, content_json, search_text,
                    provenance_json, confidence, status, conflicts_json, supersedes_json,
                    ts_epoch_ms, ts_iso_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
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
                item.get('ts_iso_utc')
            ))

            # Update FTS
            # We need to handle updates. FTS doesn't have a direct primary key link unless using external content.
            # For simplicity, delete and insert.
            cursor.execute("DELETE FROM kb_items_fts WHERE kb_item_id = ?", (item['kb_item_id'],))
            cursor.execute("""
                INSERT INTO kb_items_fts (search_text, kb_item_id, project_id)
                VALUES (?, ?, ?)
            """, (
                search_text,
                item['kb_item_id'],
                item['project_id']
            ))
            conn.commit()

    def get_kb_item(self, kb_item_id: str) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM kb_items WHERE kb_item_id = ?", (kb_item_id,))
            row = cursor.fetchone()
            if row:
                d = dict(row)
                # Deserialize JSON fields
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
                    'task_id': d['task_id']
                }
                d['tags'] = json.loads(d['tags_json']) if d['tags_json'] else []
                d['entities'] = json.loads(d['entities_json']) if d['entities_json'] else []
                return d
            return None

    def search_temp_fts(self, query: str, limit: int = 200) -> List[Dict[str, Any]]:
        # FTS5 query syntax
        # We also need to filter by scope, but FTS5 supports UNINDEXED columns which can be used in WHERE?
        # Yes, UNINDEXED columns can be used in queries but are not full-text indexed.
        pass
        # Actually, I'll implement a method that takes additional WHERE clauses or scope parameters
        # But retrieval logic usually handles the FTS query construction.
        # I'll expose a raw query method or a structured one.
        # For strict isolation, I should probably enforce scope here?
        # But `retrieval.py` handles the logic. I'll provide a generic query method.

    def fetch_all_kb_items(self) -> List[Dict[str, Any]]:
        """Used for reindexing."""
        items = []
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM kb_items")
            rows = cursor.fetchall()
            for row in rows:
                d = dict(row)
                d['content'] = json.loads(d['content_json'])
                d['provenance_event_ids'] = json.loads(d['provenance_json'])
                d['conflicts_with'] = json.loads(d['conflicts_json']) if d['conflicts_json'] else []
                d['supersedes'] = json.loads(d['supersedes_json']) if d['supersedes_json'] else []
                items.append(d)
        return items

    def fetch_all_temp_events(self) -> List[Dict[str, Any]]:
        """Used for reindexing."""
        events = []
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM temp_events")
            rows = cursor.fetchall()
            for row in rows:
                d = dict(row)
                d['scope'] = {
                    'project_id': d['project_id'],
                    'agent_id': d['agent_id'],
                    'session_id': d['session_id'],
                    'thread_id': d['thread_id'],
                    'task_id': d['task_id']
                }
                d['tags'] = json.loads(d['tags_json']) if d['tags_json'] else []
                d['entities'] = json.loads(d['entities_json']) if d['entities_json'] else []
                events.append(d)
        return events

    def search_temp_events(self, query: str, project_id: str, agent_id: str, session_id: str, limit: int = 200) -> List[Dict[str, Any]]:
        """Search temp events using FTS5, strictly scoped."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            # FTS5 match query
            # We enforce scope using filters on the FTS table UNINDEXED columns or by joining?
            # FTS5 tables support filtering on unindexed columns efficiently enough for this scale?
            # Actually, standard practice is: SELECT * FROM temp_events_fts WHERE temp_events_fts MATCH ? AND project_id = ? ...

            fts_query = query
            # Sanitize query for FTS5? Assuming basic query for now.
            # Ideally use parameterized query for MATCH, but SQLite's parameter binding for MATCH is tricky.
            # Usually strict MATCH ? is fine.

            sql = """
                SELECT bm25(temp_events_fts) AS fts_rank, rowid, * FROM temp_events_fts
                WHERE temp_events_fts MATCH ?
                AND project_id = ?
                AND agent_id = ?
                AND session_id = ?
                ORDER BY fts_rank ASC
                LIMIT ?
            """
            cursor.execute(sql, (fts_query, project_id, agent_id, session_id, limit))
            rows = cursor.fetchall()

            results = []
            for row in rows:
                # We need to fetch the full event data.
                # We can join with the main table or fetch by event_id.
                # Since we have event_id in FTS table (UNINDEXED), we can use it.
                event_id = row['event_id']

                # Fetch full details
                cursor.execute("SELECT * FROM temp_events WHERE event_id = ?", (event_id,))
                detail_row = cursor.fetchone()
                if detail_row:
                    d = dict(detail_row)
                    d['scope'] = {
                        'project_id': d['project_id'],
                        'agent_id': d['agent_id'],
                        'session_id': d['session_id'],
                        'thread_id': d['thread_id'],
                        'task_id': d['task_id']
                    }
                    d['tags'] = json.loads(d['tags_json']) if d['tags_json'] else []
                    d['entities'] = json.loads(d['entities_json']) if d['entities_json'] else []
                    d['fts_rank'] = row['fts_rank']
                    results.append(d)
            return results

    def search_kb_items(self, query: str, project_id: str, limit: int = 200) -> List[Dict[str, Any]]:
        """Search KB items using FTS5, scoped to project."""
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
                kb_item_id = row['kb_item_id']
                cursor.execute("SELECT * FROM kb_items WHERE kb_item_id = ?", (kb_item_id,))
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
