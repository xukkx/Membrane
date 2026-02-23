import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer
from membrane.storage.sqlite import SQLiteStorage

logger = logging.getLogger(__name__)

class RAMStorage:
    def __init__(self, sqlite_storage: SQLiteStorage, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.sqlite = sqlite_storage
        self.model_name = embedding_model_name
        self._model = None

        # Temp vectors
        self.temp_vectors = np.empty((0, 384)) # shape (N, D)
        self.temp_id_to_index: Dict[str, int] = {}
        self.temp_index_to_id: Dict[int, str] = {}

        # KB vectors
        self.kb_vectors = np.empty((0, 384))
        self.kb_id_to_index: Dict[str, int] = {}
        self.kb_index_to_id: Dict[int, str] = {}

    @property
    def model(self):
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 384))
        embeddings = self.model.encode(texts)
        # Normalize embeddings for cosine similarity
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / (norm + 1e-9)

    def rebuild_index(self):
        logger.info("Rebuilding RAM index from SQLite...")

        # Rebuild Temp
        events = self.sqlite.fetch_all_temp_events()
        ids = [e['event_id'] for e in events]
        texts = [e['search_text'] for e in events]

        if texts:
            embeddings = self.encode(texts)
            self.temp_vectors = embeddings
            self.temp_id_to_index = {eid: i for i, eid in enumerate(ids)}
            self.temp_index_to_id = {i: eid for i, eid in enumerate(ids)}
        else:
            self.temp_vectors = np.empty((0, 384))
            self.temp_id_to_index = {}
            self.temp_index_to_id = {}

        # Rebuild KB
        items = self.sqlite.fetch_all_kb_items()
        kb_ids = [i['kb_item_id'] for i in items]
        kb_texts = [i['search_text'] for i in items]

        if kb_texts:
            kb_embeddings = self.encode(kb_texts)
            self.kb_vectors = kb_embeddings
            self.kb_id_to_index = {kid: i for i, kid in enumerate(kb_ids)}
            self.kb_index_to_id = {i: kid for i, kid in enumerate(kb_ids)}
        else:
            self.kb_vectors = np.empty((0, 384))
            self.kb_id_to_index = {}
            self.kb_index_to_id = {}

        logger.info(f"Rebuild complete. Temp: {len(ids)}, KB: {len(kb_ids)}")

    def add_temp_event(self, event_id: str, search_text: str):
        vector = self.encode([search_text])
        if event_id in self.temp_id_to_index:
            # Update existing? (Temp events usually immutable but strictly speaking...)
            # Since strict immutability for temp events is mostly true, but let's handle update just in case.
            idx = self.temp_id_to_index[event_id]
            self.temp_vectors[idx] = vector
        else:
            # Append
            if self.temp_vectors.shape[0] == 0:
                self.temp_vectors = vector
            else:
                self.temp_vectors = np.vstack([self.temp_vectors, vector])
            idx = self.temp_vectors.shape[0] - 1
            self.temp_id_to_index[event_id] = idx
            self.temp_index_to_id[idx] = event_id

    def upsert_kb_item(self, kb_item_id: str, search_text: str):
        vector = self.encode([search_text])
        if kb_item_id in self.kb_id_to_index:
            idx = self.kb_id_to_index[kb_item_id]
            self.kb_vectors[idx] = vector
        else:
            if self.kb_vectors.shape[0] == 0:
                self.kb_vectors = vector
            else:
                self.kb_vectors = np.vstack([self.kb_vectors, vector])
            idx = self.kb_vectors.shape[0] - 1
            self.kb_id_to_index[kb_item_id] = idx
            self.kb_index_to_id[idx] = kb_item_id

    def get_temp_vector(self, event_id: str) -> Optional[np.ndarray]:
        idx = self.temp_id_to_index.get(event_id)
        if idx is not None:
            return self.temp_vectors[idx]
        return None

    def get_kb_vector(self, kb_item_id: str) -> Optional[np.ndarray]:
        idx = self.kb_id_to_index.get(kb_item_id)
        if idx is not None:
            return self.kb_vectors[idx]
        return None
