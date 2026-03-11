import os
import uuid
import lancedb
from typing import List, Optional
from langchain_community.vectorstores import LanceDB

class LanceDBSessionMemory:
    def __init__(self, embeddings, k: int = 3, score_threshold: float = 0.35):
        self.embeddings = embeddings
        self.session_id = str(uuid.uuid4())
        self.session_db_path = f"/tmp/rag_sessions/{self.session_id}"
        os.makedirs(self.session_db_path, exist_ok=True)
        
        self.db = lancedb.connect(self.session_db_path)
        self.history_vectorstore: Optional[LanceDB] = None
        self.k = k
        self.score_threshold = score_threshold

    def add_history(self, history_items: List[str]):
        if not history_items:
            return
        table_name = "history_table"
        if self.history_vectorstore is None:
            self.history_vectorstore = LanceDB.from_texts(
                history_items, 
                self.embeddings, 
                connection=self.db, 
                table_name=table_name
            )
        else:
            self.history_vectorstore.add_texts(history_items)

    def get_retriever(self):
        if self.history_vectorstore is None:
            return None
        return self.history_vectorstore.as_retriever(
            search_type="similarity_score_threshold", 
            search_kwargs={"k": self.k, "score_threshold": self.score_threshold}
        )
