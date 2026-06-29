import os
from typing import List, Dict, Any, Optional
import google.generativeai as genai
import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from config.settings import Settings
from utils.logging import logger

class GeminiEmbeddingFunction(EmbeddingFunction):
    """
    Custom embedding function for ChromaDB that uses Google's text-embedding-004 model.
    """
    def __call__(self, input: Documents) -> Embeddings:
        if not Settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set. Cannot compute embeddings.")
        
        try:
            genai.configure(api_key=Settings.GEMINI_API_KEY)
            # Embed content
            # genai.embed_content accepts a list of strings or a single string
            result = genai.embed_content(
                model=Settings.EMBEDDING_MODEL,
                content=input,
                task_type="retrieval_document"
            )
            # result['embedding'] is a list of lists if input was a list of strings
            return result['embedding']
        except Exception as e:
            logger.error(f"[GeminiEmbeddingFunction] Error generating embeddings: {e}")
            raise e

class RAGEngine:
    def __init__(self):
        self.client = None
        self.collection = None
        self.collection_name = "fact_checking_kb"
        self._init_db()

    def _init_db(self):
        """Initializes the Chroma persistent client and gets or creates the collection."""
        try:
            # Ensure the database directory exists
            os.makedirs(Settings.VECTOR_DB_DIR, exist_ok=True)
            logger.info(f"[RAGEngine] Initializing ChromaDB persistent client at: {Settings.VECTOR_DB_DIR}")
            
            self.client = chromadb.PersistentClient(path=Settings.VECTOR_DB_DIR)
            self.emb_fn = GeminiEmbeddingFunction()
            
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.emb_fn
            )
            logger.info(f"[RAGEngine] Database and collection '{self.collection_name}' initialized successfully.")
        except Exception as e:
            logger.error(f"[RAGEngine] Failed to initialize database: {e}")
            self.collection = None

    def add_reference_documents(self, documents: List[str], metadatas: List[Dict[str, Any]] = None, ids: List[str] = None):
        """
        Splits documents and adds them to the vector store.
        """
        if not self.collection:
            self._init_db()
            if not self.collection:
                logger.error("[RAGEngine] Collection not initialized. Cannot add documents.")
                return False

        if not ids:
            ids = [f"doc_{i}_{int(os.urandom(3).hex(), 16)}" for i in range(len(documents))]
        if not metadatas:
            metadatas = [{"source": "user_upload"} for _ in range(len(documents))]

        logger.info(f"[RAGEngine] Adding {len(documents)} document chunks to the database...")
        try:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.info("[RAGEngine] Document chunks added successfully.")
            return True
        except Exception as e:
            logger.error(f"[RAGEngine] Error adding documents to ChromaDB: {e}")
            return False

    def query_kb(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """
        Queries the local vector store for the closest match chunks.
        Returns a list of dictionaries with document context.
        """
        if not self.collection:
            self._init_db()
            if not self.collection:
                logger.warning("[RAGEngine] Vector database collection not available.")
                return []

        logger.info(f"[RAGEngine] Querying knowledge base for: '{query}'")
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(n_results, self.collection.count() or n_results)
            )
            
            formatted_results = []
            if results and 'documents' in results and results['documents']:
                docs = results['documents'][0]
                metas = results['metadatas'][0] if 'metadatas' in results and results['metadatas'] else [{}] * len(docs)
                ids = results['ids'][0] if 'ids' in results and results['ids'] else [""] * len(docs)
                
                for doc, meta, doc_id in zip(docs, metas, ids):
                    formatted_results.append({
                        "title": meta.get("title", meta.get("source", "Local KB")),
                        "url": meta.get("url", f"local://{doc_id}"),
                        "snippet": doc
                    })
            logger.info(f"[RAGEngine] Retrieved {len(formatted_results)} results from local KB.")
            return formatted_results
        except Exception as e:
            logger.error(f"[RAGEngine] Error querying ChromaDB: {e}")
            return []

    def get_document_count(self) -> int:
        """Returns the number of documents in the collection."""
        if not self.collection:
            return 0
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"[RAGEngine] Error getting collection count: {e}")
            return 0

    def clear_kb(self):
        """Clears the collection."""
        if not self.client:
            return
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.emb_fn
            )
            logger.info("[RAGEngine] Knowledge base cleared successfully.")
            return True
        except Exception as e:
            logger.error(f"[RAGEngine] Error clearing database: {e}")
            return False
