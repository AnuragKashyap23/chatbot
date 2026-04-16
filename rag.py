"""
FAQ Vector Database module using ChromaDB.

Loads FAQ data from faq_data.json, embeds questions into a ChromaDB collection,
and provides similarity search for incoming user queries.
"""

import json
import os
import logging
import chromadb
from chromadb.utils import embedding_functions

log = logging.getLogger(__name__)

FAQ_DATA_PATH = os.path.join(os.path.dirname(__file__), "faq_data.json")
CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "faq_collection"

SIMILARITY_THRESHOLD = 1.05

_client: chromadb.PersistentClient | None = None
_collection: chromadb.Collection | None = None


def _load_faq_data() -> list[dict]:
    with open(FAQ_DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def init_vectordb() -> chromadb.Collection:
    """Initialize ChromaDB with FAQ embeddings. Safe to call multiple times."""
    global _client, _collection

    if _collection is not None:
        return _collection

    log.info("Initializing ChromaDB vector store...")

    _client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    embedding_fn = embedding_functions.DefaultEmbeddingFunction()

    _collection = _client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    faq_data = _load_faq_data()

    existing = _collection.count()
    if existing >= len(faq_data):
        log.info(f"ChromaDB already has {existing} FAQ entries, skipping insert.")
        return _collection

    if existing > 0:
        _client.delete_collection(COLLECTION_NAME)
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    ids = [item["id"] for item in faq_data]
    documents = [item["question"] for item in faq_data]
    metadatas = [{"answer": item["answer"], "question": item["question"]} for item in faq_data]

    _collection.add(ids=ids, documents=documents, metadatas=metadatas)
    log.info(f"Loaded {len(faq_data)} FAQ entries into ChromaDB.")

    return _collection


def search_faq(query: str, n_results: int = 1) -> dict | None:
    """Search the FAQ vector DB for a matching question.

    Returns a dict with 'question', 'answer', and 'distance' if a match
    is found within the similarity threshold, otherwise None.
    """
    if _collection is None:
        init_vectordb()

    results = _collection.query(query_texts=[query], n_results=n_results)

    if not results["ids"] or not results["ids"][0]:
        return None

    distance = results["distances"][0][0]
    metadata = results["metadatas"][0][0]

    log.info(
        f"FAQ search: distance={distance:.4f}, threshold={SIMILARITY_THRESHOLD}, "
        f"matched_q='{metadata['question'][:60]}...'"
    )

    if distance <= SIMILARITY_THRESHOLD:
        return {
            "question": metadata["question"],
            "answer": metadata["answer"],
            "distance": round(distance, 4),
        }

    return None


def rebuild_vectordb():
    """Force rebuild the vector DB (useful after FAQ data changes)."""
    global _collection
    if _client is not None:
        try:
            _client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    _collection = None
    return init_vectordb()
