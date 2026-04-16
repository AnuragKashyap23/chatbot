"""
FAQ Vector Database module using ChromaDB.

Loads FAQ data from faq_data.json, embeds questions into a ChromaDB collection,
and provides similarity search with LRU caching for incoming user queries.
"""

from __future__ import annotations

import json
import os
import logging
from typing import Optional
from collections import OrderedDict
import chromadb
from chromadb.utils import embedding_functions

log = logging.getLogger(__name__)

FAQ_DATA_PATH = os.path.join(os.path.dirname(__file__), "faq_data.json")
CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "faq_collection"

SIMILARITY_THRESHOLD = 1.05
CACHE_MAX_SIZE = 512

_client: Optional[chromadb.PersistentClient] = None
_collection: Optional[chromadb.Collection] = None

_search_cache: OrderedDict[str, Optional[dict]] = OrderedDict()


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


def search_faq(query: str, n_results: int = 1) -> Optional[dict]:
    """Search the FAQ vector DB for a matching question.

    Uses an LRU cache keyed by normalized query to avoid re-embedding
    repeated or near-identical queries (e.g. same user retrying).

    Returns a dict with 'question', 'answer', and 'distance' if a match
    is found within the similarity threshold, otherwise None.
    """
    if _collection is None:
        init_vectordb()

    cache_key = query.strip().lower()
    if cache_key in _search_cache:
        _search_cache.move_to_end(cache_key)
        log.info(f"FAQ cache hit for: '{query[:50]}...'")
        return _search_cache[cache_key]

    results = _collection.query(query_texts=[query], n_results=n_results)

    if not results["ids"] or not results["ids"][0]:
        _cache_put(cache_key, None)
        return None

    distance = results["distances"][0][0]
    metadata = results["metadatas"][0][0]

    log.info(
        f"FAQ search: distance={distance:.4f}, threshold={SIMILARITY_THRESHOLD}, "
        f"matched_q='{metadata['question'][:60]}...'"
    )

    if distance <= SIMILARITY_THRESHOLD:
        result = {
            "question": metadata["question"],
            "answer": metadata["answer"],
            "distance": round(distance, 4),
        }
    else:
        result = None

    _cache_put(cache_key, result)
    return result


def _cache_put(key: str, value: Optional[dict]):
    _search_cache[key] = value
    if len(_search_cache) > CACHE_MAX_SIZE:
        _search_cache.popitem(last=False)


def rebuild_vectordb():
    """Force rebuild the vector DB (useful after FAQ data changes)."""
    global _collection
    if _client is not None:
        try:
            _client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    _collection = None
    _search_cache.clear()
    return init_vectordb()
