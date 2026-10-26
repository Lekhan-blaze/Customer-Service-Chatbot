"""Versioned vector store supporting pre-retrieval metadata filtering and persistence."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from support_bot.knowledge.embedder import BaseEmbedder, FakeEmbedder
from support_bot.knowledge.models import DocumentChunk


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


class VersionedVectorStore:
    """Stores chunks and embeddings for a specific knowledge base index version."""

    def __init__(
        self,
        version_id: str,
        storage_dir: Path | str,
        embedder: Optional[BaseEmbedder] = None,
    ) -> None:
        self.version_id = version_id
        self.storage_dir = Path(storage_dir) / self.version_id
        self.embedder = embedder or FakeEmbedder()
        self.chunks: List[DocumentChunk] = []
        self.embeddings: List[List[float]] = []

    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Embed and append new chunks to this index version."""
        if not chunks:
            return
        texts = [c.chunk_text for c in chunks]
        vecs = self.embedder.embed_documents(texts)
        self.chunks.extend(chunks)
        self.embeddings.extend(vecs)

    def search(
        self,
        query: str,
        top_k: int = 4,
        score_threshold: float = 0.2,
        metadata_filter: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> List[Tuple[DocumentChunk, float]]:
        """Search chunks with pre-retrieval metadata filtering.

        metadata_filter is applied BEFORE ranking, ensuring unauthorized,
        expired, or out-of-region passages never reach candidates or the LLM.
        """
        if not self.chunks or not self.embeddings:
            return []

        _STOP_WORDS = {
            "what", "is", "the", "a", "an", "for", "in", "to", "of", "and", "or", "on", 
            "can", "i", "you", "with", "this", "it", "my", "by", "as", "at", "are", "be", 
            "when", "how", "from", "do", "does", "did", "have", "has", "had", "will", "would"
        }
        strip_chars = '?,.:;!"\'()'
        q_words = set(w.strip(strip_chars).lower() for w in query.split() if w.strip(strip_chars).lower() not in _STOP_WORDS)

        q_vec = self.embedder.embed_query(query)
        candidates: List[Tuple[DocumentChunk, float]] = []

        for chunk, emb in zip(self.chunks, self.embeddings):
            # Apply pre-retrieval filter
            if metadata_filter is not None and not metadata_filter(chunk.metadata):
                continue

            # Ensure at least one meaningful keyword overlaps
            if q_words:
                c_words = set(w.strip(strip_chars).lower() for w in chunk.chunk_text.split() if w.strip(strip_chars).lower() not in _STOP_WORDS)
                if not q_words.intersection(c_words):
                    continue

            score = cosine_similarity(q_vec, emb)
            if score >= score_threshold:
                candidates.append((chunk, round(score, 4)))

        # Sort descending by similarity score
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]

    def save(self) -> Path:
        """Persist index chunks, embeddings, and metadata to disk."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "version_id": self.version_id,
            "count": len(self.chunks),
            "chunks": [
                {
                    "id": c.id,
                    "document_id": c.document_id,
                    "chunk_index": c.chunk_index,
                    "chunk_text": c.chunk_text,
                    "metadata": c.metadata,
                }
                for c in self.chunks
            ],
            "embeddings": self.embeddings,
        }
        target_file = self.storage_dir / "index.json"
        with open(target_file, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        return self.storage_dir

    @classmethod
    def load(
        cls,
        version_id: str,
        storage_dir: Path | str,
        embedder: Optional[BaseEmbedder] = None,
    ) -> VersionedVectorStore:
        """Load an existing index version from disk."""
        store = cls(version_id=version_id, storage_dir=storage_dir, embedder=embedder)
        target_file = store.storage_dir / "index.json"
        if not target_file.exists():
            raise FileNotFoundError(f"Index file not found for version {version_id} at {target_file}")

        with open(target_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        store.chunks = [
            DocumentChunk(
                id=c.get("id"),
                document_id=c["document_id"],
                chunk_index=c["chunk_index"],
                chunk_text=c["chunk_text"],
                metadata=c.get("metadata", {}),
            )
            for c in data.get("chunks", [])
        ]
        store.embeddings = data.get("embeddings", [])
        return store
