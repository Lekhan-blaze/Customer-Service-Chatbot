"""Embedding interfaces and fake/real implementations for Knowledge Base vectorization."""

from abc import ABC, abstractmethod
import hashlib
import math
from typing import List

_STOP_WORDS = {
    "what", "is", "the", "a", "an", "for", "in", "to", "of", "and", "or", "on", 
    "can", "i", "you", "with", "this", "it", "my", "by", "as", "at", "are", "be", 
    "when", "how", "from", "do", "does", "did", "have", "has", "had", "will", "would"
}


class BaseEmbedder(ABC):
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding vector for a single query string."""
        raise NotImplementedError

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embedding vectors for a list of document strings."""
        raise NotImplementedError


class FakeEmbedder(BaseEmbedder):
    """Deterministic fast embedder for unit/integration tests without downloading neural models.

    Computes a 64-dimensional normalized vector based on term frequencies and character hash.
    Semantic overlap in meaningful words directly translates to higher cosine similarity.
    """

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def _embed_text(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        clean = (text or "").lower().strip()
        raw_words = [w.strip("?,.:;!\"'()") for w in clean.split() if w.strip("?,.:;!\"'()")]
        if not raw_words:
            vec[0] = 1.0
            return vec

        # Distribute tokens across vector dimensions, downweighting stop words
        for i, word in enumerate(raw_words):
            if word in _STOP_WORDS:
                weight = 0.02
            else:
                weight = 2.0 + (len(word) * 0.1)

            h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            vec[idx] += weight

            # Second harmonic for n-gram density
            if i + 1 < len(raw_words) and raw_words[i+1] not in _STOP_WORDS:
                bigram = f"{word}_{raw_words[i+1]}"
                h2 = int(hashlib.sha256(bigram.encode("utf-8")).hexdigest(), 16)
                idx2 = h2 % self.dim
                vec[idx2] += 0.8

        # Normalize to unit length for cosine similarity
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        else:
            vec[0] = 1.0
        return vec

    def embed_query(self, text: str) -> List[float]:
        return self._embed_text(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_text(t) for t in texts]


def get_default_embedder(prefer_fake: bool = False) -> BaseEmbedder:
    """Return default embedder (FakeEmbedder in tests or if prefer_fake is True)."""
    if prefer_fake:
        return FakeEmbedder()
    try:
        from langchain_huggingface import HuggingFaceEmbeddings

        class LangChainHuggingFaceWrapper(BaseEmbedder):
            def __init__(self):
                self._hf = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

            def embed_query(self, text: str) -> List[float]:
                return self._hf.embed_query(text)

            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                return self._hf.embed_documents(texts)

        return LangChainHuggingFaceWrapper()
    except Exception:
        return FakeEmbedder()
