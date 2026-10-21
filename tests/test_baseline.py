"""
Smoke tests for the support-chatbot-extended baseline.

These tests require no API key:
  - CSV integrity check (row count, columns, UTF-8 cleanliness).
  - Knowledge-base build/load round-trip with a mocked LLM.
"""
import csv
import os
import shutil
import tempfile

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(ROOT, "dataset", "dataset.csv")

EXPECTED_ROWS = 76
EXPECTED_COLUMNS = {"prompt", "response"}


# ---------------------------------------------------------------------------
# 1. CSV integrity
# ---------------------------------------------------------------------------
class TestCSV:
    def test_row_count(self):
        """Dataset must have exactly 76 Q&A rows (header excluded)."""
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == EXPECTED_ROWS, (
            f"Expected {EXPECTED_ROWS} rows, got {len(rows)}"
        )

    def test_columns(self):
        """Dataset must contain 'prompt' and 'response' columns."""
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            cols = set(reader.fieldnames or [])
        assert EXPECTED_COLUMNS.issubset(cols), (
            f"Missing columns: {EXPECTED_COLUMNS - cols}"
        )

    def test_utf8_clean(self):
        """Dataset must be valid UTF-8 with no cp1252-only bytes."""
        with open(CSV_PATH, "rb") as f:
            raw = f.read()
        # This will raise if any byte sequence is not valid UTF-8
        raw.decode("utf-8")

    def test_no_empty_prompts(self):
        """Every row must have a non-empty prompt."""
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        empty = [i for i, r in enumerate(rows, 1) if not r.get("prompt", "").strip()]
        assert not empty, f"Rows with empty prompt: {empty}"


# ---------------------------------------------------------------------------
# 2. Knowledge-base build/load round-trip (no API key)
# ---------------------------------------------------------------------------
class TestKnowledgeBase:
    def test_vector_db_build_and_load(self, tmp_path, monkeypatch):
        """
        Build a FAISS index from the real CSV using the real embedding model,
        then load it back and verify a similarity search returns results.
        No LLM / API key is needed for this test.
        """
        from langchain_community.document_loaders import CSVLoader
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings

        embed = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        index_path = str(tmp_path / "faiss_test_index")

        # Build
        loader = CSVLoader(
            file_path=CSV_PATH, source_column="prompt", encoding="utf-8"
        )
        docs = loader.load()
        assert len(docs) == EXPECTED_ROWS, (
            f"CSVLoader returned {len(docs)} docs, expected {EXPECTED_ROWS}"
        )

        db = FAISS.from_documents(documents=docs, embedding=embed)
        db.save_local(index_path)

        # Load
        db2 = FAISS.load_local(
            index_path, embed, allow_dangerous_deserialization=True
        )

        # Sanity search
        results = db2.similarity_search("Can I take this course?", k=1)
        assert results, "Similarity search returned no results"
        assert results[0].page_content  # non-empty doc
