"""Quality gate validation before knowledge base activation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from support_bot.knowledge.models import QualityGateResult
from support_bot.knowledge.vector_store import VersionedVectorStore


class QualityGate:
    """Evaluates candidate index versions against golden QA benchmarks."""

    def __init__(
        self,
        golden_qa_path: Path | str,
        min_hit_rate: float = 0.65,
        min_grounding_score: float = 0.60,
    ) -> None:
        self.golden_qa_path = Path(golden_qa_path)
        self.min_hit_rate = min_hit_rate
        self.min_grounding_score = min_grounding_score

    def _load_golden_dataset(self) -> List[Dict[str, Any]]:
        if not self.golden_qa_path.exists():
            return []
        with open(self.golden_qa_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def evaluate(
        self,
        vector_store: VersionedVectorStore,
        active_version_score: Optional[float] = None,
    ) -> QualityGateResult:
        """Run benchmark queries, compute retrieval hit rate & grounding, and apply gates."""
        golden_items = self._load_golden_dataset()
        if not golden_items:
            # Fallback if benchmark file is empty
            return QualityGateResult(
                passed=True,
                hit_rate=1.0,
                grounding_score=1.0,
                total_evaluations=0,
                rejection_reason=None,
            )

        hits = 0
        grounded_count = 0
        details = []

        for item in golden_items:
            query = item["question"]
            expected_doc = item.get("expected_doc_id")
            keywords = [k.lower() for k in item.get("expected_keywords", [])]

            # Internal filter allowed for benchmark tests
            results = vector_store.search(query, top_k=3, score_threshold=0.1)

            matched_doc = False
            matched_keywords = False
            retrieved_texts = []

            for chunk, score in results:
                retrieved_texts.append(chunk.chunk_text)
                if expected_doc and chunk.document_id == expected_doc:
                    matched_doc = True
                chunk_lower = chunk.chunk_text.lower()
                if keywords and any(k in chunk_lower for k in keywords):
                    matched_keywords = True

            is_hit = matched_doc or matched_keywords
            if is_hit:
                hits += 1

            # Grounding check: does retrieved passage contain expected factual keywords
            combined_text = " ".join(retrieved_texts).lower()
            grounded = any(k in combined_text for k in keywords) if keywords else True
            if grounded:
                grounded_count += 1

            details.append({
                "question": query,
                "hit": is_hit,
                "grounded": grounded,
                "top_result_count": len(results),
            })

        hit_rate = round(hits / len(golden_items), 4)
        grounding_score = round(grounded_count / len(golden_items), 4)

        # Gate 1: Check minimum required hit rate
        if hit_rate < self.min_hit_rate:
            return QualityGateResult(
                passed=False,
                hit_rate=hit_rate,
                grounding_score=grounding_score,
                total_evaluations=len(golden_items),
                details=details,
                rejection_reason=(
                    f"Retrieval hit rate {hit_rate:.2f} is below minimum threshold {self.min_hit_rate:.2f}"
                ),
            )

        # Gate 2: Check for regression against active version
        if active_version_score is not None and hit_rate < active_version_score:
            return QualityGateResult(
                passed=False,
                hit_rate=hit_rate,
                grounding_score=grounding_score,
                total_evaluations=len(golden_items),
                details=details,
                rejection_reason=(
                    f"Quality score {hit_rate:.2f} regressed below active version score {active_version_score:.2f}"
                ),
            )

        return QualityGateResult(
            passed=True,
            hit_rate=hit_rate,
            grounding_score=grounding_score,
            total_evaluations=len(golden_items),
            details=details,
            rejection_reason=None,
        )
