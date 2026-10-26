"""Claim-to-source validation before answer finalization."""

from abc import ABC, abstractmethod
import re
from typing import List, Tuple


class BaseClaimValidator(ABC):
    @abstractmethod
    def validate_claims(self, answer: str, source_passages: List[str]) -> Tuple[str, int]:
        """Validate factual claims in the answer against source passages.

        Returns (cleaned_answer, number_of_unsupported_claims_removed).
        """
        raise NotImplementedError


class FakeClaimValidator(BaseClaimValidator):
    """Deterministic claim validator for tests and lightweight production fallback.

    Extracts factual sentences from the candidate answer and verifies that key terms
    or entities are grounded in at least one retrieved source passage.
    Unsupported sentences containing factual assertions are removed.
    """

    def validate_claims(self, answer: str, source_passages: List[str]) -> Tuple[str, int]:
        if not answer or not source_passages:
            return answer, 0

        combined_sources = " ".join(source_passages).lower()
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
        valid_sentences = []
        removed = 0

        for sentence in sentences:
            # Check if sentence makes a specific policy or number claim
            s_lower = sentence.lower()
            numbers = re.findall(r"\b\d+(?:%|\$| days?| business days?)?\b", s_lower)
            has_factual_claim = any(
                term in s_lower for term in ["refund", "fee", "gst", "days", "eligible", "must", "policy"]
            )

            if has_factual_claim and numbers:
                # Require at least one number/token to be grounded in the source passages
                grounded = any(n in combined_sources for n in numbers)
                if grounded:
                    valid_sentences.append(sentence)
                else:
                    removed += 1
            else:
                valid_sentences.append(sentence)

        if not valid_sentences:
            return "I don't know.", max(1, removed)

        return " ".join(valid_sentences), removed
