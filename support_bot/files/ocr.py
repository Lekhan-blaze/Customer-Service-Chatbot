"""OCR interface: BaseOCR, FakeOCR (for tests), real Gemini Vision fallback."""

import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set

from support_bot.files.models import OcrExtractedFields, OcrResult


# Patterns for field extraction
_ORDER_PATTERNS = [
    re.compile(r"\b(ORD[-\s]?[A-Z0-9]{4,12})\b", re.IGNORECASE),
    re.compile(r"\b(ENR[-\s]?[A-Z0-9]{4,12})\b", re.IGNORECASE),
    re.compile(r"\bOrder\s*(?:ID|#|No\.?)?[:=\s]+([A-Z0-9\-]{4,15})\b", re.IGNORECASE),
    re.compile(r"\bEnrollment\s*(?:ID|#|No\.?)?[:=\s]+([A-Z0-9\-]{4,15})\b", re.IGNORECASE),
]

_AMOUNT_PATTERNS = [
    re.compile(r"(?:Rs\.?|INR|USD|\$|₹)\s*([\d,]+(?:\.\d{1,2})?)", re.IGNORECASE),
    re.compile(r"(?:Total|Amount|Charged|Paid|Due)[\s:=]+(?:Rs\.?|INR|USD|\$|₹)?\s*([\d,]+(?:\.\d{1,2})?)", re.IGNORECASE),
]

_DATE_PATTERNS = [
    re.compile(r"\b(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})\b"),
    re.compile(r"\b(\d{4}-\d{2}-\d{2})\b"),
    re.compile(r"\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})\b", re.IGNORECASE),
]

_CARD_PATTERNS = [
    re.compile(r"\b(?:\d{4}[\s\-]){3}(\d{4})\b"),  # full card number
    re.compile(r"[Xx*]{12,15}(\d{4})\b"),                # masked card
    re.compile(r"(?:card|ending in|last\s*4|last four)[\s:]*(?:digits[:\s]*)?(\d{4})\b", re.IGNORECASE),
]

_INVOICE_PATTERNS = [
    re.compile(r"\b(?:Invoice|Inv)[\s#:\.]+([A-Z0-9\-]{4,20})\b", re.IGNORECASE),
]


def _extract_fields_from_text(text: str) -> OcrExtractedFields:
    """Extract key entities from raw OCR text."""
    order_id = None
    for pat in _ORDER_PATTERNS:
        m = pat.search(text)
        if m:
            order_id = m.group(m.lastindex or 1).strip()
            break

    amount = None
    for pat in _AMOUNT_PATTERNS:
        m = pat.search(text)
        if m:
            amount = m.group(1).strip()
            break

    date = None
    for pat in _DATE_PATTERNS:
        m = pat.search(text)
        if m:
            date = m.group(1).strip()
            break

    card_last4 = None
    for pat in _CARD_PATTERNS:
        m = pat.search(text)
        if m:
            card_last4 = m.group(1).strip()
            break

    invoice_number = None
    m = _INVOICE_PATTERNS[0].search(text)
    if m:
        invoice_number = m.group(1).strip()

    # Estimate confidence: more fields found = higher confidence
    found = sum(1 for v in [order_id, amount, date, card_last4, invoice_number] if v)
    confidence = min(0.5 + found * 0.1, 0.99)

    return OcrExtractedFields(
        order_id=order_id,
        amount=amount,
        date=date,
        card_last4=card_last4,
        invoice_number=invoice_number,
        confidence=confidence,
    )


class BaseOCR(ABC):
    """Abstract OCR interface."""

    @abstractmethod
    def process(self, file_path: str, file_type: str) -> OcrResult:
        """Process a file and return OCR text + extracted fields."""


class FakeOCR(BaseOCR):
    """Deterministic test OCR — reads embedded metadata from filename or dict fixtures."""

    def __init__(
        self,
        fixtures: Optional[Dict[str, str]] = None,
        confidence_overrides: Optional[Dict[str, float]] = None,
        duration_seconds: float = 0.0,
        fail_files: Optional[List[str]] = None,
    ) -> None:
        """fixtures: dict of {filename_stem: raw_text_to_return}
        confidence_overrides: dict of {filename_stem: confidence_float} to force-set confidence.
        duration_seconds: simulated processing duration.
        fail_files: list of filename stems or paths that should fail OCR processing.
        """
        self.fixtures = fixtures or {}
        self.confidence_overrides = confidence_overrides or {}
        self.duration_seconds = duration_seconds
        self.fail_files = set(fail_files or [])

    def process(self, file_path: str, file_type: str) -> OcrResult:
        import re
        from pathlib import Path
        stem = Path(file_path).stem
        clean_stem = re.sub(r"^[a-f0-9]{8}_", "", stem)
        if (
            stem in self.fail_files
            or clean_stem in self.fail_files
            or file_path in self.fail_files
            or Path(file_path).name in self.fail_files
        ):
            return OcrResult(
                text="",
                confidence=0.0,
                extracted_fields=OcrExtractedFields(),
                engine_used="failed",
            )
        # Check explicit fixture
        raw_text = (
            self.fixtures.get(stem)
            or self.fixtures.get(clean_stem)
            or self.fixtures.get(file_path, "")
            or self.fixtures.get(Path(file_path).name, "")
        )

        # If actual .txt file (for test convenience), read its content
        if not raw_text:
            txt_path = Path(str(file_path).replace(f".{file_type}", ".txt"))
            if txt_path.exists():
                raw_text = txt_path.read_text()

        if not raw_text:
            raw_text = f"[FakeOCR: no fixture for {stem}]"

        fields = _extract_fields_from_text(raw_text)
        # Allow test override of confidence (e.g. to simulate blurry/low-confidence scan)
        forced_conf = self.confidence_overrides.get(stem) if stem in self.confidence_overrides else self.confidence_overrides.get(clean_stem)
        confidence = forced_conf if forced_conf is not None else (fields.confidence if raw_text else 0.0)
        fields.confidence = confidence
        return OcrResult(
            text=raw_text,
            confidence=confidence,
            extracted_fields=fields,
            engine_used="fake",
        )


class GeminiVisionOCR(BaseOCR):
    """Real OCR using Gemini Vision API. Falls back to FakeOCR if no API key."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._api_key = api_key
        self._fake = FakeOCR()

    def process(self, file_path: str, file_type: str) -> OcrResult:
        if not self._api_key:
            return self._fake.process(file_path, file_type)

        try:
            import google.generativeai as genai
            import base64, pathlib
            genai.configure(api_key=self._api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            file_bytes = pathlib.Path(file_path).read_bytes()
            encoded = base64.b64encode(file_bytes).decode("utf-8")
            mime_map = {"pdf": "application/pdf", "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}
            mime = mime_map.get(file_type, "application/octet-stream")
            response = model.generate_content([
                {"inline_data": {"mime_type": mime, "data": encoded}},
                "Extract all text from this document. Include: order IDs, enrollment IDs, invoice numbers, amounts, dates, last 4 digits of payment cards if visible. Respond with the raw text only.",
            ])
            raw_text = response.text.strip()
            fields = _extract_fields_from_text(raw_text)
            return OcrResult(text=raw_text, confidence=0.9, extracted_fields=fields, engine_used="gemini_vision")
        except Exception as e:
            return OcrResult(text="", confidence=0.0, extracted_fields=OcrExtractedFields(), engine_used=f"gemini_error:{e}")
