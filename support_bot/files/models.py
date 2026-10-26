"""Data models for Module 5: Multimodal File Analysis & OCR Evidence."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class OcrExtractedFields:
    """Key entities extracted from a document via OCR."""
    order_id: Optional[str] = None
    enrollment_id: Optional[str] = None
    amount: Optional[str] = None
    date: Optional[str] = None
    card_last4: Optional[str] = None
    customer_name: Optional[str] = None
    invoice_number: Optional[str] = None
    merchant_name: Optional[str] = None
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class OcrResult:
    """Full result from OCR processing."""
    text: str
    confidence: float
    extracted_fields: OcrExtractedFields
    engine_used: str = "fake"


@dataclass
class ConflictReport:
    """Describes a mismatch between OCR-extracted data and user-stated data."""
    field: str
    user_stated: str
    ocr_extracted: str
    severity: str  # "low" | "medium" | "high"
    clarification_prompt: str


@dataclass
class FileUploadRecord:
    """Represents a processed file upload with OCR analysis."""
    id: str
    session_id: Optional[str]
    ticket_id: Optional[str]
    original_filename: str
    safe_filename: str
    file_type: str
    file_size_bytes: int
    storage_path: str
    scan_status: str = "pending"
    ocr_status: str = "pending"
    ocr_text: Optional[str] = None
    ocr_confidence: Optional[float] = None
    ocr_extracted_fields: Optional[Dict[str, Any]] = None
    extraction_conflicts: List[str] = field(default_factory=list)
    injection_risk_score: float = 0.0
    injection_reasons: List[str] = field(default_factory=list)
    expires_at: Optional[str] = None
    purged: bool = False
    purged_at: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""
