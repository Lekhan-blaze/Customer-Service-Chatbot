"""Module 5: Multimodal File Analysis & OCR Evidence."""

from support_bot.files.models import FileUploadRecord, OcrExtractedFields, OcrResult, ConflictReport
from support_bot.files.service import FileService
from support_bot.files.ocr import BaseOCR, FakeOCR, GeminiVisionOCR
from support_bot.files.validator import validate_upload, safe_filename

__all__ = [
    "FileUploadRecord", "OcrExtractedFields", "OcrResult", "ConflictReport",
    "FileService", "BaseOCR", "FakeOCR", "GeminiVisionOCR",
    "validate_upload", "safe_filename",
]
