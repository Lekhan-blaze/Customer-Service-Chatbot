"""File upload validation: type allowlist, size limits, stub malware scan."""

import os
import re
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Allowed MIME type extensions
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".pdf"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB

# Stub malware signature bytes (representative only — never use in production)
_STUB_MALWARE_SIGNATURES = [
    b"X5O!P%@AP[4\\PZX54(P^)7CC)7}$EICAR",  # EICAR test string
    b"#!/bin/sh",
    b"MZ\x90\x00",  # Windows PE header
]


def _stub_malware_scan(file_bytes: bytes) -> Tuple[bool, str]:
    """Stub malware scan — returns (is_clean, reason)."""
    for sig in _STUB_MALWARE_SIGNATURES:
        if sig in file_bytes:
            return False, f"Known malware signature detected"
    return True, "clean"


def validate_upload(
    file_path: str,
    file_bytes: Optional[bytes] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, Optional[str]]:
    """Validate an uploaded file for type, size, and malware.
    
    Returns (is_valid, rejection_reason).
    """
    cfg = config or {}
    files_cfg = cfg.get("files", {})
    max_size = files_cfg.get("max_size_bytes", MAX_FILE_SIZE_BYTES)
    allowed_exts = set(files_cfg.get("allowed_extensions", list(ALLOWED_EXTENSIONS)))

    p = Path(file_path)
    ext = p.suffix.lower()

    # 1. Extension check
    if ext not in allowed_exts:
        return False, f"File type '{ext}' not allowed. Allowed: {sorted(allowed_exts)}"

    # 2. Size check
    if file_bytes is None and p.exists():
        file_bytes = p.read_bytes()
    if file_bytes is not None and len(file_bytes) > max_size:
        return False, f"File size {len(file_bytes):,} bytes exceeds limit of {max_size:,} bytes"

    # 3. Stub malware scan
    if file_bytes is not None:
        is_clean, reason = _stub_malware_scan(file_bytes)
        if not is_clean:
            return False, f"Malware scan failed: {reason}"

    return True, None


def safe_filename(original_name: str) -> str:
    """Sanitize filename: strip path traversal, alphanumeric only, keep extension."""
    p = Path(original_name)
    stem = re.sub(r"[^\w\-]", "_", p.stem)[:64]
    ext = p.suffix.lower()[:5]
    return f"{stem}{ext}"
