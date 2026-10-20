"""Small, deterministic redaction for obvious credentials in recorded summaries."""

from __future__ import annotations

import re
from typing import Any

_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(password|passwd|secret|api[_-]?key|access[_-]?token|refresh[_-]?token|authorization)"
    r"(\s*[:=]\s*|\s+)(['\"]?)([^\s,;\}\"']+)"
)
_BEARER_TOKEN = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]+=*")
_URL_CREDENTIALS = re.compile(r"(://[^/\s:@]+:)[^@/\s]+@")
_SENSITIVE_KEY = re.compile(r"(?i)(password|passwd|secret|api[_-]?key|access[_-]?token|refresh[_-]?token|authorization)")


def redact_text(value: str) -> str:
    """Redact common key/value, bearer-token, and URL credential forms."""
    value = _SECRET_ASSIGNMENT.sub(r"\1\2[REDACTED]", value)
    value = _BEARER_TOKEN.sub("Bearer [REDACTED]", value)
    return _URL_CREDENTIALS.sub(r"\1[REDACTED]@", value)


def redact_value(value: Any) -> Any:
    """Recursively sanitize context without changing its JSON-compatible shape."""
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if _SENSITIVE_KEY.search(str(key)) else redact_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_value(item) for item in value]
    if isinstance(value, str):
        return redact_text(value)
    return value