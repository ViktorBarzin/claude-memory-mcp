"""Detect credentials and secrets in text content."""

import re
from dataclasses import dataclass


@dataclass
class DetectedCredential:
    type: str          # e.g. "password", "api_key", "private_key", "connection_string", "token"
    confidence: float  # 0.0 to 1.0
    start: int         # position in text
    end: int           # position in text
    matched_text: str  # the actual matched text (for redaction)


# Patterns ordered by confidence
_PATTERNS: list[tuple[str, str, float]] = [
    # High confidence (0.9+)
    ("private_key", r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----[\s\S]*?-----END (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----", 0.95),
    ("connection_string", r"(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqp)://[^\s'\"]+", 0.9),
    ("aws_key", r"(?:AKIA|ASIA)[A-Z0-9]{16}", 0.95),
    ("github_token", r"gh[pousr]_[A-Za-z0-9_]{36,}", 0.95),

    # Medium confidence (0.7-0.89)
    ("api_key", r"(?:api[_-]?key|apikey)\s*[:=]\s*['\"]?([A-Za-z0-9_\-]{20,})['\"]?", 0.8),
    ("password", r"(?:password|passwd|pwd)\s*[:=]\s*['\"]?([^\s'\"]{8,})['\"]?", 0.8),
    ("token", r"(?:token|secret|bearer)\s*[:=]\s*['\"]?([A-Za-z0-9_\-\.]{20,})['\"]?", 0.75),
    ("basic_auth", r"(?:Basic\s+)[A-Za-z0-9+/=]{20,}", 0.85),
    ("bearer_token", r"Bearer\s+[A-Za-z0-9_\-\.]{20,}", 0.85),

    # Lower confidence (0.5-0.69)
    ("generic_secret", r"(?:secret|credential|auth)\s*[:=]\s*['\"]?([^\s'\"]{12,})['\"]?", 0.6),
    ("hex_key", r"(?:key|secret)\s*[:=]\s*['\"]?([0-9a-fA-F]{32,})['\"]?", 0.65),
]


def detect_credentials(text: str, min_confidence: float = 0.5) -> list[DetectedCredential]:
    """Scan text for potential credentials and secrets."""
    results: list[DetectedCredential] = []
    for cred_type, pattern, confidence in _PATTERNS:
        if confidence < min_confidence:
            continue
        for match in re.finditer(pattern, text, re.IGNORECASE):
            results.append(DetectedCredential(
                type=cred_type,
                confidence=confidence,
                start=match.start(),
                end=match.end(),
                matched_text=match.group(0),
            ))
    # Deduplicate overlapping matches, keeping highest confidence
    results.sort(key=lambda c: (-c.confidence, c.start))
    filtered: list[DetectedCredential] = []
    for cred in results:
        if not any(c.start <= cred.start and c.end >= cred.end for c in filtered):
            filtered.append(cred)
    return sorted(filtered, key=lambda c: c.start)


def redact_credentials(text: str, credentials: list[DetectedCredential]) -> str:
    """Replace detected credentials with [REDACTED] markers."""
    if not credentials:
        return text
    parts: list[str] = []
    last_end = 0
    for cred in sorted(credentials, key=lambda c: c.start):
        parts.append(text[last_end:cred.start])
        parts.append(f"[REDACTED:{cred.type}]")
        last_end = cred.end
    parts.append(text[last_end:])
    return "".join(parts)


def is_sensitive(text: str, min_confidence: float = 0.7) -> bool:
    """Quick check if text likely contains credentials."""
    return len(detect_credentials(text, min_confidence)) > 0
