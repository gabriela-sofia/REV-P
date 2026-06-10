"""Regras de compatibilidade para a auditoria de terminologia publica."""

from __future__ import annotations

import re


PROHIBITED_PATTERNS = {
    "review_terminology": [
        r"\breview_gate\b",
        r"\brequires_reviewer_confirmation\b",
        r"\brevis[aã]o supervisora\b",
        r"\bvisual review update update\b",
    ],
    "automation_terminology": [
        r"\bautonomous automation\b",
    ],
    "ai_model_vendor": [
        r"\bclaude-based\b",
    ],
}

FALSE_POSITIVE_PATTERNS = [
    r"\bv1ia\b",
    r"\bv1ic\b",
    r"\bdaily rainfall\b",
    r"\brainfall_daily\b",
    r"\bsource_family\b",
]


def is_false_positive(path: str, text: str) -> bool:
    """Retorna verdadeiro para termos tecnicos preservados pela auditoria."""
    del path
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in FALSE_POSITIVE_PATTERNS)
