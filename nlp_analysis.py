"""NLP helpers for MD&A and web snippets."""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, Iterable, List

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", "in", "is",
    "it", "its", "of", "on", "that", "the", "to", "was", "were", "will", "with", "we", "our", "this",
    "their", "or", "they", "which", "have", "had", "not", "can", "may", "also", "than", "into", "during",
}

THEME_KEYWORDS = {
    "liquidity_concerns": [
        "liquidity", "going concern", "cash shortfall", "working capital", "current liabilities", "tight cash",
    ],
    "debt_stress": [
        "covenant", "covenant breach", "debt maturity", "refinancing", "high leverage", "interest burden",
    ],
    "demand_decline": [
        "declining demand", "soft demand", "reduced orders", "volume decline", "customer slowdown",
    ],
    "margin_pressure": [
        "margin compression", "cost inflation", "pricing pressure", "gross margin decline",
    ],
    "legal_regulatory": [
        "litigation", "lawsuit", "regulatory", "compliance risk", "investigation",
    ],
    "bankruptcy_language": [
        "substantial doubt", "restructuring", "insolvency", "chapter 11", "distressed",
    ],
}


def clean_text(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text or "")
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def extract_keywords(text: str, top_n: int = 15) -> List[str]:
    cleaned = clean_text(text)
    tokens = [tok for tok in cleaned.split() if len(tok) > 2 and tok not in STOP_WORDS]
    counts = Counter(tokens)
    return [word for word, _ in counts.most_common(top_n)]


def extract_risk_themes(text: str) -> Dict[str, int]:
    cleaned = clean_text(text)
    scores: Dict[str, int] = {}
    for theme, keywords in THEME_KEYWORDS.items():
        count = sum(1 for phrase in keywords if phrase in cleaned)
        scores[theme] = count
    return scores


def combine_text_sources(mdna_text: str, snippets: Iterable[str]) -> str:
    parts = [mdna_text.strip()] if mdna_text.strip() else []
    parts.extend([s.strip() for s in snippets if s and s.strip()])
    return "\n".join(parts)


def qualitative_summary(mdna_text: str, snippets: Iterable[str]) -> Dict[str, object]:
    combined = combine_text_sources(mdna_text, snippets)
    if not combined:
        return {
            "keywords": [],
            "themes": {theme: 0 for theme in THEME_KEYWORDS},
            "theme_signals": [],
        }

    keywords = extract_keywords(combined, top_n=15)
    themes = extract_risk_themes(combined)
    theme_signals = [theme for theme, score in themes.items() if score > 0]
    return {
        "keywords": keywords,
        "themes": themes,
        "theme_signals": theme_signals,
    }
