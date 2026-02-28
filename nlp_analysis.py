"""NLP forensics over filings/news snippets with negation-aware theme extraction."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
    "we",
    "our",
    "this",
    "their",
    "or",
    "they",
    "which",
    "have",
    "had",
    "not",
    "can",
    "may",
    "also",
    "than",
    "into",
    "during",
    "about",
    "over",
    "under",
    "after",
    "before",
}

THEME_PATTERNS: Dict[str, List[Tuple[str, float]]] = {
    "liquidity_concerns": [
        (r"\bliquidity\b", 1.2),
        (r"\bgoing concern\b", 1.8),
        (r"\bcash shortfall\b", 1.5),
        (r"\bworking capital\b", 1.1),
        (r"\bcurrent liabilities\b", 1.0),
        (r"\btight cash\b", 1.4),
        (r"\blimited cash runway\b", 1.7),
    ],
    "debt_stress": [
        (r"\bcovenant(?: breach)?\b", 1.8),
        (r"\bdebt maturit(?:y|ies)\b", 1.3),
        (r"\brefinanc(?:e|ing)\b", 1.2),
        (r"\bhigh leverage\b", 1.3),
        (r"\binterest burden\b", 1.3),
        (r"\bdefault(?: risk)?\b", 1.7),
        (r"\bdebt restructuring\b", 1.6),
    ],
    "demand_decline": [
        (r"\bdeclin(?:ing|ed)\s+demand\b", 1.6),
        (r"\bsoft demand\b", 1.4),
        (r"\breduced orders?\b", 1.3),
        (r"\bvolume decline\b", 1.3),
        (r"\bcustomer slowdown\b", 1.3),
        (r"\bweaker demand\b", 1.2),
    ],
    "margin_pressure": [
        (r"\bmargin compression\b", 1.6),
        (r"\bcost inflation\b", 1.4),
        (r"\bpricing pressure\b", 1.3),
        (r"\bgross margin decline\b", 1.5),
        (r"\boperating margin contraction\b", 1.5),
    ],
    "legal_regulatory": [
        (r"\blitigation\b", 1.0),
        (r"\blawsuit\b", 1.0),
        (r"\bregulator(?:y)?\b", 1.0),
        (r"\bcompliance risk\b", 1.1),
        (r"\binvestigation\b", 1.2),
        (r"\benforcement action\b", 1.4),
    ],
    "bankruptcy_language": [
        (r"\bsubstantial doubt\b", 2.1),
        (r"\binsolven(?:cy|t)\b", 2.0),
        (r"\bchapter\s+(?:11|7)\b", 2.4),
        (r"\bbankrupt(?:cy)?\b", 2.2),
        (r"\bliquidation\b", 2.0),
        (r"\brestructur(?:e|ing)\b", 1.5),
        (r"\bdistress(?:ed)?\b", 1.2),
        (r"\bceased operations\b", 2.0),
    ],
}

THEME_IMPORTANCE = {
    "liquidity_concerns": 1.25,
    "debt_stress": 1.20,
    "demand_decline": 1.0,
    "margin_pressure": 0.95,
    "legal_regulatory": 0.80,
    "bankruptcy_language": 1.45,
}

SEVERITY_TERMS = {
    "substantial",
    "severe",
    "material",
    "acute",
    "critical",
    "significant",
    "default",
    "covenant breach",
    "bankruptcy",
    "insolvency",
    "liquidation",
}

UNCERTAINTY_TERMS = {
    "may",
    "might",
    "could",
    "potential",
    "possible",
    "risk of",
}

NEGATION_PATTERNS = [
    r"\bno\b",
    r"\bnot\b",
    r"\bnever\b",
    r"\bwithout\b",
    r"\bdid not\b",
    r"\bno evidence(?: of)?\b",
    r"\bnot filed\b",
    r"\bremains operational\b",
    r"\bstill operating\b",
    r"\bcontinues to operate\b",
]


@dataclass
class ThemeHit:
    phrase: str
    sentence: str
    weight: float
    severity_multiplier: float
    negated: bool


def clean_text(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text or "")
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _split_sentences(text: str) -> List[str]:
    normalized = _normalize_text(text)
    if not normalized:
        return []
    parts = re.split(r"(?<=[.!?])\s+|\s*\n+\s*", normalized)
    return [p.strip() for p in parts if p and p.strip()]


def _severity_multiplier(sentence: str) -> float:
    low = sentence.lower()
    severe_hits = sum(1 for t in SEVERITY_TERMS if t in low)
    uncertain_hits = sum(1 for t in UNCERTAINTY_TERMS if t in low)
    return max(0.65, min(1.8, 1.0 + 0.18 * severe_hits - 0.10 * uncertain_hits))


def _is_negated(sentence: str, start_idx: int) -> bool:
    lower = sentence.lower()
    prefix = lower[max(0, start_idx - 55) : start_idx]
    return any(re.search(pat, prefix) for pat in NEGATION_PATTERNS)


def _dedupe_keep_order(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        key = item.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item.strip())
    return out


def extract_keywords(text: str, top_n: int = 15) -> List[str]:
    cleaned = clean_text(text)
    tokens = [tok for tok in cleaned.split() if len(tok) > 2 and tok not in STOP_WORDS]
    counts: Counter[str] = Counter(tokens)

    bigrams = [
        f"{tokens[i]} {tokens[i + 1]}"
        for i in range(len(tokens) - 1)
        if tokens[i] not in STOP_WORDS and tokens[i + 1] not in STOP_WORDS
    ]
    counts.update({bg: 1 for bg in bigrams})
    return [word for word, _ in counts.most_common(top_n)]


def combine_text_sources(mdna_text: str, snippets: Iterable[str]) -> str:
    parts = [mdna_text.strip()] if mdna_text.strip() else []
    parts.extend([s.strip() for s in snippets if s and s.strip()])
    return "\n".join(_dedupe_keep_order(parts))


def _extract_theme_hits(sentences: List[str]) -> Dict[str, List[ThemeHit]]:
    hits: Dict[str, List[ThemeHit]] = {theme: [] for theme in THEME_PATTERNS}
    for sentence in sentences:
        low = sentence.lower()
        sev = _severity_multiplier(sentence)
        for theme, patterns in THEME_PATTERNS.items():
            local_seen = set()
            for pattern, weight in patterns:
                for match in list(re.finditer(pattern, low))[:2]:
                    phrase = match.group(0)
                    key = (theme, phrase)
                    if key in local_seen:
                        continue
                    local_seen.add(key)
                    hits[theme].append(
                        ThemeHit(
                            phrase=phrase,
                            sentence=sentence,
                            weight=weight,
                            severity_multiplier=sev,
                            negated=_is_negated(sentence, match.start()),
                        )
                    )
    return hits


def extract_risk_themes(text: str) -> Dict[str, int]:
    sentences = _split_sentences(text)
    theme_hits = _extract_theme_hits(sentences)
    scores: Dict[str, int] = {}
    for theme, hits in theme_hits.items():
        positive = sum(1 for h in hits if not h.negated)
        scores[theme] = positive
    return scores


def _theme_forensics(sentences: List[str]) -> Dict[str, object]:
    theme_hits = _extract_theme_hits(sentences)
    theme_counts: Dict[str, int] = {}
    theme_scores: Dict[str, float] = {}
    theme_evidence: Dict[str, List[str]] = {}
    negated_mentions: Dict[str, int] = {}
    raw_weighted: Dict[str, float] = {}
    severity_hits = 0

    for sentence in sentences:
        low = sentence.lower()
        severity_hits += sum(1 for t in SEVERITY_TERMS if t in low)

    for theme, hits in theme_hits.items():
        pos_hits = [h for h in hits if not h.negated]
        neg_hits = [h for h in hits if h.negated]
        weighted = sum(h.weight * h.severity_multiplier for h in pos_hits)
        positive_sentences = {h.sentence for h in pos_hits}
        negated_sentences = {h.sentence for h in neg_hits}

        theme_counts[theme] = len(positive_sentences)
        negated_mentions[theme] = len(negated_sentences)
        raw_weighted[theme] = round(weighted, 4)
        theme_scores[theme] = round(weighted / (weighted + 2.2), 4) if weighted > 0 else 0.0

        ranked_hits = sorted(pos_hits, key=lambda h: h.weight * h.severity_multiplier, reverse=True)
        theme_evidence[theme] = _dedupe_keep_order([h.sentence for h in ranked_hits[:3]])

    importance_total = sum(THEME_IMPORTANCE.values()) or 1.0
    weighted_theme_pressure = sum(theme_scores[t] * THEME_IMPORTANCE[t] for t in THEME_IMPORTANCE) / importance_total
    severity_ratio = min(1.0, severity_hits / max(1.0, len(sentences) * 1.5))
    positive_total = sum(theme_counts.values())
    negated_total = sum(negated_mentions.values())
    negation_relief = negated_total / max(1.0, float(positive_total + negated_total))

    distress_intensity = max(
        0.0,
        min(
            10.0,
            10.0 * (0.74 * weighted_theme_pressure + 0.24 * severity_ratio - 0.16 * negation_relief),
        ),
    )
    confidence = max(
        0.35,
        min(0.98, 0.40 + 0.08 * min(6, positive_total) + 0.18 * weighted_theme_pressure),
    )

    ranked_themes = sorted(theme_scores.items(), key=lambda kv: kv[1], reverse=True)
    strong_themes = [theme for theme, score in ranked_themes if score > 0.28]

    if strong_themes:
        top_names = ", ".join(str(t).replace("_", " ") for t in strong_themes[:3])
        forensic_summary = (
            f"Primary qualitative stress appears in {top_names}. "
            f"Detected {positive_total} distress mentions with intensity {distress_intensity:.1f}/10."
        )
    else:
        forensic_summary = "No strong qualitative distress language was detected in the available snippets."

    return {
        "themes": theme_counts,
        "theme_scores": theme_scores,
        "theme_evidence": theme_evidence,
        "negated_mentions": negated_mentions,
        "raw_theme_weighted": raw_weighted,
        "distress_intensity": round(distress_intensity, 3),
        "confidence": round(confidence, 3),
        "forensic_summary": forensic_summary,
        "severity_hits": severity_hits,
        "positive_mentions": positive_total,
        "negated_total": negated_total,
    }


def qualitative_summary(mdna_text: str, snippets: Iterable[str]) -> Dict[str, object]:
    combined = combine_text_sources(mdna_text, snippets)
    empty_payload = {
        "keywords": [],
        "themes": {theme: 0 for theme in THEME_PATTERNS},
        "theme_signals": [],
        "theme_scores": {theme: 0.0 for theme in THEME_PATTERNS},
        "theme_evidence": {theme: [] for theme in THEME_PATTERNS},
        "negated_mentions": {theme: 0 for theme in THEME_PATTERNS},
        "raw_theme_weighted": {theme: 0.0 for theme in THEME_PATTERNS},
        "distress_intensity": 0.0,
        "confidence": 0.0,
        "forensic_summary": "No qualitative evidence available.",
        "severity_hits": 0,
        "positive_mentions": 0,
        "negated_total": 0,
        "source_sentence_count": 0,
    }
    if not combined:
        return empty_payload

    sentences = _split_sentences(combined)
    keywords = extract_keywords(combined, top_n=18)
    forensics = _theme_forensics(sentences)
    theme_signals = [theme for theme, count in (forensics.get("themes", {}) or {}).items() if int(count) > 0]

    out = dict(empty_payload)
    out.update(
        {
            "keywords": keywords,
            "themes": forensics["themes"],
            "theme_signals": theme_signals,
            "theme_scores": forensics["theme_scores"],
            "theme_evidence": forensics["theme_evidence"],
            "negated_mentions": forensics["negated_mentions"],
            "raw_theme_weighted": forensics["raw_theme_weighted"],
            "distress_intensity": forensics["distress_intensity"],
            "confidence": forensics["confidence"],
            "forensic_summary": forensics["forensic_summary"],
            "severity_hits": forensics["severity_hits"],
            "positive_mentions": forensics["positive_mentions"],
            "negated_total": forensics["negated_total"],
            "source_sentence_count": len(sentences),
        }
    )
    return out
