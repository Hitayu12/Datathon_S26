"""Shared schemas and normalization helpers for collaborative reasoning."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _clean_evidence_ids(values: Any) -> List[int]:
    if not isinstance(values, list):
        return []
    cleaned: List[int] = []
    for value in values:
        try:
            idx = int(value)
        except (TypeError, ValueError):
            continue
        if idx > 0 and idx not in cleaned:
            cleaned.append(idx)
    return cleaned


@dataclass
class EvidenceClaim:
    driver: str
    evidence_ids: List[int] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class StrategyClaim:
    strategy: str
    evidence_ids: List[int] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class Recommendation:
    action: str
    expected_effect: str
    confidence: float = 0.0


@dataclass
class Disagreement:
    topic: str
    groq_view: str
    watsonx_view: str
    local_view: str


@dataclass
class CounterfactualImpact:
    before_score: float = 0.0
    after_score: float = 0.0
    improvement_pct: float = 0.0


@dataclass
class ModelBreakdownEntry:
    raw: Dict[str, Any] = field(default_factory=dict)
    latency_ms: int = 0
    errors: Optional[str] = None


@dataclass
class CouncilOutput:
    executive_summary: str
    failure_drivers: List[EvidenceClaim]
    survivor_strategies: List[StrategyClaim]
    counterfactual_impact: CounterfactualImpact
    disagreements: List[Disagreement]
    final_recommendations: List[Recommendation]
    overall_confidence: float
    model_breakdown: Dict[str, ModelBreakdownEntry]

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["overall_confidence"] = round(_clamp(self.overall_confidence), 3)
        return payload


def validate_council_output_dict(payload: Dict[str, Any]) -> Dict[str, Any]:
    return normalize_council_output(payload).to_dict()


def normalize_council_output(payload: Optional[Dict[str, Any]]) -> CouncilOutput:
    raw = payload or {}
    failure_drivers: List[EvidenceClaim] = []
    for row in raw.get("failure_drivers", []) or []:
        if isinstance(row, dict):
            failure_drivers.append(
                EvidenceClaim(
                    driver=str(row.get("driver", "")).strip() or "Evidence unavailable",
                    evidence_ids=_clean_evidence_ids(row.get("evidence_ids")),
                    confidence=_clamp(_coerce_float(row.get("confidence"), 0.0)),
                )
            )

    survivor_strategies: List[StrategyClaim] = []
    for row in raw.get("survivor_strategies", []) or []:
        if isinstance(row, dict):
            survivor_strategies.append(
                StrategyClaim(
                    strategy=str(row.get("strategy", "")).strip() or "Evidence unavailable",
                    evidence_ids=_clean_evidence_ids(row.get("evidence_ids")),
                    confidence=_clamp(_coerce_float(row.get("confidence"), 0.0)),
                )
            )

    disagreements: List[Disagreement] = []
    for row in raw.get("disagreements", []) or []:
        if isinstance(row, dict):
            disagreements.append(
                Disagreement(
                    topic=str(row.get("topic", "")).strip() or "Open issue",
                    groq_view=str(row.get("groq_view", "")).strip(),
                    watsonx_view=str(row.get("watsonx_view", "")).strip(),
                    local_view=str(row.get("local_view", "")).strip(),
                )
            )

    final_recommendations: List[Recommendation] = []
    for row in raw.get("final_recommendations", []) or []:
        if isinstance(row, dict):
            final_recommendations.append(
                Recommendation(
                    action=str(row.get("action", "")).strip() or "Evidence unavailable",
                    expected_effect=str(row.get("expected_effect", "")).strip() or "Evidence unavailable",
                    confidence=_clamp(_coerce_float(row.get("confidence"), 0.0)),
                )
            )

    counterfactual = raw.get("counterfactual_impact", {}) or {}
    breakdown_raw = raw.get("model_breakdown", {}) or {}
    model_breakdown = {
        key: ModelBreakdownEntry(
            raw=(breakdown_raw.get(key, {}) or {}).get("raw", {}) if isinstance(breakdown_raw.get(key, {}), dict) else {},
            latency_ms=_coerce_int((breakdown_raw.get(key, {}) or {}).get("latency_ms"), 0),
            errors=(breakdown_raw.get(key, {}) or {}).get("errors") if isinstance(breakdown_raw.get(key, {}), dict) else None,
        )
        for key in ["groq", "watsonx", "local"]
    }

    return CouncilOutput(
        executive_summary=str(raw.get("executive_summary", "")).strip() or "Evidence unavailable",
        failure_drivers=failure_drivers[:5],
        survivor_strategies=survivor_strategies[:5],
        counterfactual_impact=CounterfactualImpact(
            before_score=_coerce_float(counterfactual.get("before_score"), 0.0),
            after_score=_coerce_float(counterfactual.get("after_score"), 0.0),
            improvement_pct=_coerce_float(counterfactual.get("improvement_pct"), 0.0),
        ),
        disagreements=disagreements[:6],
        final_recommendations=final_recommendations[:5],
        overall_confidence=_clamp(_coerce_float(raw.get("overall_confidence"), 0.0)),
        model_breakdown=model_breakdown,
    )
