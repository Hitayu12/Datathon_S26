"""Collaborative reasoning council orchestration."""

from __future__ import annotations

import ast
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from local_reasoner import LocalAnalystModel
from schemas import normalize_council_output
from watsonx_client import WatsonxReasoningClient

_CACHE_LOCK = threading.Lock()
_COUNCIL_CACHE: Dict[str, Dict[str, Any]] = {}
_COUNCIL_SCHEMA_VERSION = "v2"


def _cache_key(inputs: Dict[str, Any]) -> str:
    company = inputs.get("company_profile", {}) or {}
    failure_year = inputs.get("failure_year")
    key_payload = {
        "schema_version": _COUNCIL_SCHEMA_VERSION,
        "company": str(company.get("name", "")),
        "ticker": str(company.get("ticker", "")),
        "failure_year": failure_year,
    }
    return json.dumps(key_payload, sort_keys=True)


def _with_timing(fn, *args, **kwargs) -> Tuple[Optional[Dict[str, Any]], int, Optional[str]]:
    started = time.perf_counter()
    try:
        result = fn(*args, **kwargs)
        latency_ms = int((time.perf_counter() - started) * 1000)
        return result, latency_ms, None
    except Exception as exc:
        latency_ms = int((time.perf_counter() - started) * 1000)
        return None, latency_ms, str(exc)


def _normalize_error(provider: str, error: Optional[str]) -> Optional[str]:
    if not error:
        return None
    text = str(error).strip()
    if not text:
        return None
    if provider == "watsonx":
        return WatsonxReasoningClient.summarize_error(text)
    compact = " ".join(text.split())
    return compact if len(compact) <= 260 else f"{compact[:257]}..."


def _is_watsonx_quota_error(error: Optional[str]) -> bool:
    return WatsonxReasoningClient.is_quota_error(str(error or ""))


def _summarize_signal_inputs(evidence_bundle: Dict[str, Any]) -> Dict[str, Any]:
    snippets = list((evidence_bundle or {}).get("snippets", []) or [])
    channels = sorted({str((row or {}).get("label", "")).strip() for row in snippets if str((row or {}).get("label", "")).strip()})
    source_count = sum(1 for row in snippets if str((row or {}).get("source", "")).strip())
    return {
        "snippet_count": len(snippets),
        "source_count": source_count,
        "channels": channels,
    }


def _ensure_evidence_ids(items: List[Any], available_ids: List[int]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for row in items:
        if isinstance(row, str):
            # If the LLM returned a list of strings instead of dicts
            row = {"driver": row, "strategy": row, "evidence_ids": [], "confidence": 0.45}
        elif not isinstance(row, dict):
            continue

        evidence_ids = []
        for value in row.get("evidence_ids", []) or []:
            try:
                idx = int(value)
            except (TypeError, ValueError):
                continue
            if idx in available_ids and idx not in evidence_ids:
                evidence_ids.append(idx)

        new_row = dict(row)
        if not evidence_ids:
            new_row["evidence_ids"] = []
            new_row["confidence"] = min(float(new_row.get("confidence", 0.0) or 0.0), 0.45)
        else:
            new_row["evidence_ids"] = evidence_ids
        output.append(new_row)
    return output


def _agreement_confidence(
    evidence_coverage: float,
    groq_ok: bool,
    watsonx_ok: bool,
    local_ok: bool,
    disagreement_count: int,
) -> float:
    agreement = 0.25
    agreement += 0.2 if groq_ok else 0.0
    agreement += 0.2 if watsonx_ok else 0.0
    agreement += 0.15 if local_ok else 0.0
    agreement += 0.2 * max(0.0, min(1.0, evidence_coverage))
    agreement -= min(0.25, disagreement_count * 0.05)
    return max(0.05, min(0.95, agreement))


def _fallback_draft(inputs: Dict[str, Any]) -> Dict[str, Any]:
    recommendations = list(inputs.get("recommendations", []) or [])
    comparison = inputs.get("peer_summary", {}) or {}
    simulation = inputs.get("simulation", {}) or {}
    metric_gaps = comparison.get("metric_gaps", {}) or {}
    drivers = []
    for key, value in list(metric_gaps.items())[:3]:
        drivers.append(f"{str(key).replace('_', ' ')}: {value}")
    return {
        "plain_english_explainer": "Evidence unavailable. Deterministic fallback used because Groq draft failed.",
        "executive_summary": "Collaborative council fallback draft was generated without Groq.",
        "failure_drivers": drivers[:3] or ["Evidence unavailable"],
        "survivor_differences": ["Survivor benchmark suggests stronger liquidity and leverage discipline."],
        "prevention_measures": recommendations[:3] or ["Evidence unavailable"],
        "technical_notes": [f"Counterfactual improvement: {simulation.get('improvement_percentage', 0)}%"],
    }


def _local_sanity_check(inputs: Dict[str, Any]) -> Dict[str, Any]:
    local_model: LocalAnalystModel = inputs["local_model"]
    metrics = inputs.get("metrics", {}) or {}
    macro_stress_score = float(inputs.get("macro_stress_score", 50.0) or 50.0)
    qualitative_intensity = float(inputs.get("qualitative_intensity", 0.0) or 0.0)
    comparison = inputs.get("peer_summary", {}) or {}
    simulation = inputs.get("simulation", {}) or {}

    before = local_model.predict(metrics, macro_stress_score, qualitative_intensity)
    after = local_model.predict(simulation.get("adjusted_metrics", metrics), macro_stress_score, qualitative_intensity)

    flags: List[str] = []
    if before.risk_probability < 0.45 and float(inputs.get("failing_risk_score", 0.0) or 0.0) > 65:
        flags.append("Narrative may overstate distress relative to local probability.")
    if before.risk_probability > after.risk_probability and simulation.get("improvement_percentage", 0) <= 0:
        flags.append("Counterfactual narrative conflicts with local risk delta.")
    if not flags:
        flags.append("Narrative broadly aligns with metric-derived distress profile.")

    return {
        "failure_probability": round(before.risk_probability, 4),
        "counterfactual_probability": round(after.risk_probability, 4),
        "top_numeric_drivers": before.top_drivers,
        "narrative_alignment_flags": flags,
        "feature_values": before.feature_values,
    }


def _build_consensus_fallback(
    inputs: Dict[str, Any],
    evidence_ids: List[int],
    groq_raw: Optional[Dict[str, Any]],
    groq_error: Optional[str],
    groq_latency_ms: int,
    watsonx_raw: Optional[Dict[str, Any]],
    watsonx_error: Optional[str],
    watsonx_latency_ms: int,
    local_raw: Optional[Dict[str, Any]],
    local_error: Optional[str],
    local_latency_ms: int,
) -> Dict[str, Any]:
    signal_summary = _summarize_signal_inputs(inputs.get("evidence_bundle", {}) or {})

    def _extract_claim_payload(
        row: Any,
        *,
        text_keys: List[str],
    ) -> Tuple[str, Optional[float], List[int]]:
        payload: Dict[str, Any] = {}
        if isinstance(row, dict):
            payload = row
        elif isinstance(row, str):
            raw = row.strip()
            if raw.startswith("{") and raw.endswith("}"):
                try:
                    payload = json.loads(raw)
                except Exception:
                    try:
                        parsed = ast.literal_eval(raw)
                        if isinstance(parsed, dict):
                            payload = parsed
                    except Exception:
                        payload = {}
            if not payload:
                return raw, None, []

        if payload:
            text = ""
            for key in text_keys:
                candidate = str(payload.get(key, "")).strip()
                if candidate:
                    text = candidate
                    break
            if not text:
                text = str(payload).strip()
            conf: Optional[float] = None
            if "confidence" in payload:
                try:
                    conf = float(payload.get("confidence"))
                except (TypeError, ValueError):
                    conf = None
            eids: List[int] = []
            for value in list(payload.get("evidence_ids", []) or []):
                try:
                    idx = int(value)
                except (TypeError, ValueError):
                    continue
                if idx > 0 and idx not in eids:
                    eids.append(idx)
            return text, conf, eids

        return "Evidence unavailable", None, []

    recommendations = list(inputs.get("recommendations", []) or [])
    comparison = inputs.get("peer_summary", {}) or {}
    simulation = inputs.get("simulation", {}) or {}
    strategy_ids = evidence_ids[:2]
    driver_ids = evidence_ids[:2]
    fallback_drivers: List[Dict[str, Any]] = []
    for row in list((groq_raw or {}).get("failure_drivers", []) or [])[:3]:
        text, conf, eids = _extract_claim_payload(row, text_keys=["driver", "strategy", "action"])
        if not text:
            continue
        if conf is None:
            conf = 0.62 if eids else 0.5
        conf = max(0.3, min(0.85, conf))
        fallback_drivers.append(
            {
                "driver": text,
                "evidence_ids": eids if eids else driver_ids,
                "confidence": conf,
            }
        )

    fallback_strategies: List[Dict[str, Any]] = []
    for row in recommendations[:3]:
        text, conf, eids = _extract_claim_payload(row, text_keys=["strategy", "action", "driver"])
        if not text:
            continue
        if conf is None:
            conf = 0.58 if eids else 0.48
        conf = max(0.3, min(0.8, conf))
        fallback_strategies.append(
            {
                "strategy": text,
                "evidence_ids": eids if eids else strategy_ids,
                "confidence": conf,
            }
        )

    fallback = {
        "executive_summary": "Evidence unavailable. Consensus fallback was assembled from the available deterministic signals.",
        "failure_drivers": fallback_drivers,
        "survivor_strategies": fallback_strategies,
        "counterfactual_impact": {
            "before_score": float(inputs.get("failing_risk_score", 0.0) or 0.0),
            "after_score": float(simulation.get("adjusted_score", 0.0) or 0.0),
            "improvement_pct": float(simulation.get("improvement_percentage", 0.0) or 0.0),
        },
        "disagreements": [],
        "final_recommendations": [
            {"action": str(item), "expected_effect": "Reduce distress risk versus current baseline.", "confidence": 0.45}
            for item in recommendations[:3]
        ],
        "overall_confidence": 0.3,
        "model_breakdown": {
            "groq": {"raw": groq_raw or {}, "latency_ms": groq_latency_ms, "errors": groq_error, "signal_summary": signal_summary},
            "watsonx": {"raw": watsonx_raw or {}, "latency_ms": watsonx_latency_ms, "errors": watsonx_error, "signal_summary": signal_summary},
            "local": {"raw": local_raw or {}, "latency_ms": local_latency_ms, "errors": local_error, "signal_summary": signal_summary},
        },
        "signal_summary": signal_summary,
    }

    if not fallback["failure_drivers"]:
        metric_gaps = comparison.get("metric_gaps", {}) or {}
        fallback["failure_drivers"] = [
            {"driver": f"{str(k).replace('_', ' ')}: {v}", "evidence_ids": [], "confidence": 0.25}
            for k, v in list(metric_gaps.items())[:3]
        ]
    if not fallback["survivor_strategies"]:
        fallback["survivor_strategies"] = [
            {"strategy": "Evidence unavailable", "evidence_ids": [], "confidence": 0.2}
        ]
    if not fallback["final_recommendations"]:
        fallback["final_recommendations"] = [
            {"action": "Evidence unavailable", "expected_effect": "Evidence unavailable", "confidence": 0.2}
        ]
    return fallback


def run_reasoning_council(inputs: Dict[str, Any]) -> Dict[str, Any]:
    key = _cache_key(inputs)
    with _CACHE_LOCK:
        if key in _COUNCIL_CACHE:
            return _COUNCIL_CACHE[key]

    evidence_bundle = inputs.get("evidence_bundle", {}) or {}
    evidence_items = list(evidence_bundle.get("snippets", []) or [])
    evidence_ids = [int(item.get("id")) for item in evidence_items if str(item.get("id", "")).isdigit()]
    signal_summary = _summarize_signal_inputs(evidence_bundle)

    groq_client = inputs.get("groq_client")
    watsonx_client = inputs.get("watsonx_client")
    local_model = inputs.get("local_model")

    if local_model is None:
        raise ValueError("run_reasoning_council requires local_model in inputs.")

    draft_inputs = {
        "company_profile": inputs.get("company_profile", {}) or {},
        "metrics": inputs.get("metrics", {}) or {},
        "peer_summary": inputs.get("peer_summary", {}) or {},
        "evidence_bundle": evidence_bundle,
    }

    groq_raw: Optional[Dict[str, Any]] = None
    groq_latency_ms = 0
    groq_error: Optional[str] = None
    if groq_client is not None:
        groq_raw, groq_latency_ms, groq_error = _with_timing(groq_client.generate_council_draft, **draft_inputs)
        groq_error = _normalize_error("groq", groq_error)
    if groq_raw is None:
        groq_raw = _fallback_draft(inputs)

    with ThreadPoolExecutor(max_workers=2) as pool:
        critique_future = None
        if watsonx_client is not None:
            critique_future = pool.submit(
                _with_timing,
                watsonx_client.generate_council_critique,
                company_profile=inputs.get("company_profile", {}) or {},
                metrics=inputs.get("metrics", {}) or {},
                peer_summary=inputs.get("peer_summary", {}) or {},
                evidence_bundle=evidence_bundle,
                groq_draft=groq_raw,
            )
        local_future = pool.submit(_with_timing, _local_sanity_check, dict(inputs, local_model=local_model))

        watsonx_raw: Optional[Dict[str, Any]] = None
        watsonx_latency_ms = 0
        watsonx_error: Optional[str] = None
        if critique_future is not None:
            watsonx_raw, watsonx_latency_ms, watsonx_error = critique_future.result(timeout=60)
            watsonx_error = _normalize_error("watsonx", watsonx_error)

        local_raw, local_latency_ms, local_error = local_future.result(timeout=60)
        local_error = _normalize_error("local", local_error)

    synthesis_provider = inputs.get("synthesis_provider", "watsonx")
    synthesis_client = watsonx_client if synthesis_provider == "watsonx" and watsonx_client is not None else groq_client
    if synthesis_client is None and watsonx_client is not None:
        synthesis_client = watsonx_client
    if synthesis_client is None and groq_client is not None:
        synthesis_client = groq_client

    final_raw: Optional[Dict[str, Any]] = None
    synthesis_error: Optional[str] = None
    if synthesis_client is not None:
        final_raw, synthesis_latency_ms, synthesis_error = _with_timing(
            synthesis_client.synthesize_council_output,
            company_profile=inputs.get("company_profile", {}) or {},
            metrics=inputs.get("metrics", {}) or {},
            peer_summary=inputs.get("peer_summary", {}) or {},
            evidence_bundle=evidence_bundle,
            groq_draft=groq_raw,
            watsonx_critique=watsonx_raw or {},
            local_sanity_check=local_raw or {},
        )
        synthesis_error = _normalize_error("watsonx" if synthesis_provider == "watsonx" else "groq", synthesis_error)
        if synthesis_error:
            if synthesis_provider == "watsonx":
                watsonx_error = synthesis_error
                watsonx_latency_ms = max(watsonx_latency_ms, synthesis_latency_ms)
            else:
                groq_error = synthesis_error
                groq_latency_ms = max(groq_latency_ms, synthesis_latency_ms)

    # If watsonx synthesis is blocked (quota/permission/transient issue), retry synthesis with Groq.
    if (
        final_raw is None
        and synthesis_provider == "watsonx"
        and watsonx_client is not None
        and groq_client is not None
    ):
        retry_raw, retry_latency_ms, retry_error = _with_timing(
            groq_client.synthesize_council_output,
            company_profile=inputs.get("company_profile", {}) or {},
            metrics=inputs.get("metrics", {}) or {},
            peer_summary=inputs.get("peer_summary", {}) or {},
            evidence_bundle=evidence_bundle,
            groq_draft=groq_raw,
            watsonx_critique=watsonx_raw or {},
            local_sanity_check=local_raw or {},
        )
        retry_error = _normalize_error("groq", retry_error)
        groq_latency_ms = max(groq_latency_ms, retry_latency_ms)
        if retry_raw is not None:
            final_raw = retry_raw
            if watsonx_error:
                fallback_note = (
                    "watsonx quota/permission blocked; synthesis automatically failed over to Groq."
                    if _is_watsonx_quota_error(watsonx_error)
                    else "watsonx synthesis failed; synthesis automatically failed over to Groq."
                )
                watsonx_error = f"{watsonx_error} | {fallback_note}"
            else:
                watsonx_error = "watsonx synthesis unavailable; synthesis automatically failed over to Groq."
        elif retry_error:
            groq_error = retry_error if not groq_error else f"{groq_error} | {retry_error}"

    if final_raw is None:
        final_raw = _build_consensus_fallback(
            inputs,
            evidence_ids,
            groq_raw,
            groq_error,
            groq_latency_ms,
            watsonx_raw,
            watsonx_error,
            watsonx_latency_ms,
            local_raw,
            local_error,
            local_latency_ms,
        )

    final_raw["failure_drivers"] = _ensure_evidence_ids(list(final_raw.get("failure_drivers", []) or []), evidence_ids)
    final_raw["survivor_strategies"] = _ensure_evidence_ids(list(final_raw.get("survivor_strategies", []) or []), evidence_ids)

    coverage_pool = final_raw.get("failure_drivers", []) + final_raw.get("survivor_strategies", [])
    cited_items = sum(1 for row in coverage_pool if row.get("evidence_ids"))
    evidence_coverage = cited_items / max(len(coverage_pool), 1)
    final_raw["overall_confidence"] = _agreement_confidence(
        evidence_coverage=evidence_coverage,
        groq_ok=groq_error is None,
        watsonx_ok=watsonx_error is None,
        local_ok=local_error is None,
        disagreement_count=len(list(final_raw.get("disagreements", []) or [])),
    )
    final_raw["model_breakdown"] = {
        "groq": {"raw": groq_raw or {}, "latency_ms": groq_latency_ms, "errors": groq_error, "signal_summary": signal_summary},
        "watsonx": {"raw": watsonx_raw or {}, "latency_ms": watsonx_latency_ms, "errors": watsonx_error, "signal_summary": signal_summary},
        "local": {"raw": local_raw or {}, "latency_ms": local_latency_ms, "errors": local_error, "signal_summary": signal_summary},
    }
    final_raw["signal_summary"] = signal_summary

    normalized = normalize_council_output(final_raw).to_dict()
    with _CACHE_LOCK:
        _COUNCIL_CACHE[key] = normalized
    return normalized
