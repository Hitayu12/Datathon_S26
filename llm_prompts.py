"""Shared prompt templates for structured reasoning clients."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple


VERIFY_FAILURE_STATUS_SYSTEM_PROMPT = (
    "Classify company status. Return JSON only with keys: "
    "is_failed (bool), status_label (failed|not_failed|unclear), "
    "confidence (0-1), reason (<=20 words), evidence (<=3 short bullets)."
)

GENERATE_REASONING_SYSTEM_PROMPT = (
    "You are a distressed-company forensic strategist. "
    "Return strict JSON with keys: "
    "plain_english_explainer (string, easy language), "
    "executive_summary (string), "
    "failure_drivers (array of 3 short bullets), "
    "survivor_differences (array of 3 short bullets), "
    "prevention_measures (array of 3 concrete actions), "
    "technical_notes (array of 3 concise technical bullets)."
)

ANSWER_REPORT_QUESTION_SYSTEM_PROMPT = (
    "You are a senior restructuring analyst answering follow-up questions about a forensic report. "
    "Use both report context and web evidence if available. "
    "Respond in strict JSON with keys: answer, rationale, caveat, confidence. "
    "Keep answer concise and actionable."
)


def compact_text(text: str, max_len: int) -> str:
    compacted = " ".join((text or "").split())
    return compacted[:max_len]


def build_verify_failure_status_inputs(
    *,
    company_input: str,
    resolved_name: str,
    ticker: str,
    tavily_answer: str,
    tavily_snippets: List[str],
) -> Tuple[List[str], str]:
    curated_snippets = [compact_text(s, 120) for s in tavily_snippets if str(s).strip()][:2]
    combined = "\n".join([compact_text(tavily_answer, 160), *curated_snippets])
    return curated_snippets, combined


def build_verify_failure_status_user_prompt(
    *,
    company_input: str,
    resolved_name: str,
    ticker: str,
    combined_evidence: str,
) -> str:
    return (
        "Determine if this company is a failed/distressed case (bankruptcy, liquidation, major collapse, or insolvency event). "
        f"Company input: {company_input}\n"
        f"Resolved name: {resolved_name}\n"
        f"Ticker: {ticker}\n"
        f"Web evidence:\n{combined_evidence}"
    )


def build_generate_reasoning_user_prompt(payload_context: Dict[str, Any]) -> str:
    return (
        "Analyze this failed company against survivor peers and produce clear prevention steps. "
        f"Context JSON:\n{json.dumps(payload_context)}"
    )


def build_answer_report_question_user_prompt(
    *,
    question: str,
    report_context: Dict[str, Any],
    web_evidence: List[Dict[str, str]],
) -> str:
    return (
        f"Question: {question}\n"
        f"Report context JSON:\n{json.dumps(report_context)}\n"
        f"Web evidence JSON:\n{json.dumps(web_evidence)}"
    )
