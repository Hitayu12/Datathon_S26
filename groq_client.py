"""Groq client for structured reasoning over failure-vs-survivor analysis."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

import requests

from llm_prompts import (
    ANSWER_REPORT_QUESTION_SYSTEM_PROMPT,
    GENERATE_REASONING_SYSTEM_PROMPT,
    VERIFY_FAILURE_STATUS_SYSTEM_PROMPT,
    build_answer_report_question_user_prompt,
    build_generate_reasoning_user_prompt,
    build_verify_failure_status_inputs,
    build_verify_failure_status_user_prompt,
)

GAP_LABELS = {
    "debt_to_equity_gap": "Debt to equity gap",
    "current_ratio_gap": "Liquidity buffer gap (current ratio)",
    "revenue_growth_gap": "Revenue growth gap",
    "cash_burn_gap": "Cash burn gap",
}


class GroqReasoningClient:
    """Calls Groq chat completions and requests structured JSON output."""

    _JSON_REPAIR_PROMPT = "Return ONLY valid JSON matching the schema. No commentary."

    def __init__(self, api_key: Optional[str], timeout_seconds: int = 35) -> None:
        self.api_key = (api_key or "").strip()
        self.timeout_seconds = timeout_seconds
        self.endpoint = "https://api.groq.com/openai/v1/chat/completions"
        self.last_error = ""
        self.models = [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
        ]

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def _extract_json(self, content: str) -> Optional[Dict[str, Any]]:
        if not content:
            return None

        cleaned = content.strip()
        fenced_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, flags=re.IGNORECASE)
        if fenced_match:
            cleaned = fenced_match.group(1).strip()
        else:
            cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
            cleaned = re.sub(r"```$", "", cleaned).strip()

        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                return None
        return None

    def _chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_completion_tokens: int = 180,
    ) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None

        self.last_error = ""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        for model in self.models:
            attempt_user_prompt = user_prompt
            for attempt in range(2):
                body = {
                    "model": model,
                    "temperature": temperature,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": attempt_user_prompt},
                    ],
                    "response_format": {"type": "json_object"},
                    "max_completion_tokens": max_completion_tokens,
                }
                try:
                    response = requests.post(
                        self.endpoint,
                        headers=headers,
                        json=body,
                        timeout=self.timeout_seconds,
                    )
                    response.raise_for_status()
                    content = (
                        response.json()
                        .get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                    )
                    parsed = self._extract_json(str(content))
                    if parsed:
                        parsed["model_used"] = model
                        return parsed
                    self.last_error = f"{model}: invalid JSON response"
                    attempt_user_prompt = f"{user_prompt}\n\n{self._JSON_REPAIR_PROMPT}"
                except Exception as exc:
                    self.last_error = f"{model}: {type(exc).__name__}"
                    break

        return None

    def verify_failure_status(
        self,
        *,
        company_input: str,
        resolved_name: str,
        ticker: str,
        tavily_answer: str,
        tavily_snippets: List[str],
    ) -> Dict[str, Any]:
        curated_snippets, combined = build_verify_failure_status_inputs(
            company_input=company_input,
            resolved_name=resolved_name,
            ticker=ticker,
            tavily_answer=tavily_answer,
            tavily_snippets=tavily_snippets,
        )
        combined_lower = combined.lower()
        failed_patterns = [
            r"\bfiled for chapter 11\b",
            r"\bfiled chapter 11\b",
            r"\bfiled for bankruptcy\b",
            r"\bbankrupt(?:cy)?\b",
            r"\bentered liquidation\b",
            r"\binsolven(?:cy|t)\b",
            r"\bceased operations\b",
            r"\bshut down\b",
        ]
        negative_patterns = [
            r"\bno bankruptcy\b",
            r"\bno chapter 11\b",
            r"\bdid not file\b",
            r"\bnot bankrupt\b",
            r"\bremains operational\b",
            r"\bstill operating\b",
        ]
        failed_hit_count = sum(1 for pat in failed_patterns if re.search(pat, combined_lower))
        negative_hit_count = sum(1 for pat in negative_patterns if re.search(pat, combined_lower))

        if not self.enabled:
            hit = failed_hit_count > 0
            return {
                "is_failed": bool(hit),
                "status_label": "failed" if hit else "not_failed",
                "confidence": 0.55,
                "reason": "Heuristic classification used because Groq was unavailable.",
                "evidence": [line for line in curated_snippets[:3] if line],
                "model_used": "fallback",
            }

        system_prompt = VERIFY_FAILURE_STATUS_SYSTEM_PROMPT
        user_prompt = build_verify_failure_status_user_prompt(
            company_input=company_input,
            resolved_name=resolved_name,
            ticker=ticker,
            combined_evidence=combined,
        )

        parsed = self._chat_json(system_prompt, user_prompt, temperature=0.0, max_completion_tokens=96)
        if parsed:
            parsed.setdefault("model_used", "unknown")
            parsed.setdefault("is_failed", False)
            parsed.setdefault("status_label", "unclear")
            parsed.setdefault("confidence", 0.5)
            parsed.setdefault("reason", "No reason returned.")
            parsed.setdefault("evidence", curated_snippets[:3])

            # Guardrail: when web snippets strongly indicate bankruptcy, avoid false negatives.
            if failed_hit_count >= 2 and negative_hit_count == 0 and not bool(parsed.get("is_failed", False)):
                parsed["is_failed"] = True
                parsed["status_label"] = "failed"
                parsed["confidence"] = max(float(parsed.get("confidence", 0.0)), 0.75)
                parsed["reason"] = (
                    "Keyword evidence strongly indicates a failure/distress event "
                    "(bankruptcy/liquidation language)."
                )
            return parsed

        fallback_reason = "Could not verify failure status from model output."
        if self.last_error:
            fallback_reason = f"{fallback_reason} Last model error: {self.last_error}."
        return {
            "is_failed": False,
            "status_label": "unclear",
            "confidence": 0.5,
            "reason": fallback_reason,
            "evidence": curated_snippets[:3],
            "model_used": "fallback",
        }

    def _fallback(self, metric_gaps: Dict[str, Any], recommendations: List[str]) -> Dict[str, Any]:
        top_gap_lines: List[str] = []
        for key, value in metric_gaps.items():
            if value is None:
                continue
            label = GAP_LABELS.get(str(key), str(key).replace("_", " "))
            try:
                num = float(value)
                top_gap_lines.append(f"{label}: {num:.2f}")
            except (TypeError, ValueError):
                top_gap_lines.append(f"{label}: {value}")

        return {
            "plain_english_explainer": "The company likely failed because cash pressure and debt stress built up faster than it could recover sales.",
            "executive_summary": "Deterministic fallback used because Groq reasoning was unavailable.",
            "failure_drivers": top_gap_lines[:3] or ["Insufficient data to rank failure drivers."],
            "survivor_differences": ["Survivors showed stronger liquidity/leverage balance in benchmark metrics."],
            "prevention_measures": recommendations[:3] or ["Improve liquidity and reduce leverage."],
            "technical_notes": ["Fallback reasoning path was used."],
            "model_used": "fallback",
        }

    def generate_reasoning(
        self,
        *,
        company_name: str,
        ticker: str,
        industry: str,
        failing_risk_score: float,
        survivor_tickers: List[str],
        layer_signals: Dict[str, List[str]],
        metric_gaps: Dict[str, Any],
        simulation: Dict[str, Any],
        recommendations: List[str],
        tavily_notes: List[str],
    ) -> Dict[str, Any]:
        if not self.enabled:
            return self._fallback(metric_gaps, recommendations)

        payload_context = {
            "company": company_name,
            "ticker": ticker,
            "industry": industry,
            "failing_risk_score": failing_risk_score,
            "survivor_tickers": survivor_tickers,
            "layer_signals": layer_signals,
            "metric_gaps": {
                GAP_LABELS.get(str(k), str(k).replace("_", " ")): v
                for k, v in metric_gaps.items()
            },
            "simulation": simulation,
            "recommendations": recommendations,
            "tavily_notes": tavily_notes[:5],
        }

        system_prompt = GENERATE_REASONING_SYSTEM_PROMPT
        user_prompt = build_generate_reasoning_user_prompt(payload_context)

        parsed = self._chat_json(system_prompt, user_prompt, temperature=0.15)
        if parsed:
            parsed.setdefault("plain_english_explainer", "No plain-English explanation returned.")
            parsed.setdefault("executive_summary", "No summary returned.")
            parsed.setdefault("failure_drivers", [])
            parsed.setdefault("survivor_differences", [])
            parsed.setdefault("prevention_measures", recommendations[:3])
            parsed.setdefault("technical_notes", [])
            return parsed

        return self._fallback(metric_gaps, recommendations)

    def answer_report_question(
        self,
        *,
        question: str,
        report_context: Dict[str, Any],
        web_evidence: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, str]:
        """Answer user follow-up questions about the generated report."""
        def _heuristic_answer() -> Dict[str, str]:
            simple = report_context.get("simple_view", {}) if isinstance(report_context, dict) else {}
            analyst = report_context.get("analyst_view", {}) if isinstance(report_context, dict) else {}
            actions = list(simple.get("prevention_measures", []) or [])
            drivers = list(simple.get("failure_drivers", []) or [])
            risk = simple.get("risk_scores", {}) if isinstance(simple, dict) else {}
            improvement = risk.get("improvement_pct")
            q = question.lower().strip()

            if ("first" in q or "single" in q or "highest" in q) and actions:
                answer = actions[0]
            elif ("why" in q or "driver" in q or "failed" in q) and drivers:
                answer = drivers[0]
            elif "improve" in q and actions:
                answer = actions[0]
            else:
                answer = actions[0] if actions else "Prioritize liquidity and deleveraging in the first 90 days."

            rationale = (
                f"Baseline risk is {risk.get('failing_risk', 'N/A')} and simulated improvement is "
                f"{improvement if improvement is not None else 'N/A'}%, so near-term capital structure and liquidity actions dominate."
            )

            caveat = (
                "This fallback answer is heuristic; model/web-backed confidence improves when Groq responds successfully."
            )
            return {"answer": str(answer), "rationale": rationale, "caveat": caveat, "confidence": "0.62"}

        if not self.enabled:
            return _heuristic_answer()

        system_prompt = ANSWER_REPORT_QUESTION_SYSTEM_PROMPT
        user_prompt = build_answer_report_question_user_prompt(
            question=question,
            report_context=report_context,
            web_evidence=web_evidence or [],
        )

        parsed = self._chat_json(system_prompt, user_prompt, temperature=0.2)
        if not parsed:
            return _heuristic_answer()

        return {
            "answer": str(parsed.get("answer", "No answer returned.")),
            "rationale": str(parsed.get("rationale", "No rationale returned.")),
            "caveat": str(parsed.get("caveat", "")),
            "confidence": str(parsed.get("confidence", "")),
        }

    def generate_council_draft(
        self,
        *,
        company_profile: Dict[str, Any],
        metrics: Dict[str, Any],
        peer_summary: Dict[str, Any],
        evidence_bundle: Dict[str, Any],
    ) -> Dict[str, Any]:
        system_prompt = (
            "You are Stage 1 of a Collaborative Reasoning Council for distressed-company analysis. "
            "Return JSON only. Use only evidence snippet IDs provided in evidence_bundle.snippets. "
            "No hallucinated citations. If support is missing, write 'Evidence unavailable' and lower confidence. "
            "Schema keys: executive_summary (string), failure_drivers (array of objects with driver, evidence_ids, confidence), "
            "survivor_strategies (array of objects with strategy, evidence_ids, confidence), "
            "counterfactual_impact (object with before_score, after_score, improvement_pct), "
            "disagreements (array), final_recommendations (array of objects with action, expected_effect, confidence), "
            "overall_confidence (0-1)."
        )
        user_prompt = (
            "Create the initial draft for the council. "
            f"Context JSON:\n{json.dumps({'company_profile': company_profile, 'metrics': metrics, 'peer_summary': peer_summary, 'evidence_bundle': evidence_bundle})}"
        )
        parsed = self._chat_json(system_prompt, user_prompt, temperature=0.1, max_completion_tokens=520)
        if not parsed:
            raise RuntimeError(self.last_error or "Groq draft generation failed.")
        return parsed

    def generate_council_critique(
        self,
        *,
        company_profile: Dict[str, Any],
        metrics: Dict[str, Any],
        peer_summary: Dict[str, Any],
        evidence_bundle: Dict[str, Any],
        groq_draft: Dict[str, Any],
    ) -> Dict[str, Any]:
        system_prompt = (
            "You are a rigorous reviewer in a Collaborative Reasoning Council. "
            "Return JSON only with keys: supported_claims, unsupported_claims, missing_factors, rewrite_suggestions. "
            "Use only evidence snippet IDs provided in evidence_bundle.snippets. No hallucinated citations."
        )
        user_prompt = (
            "Critique the draft reasoning for evidentiary support, missing factors, and suggested rewrites. "
            f"Context JSON:\n{json.dumps({'company_profile': company_profile, 'metrics': metrics, 'peer_summary': peer_summary, 'evidence_bundle': evidence_bundle, 'groq_draft': groq_draft})}"
        )
        parsed = self._chat_json(system_prompt, user_prompt, temperature=0.0, max_completion_tokens=360)
        if not parsed:
            raise RuntimeError(self.last_error or "Groq critique generation failed.")
        return parsed

    def synthesize_council_output(
        self,
        *,
        company_profile: Dict[str, Any],
        metrics: Dict[str, Any],
        peer_summary: Dict[str, Any],
        evidence_bundle: Dict[str, Any],
        groq_draft: Dict[str, Any],
        watsonx_critique: Dict[str, Any],
        local_sanity_check: Dict[str, Any],
    ) -> Dict[str, Any]:
        system_prompt = (
            "You are the final synthesis step of a Collaborative Reasoning Council. "
            "Return JSON only. Use only evidence snippet IDs from evidence_bundle.snippets. "
            "No hallucinated citations. Remove or downgrade unsupported claims. Prioritize claims agreed by at least two sources. "
            "Schema keys: executive_summary, failure_drivers, survivor_strategies, counterfactual_impact, disagreements, final_recommendations, overall_confidence."
        )
        user_prompt = (
            "Merge the draft, critique, and local quantitative sanity check into one consensus output. "
            f"Context JSON:\n{json.dumps({'company_profile': company_profile, 'metrics': metrics, 'peer_summary': peer_summary, 'evidence_bundle': evidence_bundle, 'groq_draft': groq_draft, 'watsonx_critique': watsonx_critique, 'local_sanity_check': local_sanity_check})}"
        )
        parsed = self._chat_json(system_prompt, user_prompt, temperature=0.1, max_completion_tokens=700)
        if not parsed:
            raise RuntimeError(self.last_error or "Groq council synthesis failed.")
        return parsed
