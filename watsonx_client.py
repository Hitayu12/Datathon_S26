"""watsonx reasoning client using the OpenAI-compatible chat completions gateway."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

import requests

from ibm_iam import get_iam_token
from llm_prompts import (
    ANSWER_REPORT_QUESTION_SYSTEM_PROMPT,
    GENERATE_REASONING_SYSTEM_PROMPT,
    VERIFY_FAILURE_STATUS_SYSTEM_PROMPT,
    build_answer_report_question_user_prompt,
    build_generate_reasoning_user_prompt,
    build_verify_failure_status_inputs,
    build_verify_failure_status_user_prompt,
)


class WatsonxReasoningClient:
    """Calls watsonx chat completions and requires strict JSON responses."""

    _JSON_REPAIR_PROMPT = "Return ONLY valid JSON matching the schema. No commentary."

    def __init__(
        self,
        api_key: Optional[str] = None,
        project_id: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: int = 35,
    ) -> None:
        self.api_key = (api_key or os.getenv("WATSONX_API_KEY", "")).strip()
        self.project_id = (project_id or os.getenv("WATSONX_PROJECT_ID", "")).strip()
        self.base_url = (base_url or os.getenv("WATSONX_URL", "")).strip().rstrip("/")
        self.model = (model or os.getenv("WATSONX_MODEL", "")).strip()
        self.timeout_seconds = timeout_seconds
        self.last_error = ""

    def _candidate_endpoints(self) -> List[str]:
        if not self.base_url:
            raise ValueError("WATSONX_URL is required.")

        base = self.base_url.rstrip("/")
        if base.endswith("/ml/v1/chat/completions") or base.endswith("/ml/gateway/v1/chat/completions"):
            return [base]
        if base.endswith("/v1/chat/completions"):
            return [base]

        return [
            f"{base}/ml/v1/chat/completions",
            f"{base}/ml/gateway/v1/chat/completions",
            f"{base}/v1/chat/completions",
        ]

    def _extract_json(self, content: str) -> Dict[str, Any]:
        cleaned = (content or "").strip()
        fenced_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, flags=re.IGNORECASE)
        if fenced_match:
            cleaned = fenced_match.group(1).strip()
        else:
            cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
            cleaned = re.sub(r"```$", "", cleaned).strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\}", cleaned)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"watsonx returned non-JSON content: {cleaned}") from exc
            else:
                raise ValueError(f"watsonx returned non-JSON content: {cleaned}")

        if not isinstance(parsed, dict):
            raise ValueError(f"watsonx returned JSON that was not an object: {cleaned}")
        return parsed

    def _chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float,
        max_tokens: int,
    ) -> Dict[str, Any]:
        if not self.api_key:
            raise ValueError("WATSONX_API_KEY is required.")
        if not self.project_id:
            raise ValueError("WATSONX_PROJECT_ID is required.")
        if not self.model:
            raise ValueError("WATSONX_MODEL is required.")

        self.last_error = ""
        access_token, _expires_at = get_iam_token(self.api_key)
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Watsonx-Project-Id": self.project_id,
            "X-Project-Id": self.project_id,
        }
        body = {
            "model": self.model,
            "project_id": self.project_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        attempt_user_prompt = user_prompt
        attempt_max_tokens = max_tokens
        last_parse_error: Optional[Exception] = None
        endpoint_errors: List[str] = []
        for attempt in range(2):
            body["messages"][1]["content"] = attempt_user_prompt
            body["max_tokens"] = attempt_max_tokens
            for endpoint in self._candidate_endpoints():
                try:
                    response = requests.post(
                        endpoint,
                        headers=headers,
                        json=body,
                        timeout=self.timeout_seconds,
                        allow_redirects=False,
                    )
                except requests.RequestException as exc:
                    endpoint_errors.append(f"{endpoint}: {exc}")
                    continue

                if response.is_redirect or response.is_permanent_redirect:
                    location = response.headers.get("Location", "")
                    endpoint_errors.append(
                        f"{endpoint}: redirected to {location or 'another URL'}; use the raw ml.cloud.ibm.com API host in WATSONX_URL"
                    )
                    continue

                if not response.ok:
                    detail = response.text.strip() or f"HTTP {response.status_code}"
                    endpoint_errors.append(f"{endpoint}: HTTP {response.status_code}: {detail}")
                    continue

                try:
                    payload = response.json()
                except ValueError as exc:
                    raise RuntimeError("watsonx response was not valid JSON.") from exc

                try:
                    content = (
                        payload.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                    )
                except Exception as exc:
                    raise RuntimeError("watsonx response did not contain a chat completion message.") from exc

                try:
                    parsed = self._extract_json(str(content))
                    parsed.setdefault("model_used", self.model)
                    return parsed
                except ValueError as exc:
                    last_parse_error = exc
                    endpoint_errors.append(f"{endpoint}: invalid JSON response")
                    raw_content = str(content).strip()
                    attempt_user_prompt = (
                        f"{user_prompt}\n\n"
                        "Your previous response was invalid or truncated JSON.\n"
                        "Repair it and return ONLY valid JSON matching the schema.\n"
                        "Do not omit any required keys.\n"
                        f"Previous response:\n{raw_content}\n\n"
                        f"{self._JSON_REPAIR_PROMPT}"
                    )
                    attempt_max_tokens = max(max_tokens * 2, 512)
                    break

            if last_parse_error is None and endpoint_errors:
                self.last_error = " | ".join(endpoint_errors[-3:])

        if last_parse_error is not None:
            raise last_parse_error
        if endpoint_errors:
            raise RuntimeError(
                "watsonx chat completion request failed. "
                + " | ".join(endpoint_errors[-3:])
            )
        raise RuntimeError("watsonx returned an unreadable response.")

    def verify_failure_status(
        self,
        *,
        company_input: str,
        resolved_name: str,
        ticker: str,
        tavily_answer: str,
        tavily_snippets: List[str],
    ) -> dict:
        curated_snippets, combined = build_verify_failure_status_inputs(
            company_input=company_input,
            resolved_name=resolved_name,
            ticker=ticker,
            tavily_answer=tavily_answer,
            tavily_snippets=tavily_snippets,
        )
        system_prompt = VERIFY_FAILURE_STATUS_SYSTEM_PROMPT
        user_prompt = build_verify_failure_status_user_prompt(
            company_input=company_input,
            resolved_name=resolved_name,
            ticker=ticker,
            combined_evidence=combined,
        )

        parsed = self._chat_json(
            system_prompt,
            user_prompt,
            temperature=0.0,
            max_tokens=160,
        )
        parsed.setdefault("is_failed", False)
        parsed.setdefault("status_label", "unclear")
        parsed.setdefault("confidence", 0.5)
        parsed.setdefault("reason", "No reason returned.")
        parsed.setdefault("evidence", curated_snippets[:3])
        return parsed

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
    ) -> dict:
        payload_context = {
            "company": company_name,
            "ticker": ticker,
            "industry": industry,
            "failing_risk_score": failing_risk_score,
            "survivor_tickers": survivor_tickers,
            "layer_signals": layer_signals,
            "metric_gaps": metric_gaps,
            "simulation": simulation,
            "recommendations": recommendations,
            "tavily_notes": tavily_notes[:5],
        }
        system_prompt = GENERATE_REASONING_SYSTEM_PROMPT
        user_prompt = build_generate_reasoning_user_prompt(payload_context)

        parsed = self._chat_json(
            system_prompt,
            user_prompt,
            temperature=0.15,
            max_tokens=700,
        )
        parsed.setdefault("plain_english_explainer", "No plain-English explanation returned.")
        parsed.setdefault("executive_summary", "No summary returned.")
        parsed.setdefault("failure_drivers", [])
        parsed.setdefault("survivor_differences", [])
        parsed.setdefault("prevention_measures", recommendations[:3])
        parsed.setdefault("technical_notes", [])
        return parsed

    def answer_report_question(
        self,
        *,
        question: str,
        report_context: Dict[str, Any],
        web_evidence: Optional[List[Dict[str, str]]] = None,
    ) -> dict:
        system_prompt = ANSWER_REPORT_QUESTION_SYSTEM_PROMPT
        user_prompt = build_answer_report_question_user_prompt(
            question=question,
            report_context=report_context,
            web_evidence=web_evidence or [],
        )

        parsed = self._chat_json(
            system_prompt,
            user_prompt,
            temperature=0.2,
            max_tokens=300,
        )
        parsed.setdefault("answer", "No answer returned.")
        parsed.setdefault("rationale", "No rationale returned.")
        parsed.setdefault("caveat", "")
        parsed.setdefault("confidence", "")
        return {
            "answer": str(parsed.get("answer", "No answer returned.")),
            "rationale": str(parsed.get("rationale", "No rationale returned.")),
            "caveat": str(parsed.get("caveat", "")),
            "confidence": str(parsed.get("confidence", "")),
        }

    def answer_question(self, report_bundle: dict, question: str) -> dict:
        """Backward-compatible wrapper for the earlier watsonx client signature."""
        return self.answer_report_question(
            question=question,
            report_context=report_bundle,
            web_evidence=[],
        )

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
            "counterfactual_impact (object with before_score, after_score, improvement_pct), disagreements (array), "
            "final_recommendations (array of objects with action, expected_effect, confidence), overall_confidence (0-1)."
        )
        user_prompt = (
            "Create the initial draft for the council. "
            f"Context JSON:\n{json.dumps({'company_profile': company_profile, 'metrics': metrics, 'peer_summary': peer_summary, 'evidence_bundle': evidence_bundle})}"
        )
        return self._chat_json(system_prompt, user_prompt, temperature=0.1, max_tokens=650)

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
            "You are Stage 2 reviewer in a Collaborative Reasoning Council. "
            "Return JSON only with keys: supported_claims, unsupported_claims, missing_factors, rewrite_suggestions. "
            "Use only evidence snippet IDs provided in evidence_bundle.snippets. No hallucinated citations."
        )
        user_prompt = (
            "Review the Groq draft against the same metrics and evidence. "
            f"Context JSON:\n{json.dumps({'company_profile': company_profile, 'metrics': metrics, 'peer_summary': peer_summary, 'evidence_bundle': evidence_bundle, 'groq_draft': groq_draft})}"
        )
        return self._chat_json(system_prompt, user_prompt, temperature=0.0, max_tokens=420)

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
            "You are Stage 4 final synthesis in a Collaborative Reasoning Council. "
            "Return JSON only. Use only evidence snippet IDs from evidence_bundle.snippets. "
            "No hallucinated citations. Remove or downgrade unsupported claims. Prioritize claims agreed by at least two sources. "
            "If evidence is missing, explicitly say 'Evidence unavailable' and reduce confidence. "
            "Schema keys: executive_summary, failure_drivers, survivor_strategies, counterfactual_impact, disagreements, final_recommendations, overall_confidence."
        )
        user_prompt = (
            "Merge the draft, critique, and local quantitative sanity check into one consensus output. "
            f"Context JSON:\n{json.dumps({'company_profile': company_profile, 'metrics': metrics, 'peer_summary': peer_summary, 'evidence_bundle': evidence_bundle, 'groq_draft': groq_draft, 'watsonx_critique': watsonx_critique, 'local_sanity_check': local_sanity_check})}"
        )
        return self._chat_json(system_prompt, user_prompt, temperature=0.1, max_tokens=900)
