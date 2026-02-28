"""Groq client for structured reasoning over failure-vs-survivor analysis."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

import requests


class GroqReasoningClient:
    """Calls Groq chat completions and requests structured JSON output."""

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
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
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
            body = {
                "model": model,
                "temperature": temperature,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
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
            except Exception as exc:
                self.last_error = f"{model}: {type(exc).__name__}"
                continue

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
        def _compact(text: str, max_len: int = 120) -> str:
            t = " ".join((text or "").split())
            return t[:max_len]

        curated_snippets = [_compact(s, 120) for s in tavily_snippets if str(s).strip()][:2]
        combined = "\n".join([_compact(tavily_answer, 160), *curated_snippets])
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

        system_prompt = (
            "Classify company status. Return JSON only with keys: "
            "is_failed (bool), status_label (failed|not_failed|unclear), "
            "confidence (0-1), reason (<=20 words), evidence (<=3 short bullets)."
        )
        user_prompt = (
            "Determine if this company is a failed/distressed case (bankruptcy, liquidation, major collapse, or insolvency event). "
            f"Company input: {company_input}\n"
            f"Resolved name: {resolved_name}\n"
            f"Ticker: {ticker}\n"
            f"Web evidence:\n{combined}"
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
            top_gap_lines.append(f"{key}: {value}")

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
            "metric_gaps": metric_gaps,
            "simulation": simulation,
            "recommendations": recommendations,
            "tavily_notes": tavily_notes[:5],
        }

        system_prompt = (
            "You are a distressed-company forensic strategist. "
            "Return strict JSON with keys: "
            "plain_english_explainer (string, easy language), "
            "executive_summary (string), "
            "failure_drivers (array of 3 short bullets), "
            "survivor_differences (array of 3 short bullets), "
            "prevention_measures (array of 3 concrete actions), "
            "technical_notes (array of 3 concise technical bullets)."
        )
        user_prompt = (
            "Analyze this failed company against survivor peers and produce clear prevention steps. "
            f"Context JSON:\n{json.dumps(payload_context)}"
        )

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

        system_prompt = (
            "You are a senior restructuring analyst answering follow-up questions about a forensic report. "
            "Use both report context and web evidence if available. "
            "Respond in strict JSON with keys: answer, rationale, caveat, confidence. "
            "Keep answer concise and actionable."
        )
        user_prompt = (
            f"Question: {question}\n"
            f"Report context JSON:\n{json.dumps(report_context)}\n"
            f"Web evidence JSON:\n{json.dumps(web_evidence or [])}"
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
