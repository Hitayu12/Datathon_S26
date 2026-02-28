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
        self.models = [
            "deepseek-r1-distill-llama-70b",
            "llama-3.3-70b-versatile",
            "mixtral-8x7b-32768",
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

    def _chat_json(self, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None

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
            except Exception:
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
        combined = "\n".join([tavily_answer, *tavily_snippets])
        combined_lower = combined.lower()
        failed_keywords = [
            "chapter 11",
            "chapter 7",
            "bankruptcy",
            "bankrupt",
            "liquidation",
            "insolvency",
            "restructuring filing",
            "ceased operations",
        ]
        failed_hit_count = sum(1 for word in failed_keywords if word in combined_lower)

        if not self.enabled:
            hit = failed_hit_count > 0
            return {
                "is_failed": bool(hit),
                "status_label": "failed" if hit else "not_failed",
                "confidence": 0.55,
                "reason": "Heuristic classification used because Groq was unavailable.",
                "evidence": [line for line in tavily_snippets[:3] if line],
                "model_used": "fallback",
            }

        system_prompt = (
            "You are a strict business-status classifier. "
            "Return strict JSON with keys: "
            "is_failed (boolean), status_label (one of failed, not_failed, unclear), "
            "confidence (0 to 1), reason (string <= 35 words), evidence (array of up to 3 short bullet strings)."
        )
        user_prompt = (
            "Determine if this company is a failed/distressed case (bankruptcy, liquidation, major collapse, or insolvency event). "
            f"Company input: {company_input}\n"
            f"Resolved name: {resolved_name}\n"
            f"Ticker: {ticker}\n"
            f"Web evidence:\n{combined}"
        )

        parsed = self._chat_json(system_prompt, user_prompt, temperature=0.0)
        if parsed:
            parsed.setdefault("model_used", "unknown")
            parsed.setdefault("is_failed", False)
            parsed.setdefault("status_label", "unclear")
            parsed.setdefault("confidence", 0.5)
            parsed.setdefault("reason", "No reason returned.")
            parsed.setdefault("evidence", tavily_snippets[:3])

            # Guardrail: when web snippets strongly indicate bankruptcy, avoid false negatives.
            if failed_hit_count >= 2 and not bool(parsed.get("is_failed", False)):
                parsed["is_failed"] = True
                parsed["status_label"] = "failed"
                parsed["confidence"] = max(float(parsed.get("confidence", 0.0)), 0.75)
                parsed["reason"] = (
                    "Keyword evidence strongly indicates a failure/distress event "
                    "(bankruptcy/liquidation language)."
                )
            return parsed

        return {
            "is_failed": False,
            "status_label": "unclear",
            "confidence": 0.5,
            "reason": "Could not verify failure status from model output.",
            "evidence": tavily_snippets[:3],
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
