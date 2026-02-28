"""Minimal smoke test for the collaborative reasoning council."""

from __future__ import annotations

import json
import os

from collaborative_reasoning import run_reasoning_council
from groq_client import GroqReasoningClient
from local_reasoner import LocalAnalystModel
from watsonx_client import WatsonxReasoningClient


def main() -> None:
    local_model = LocalAnalystModel(random_state=42)
    local_model.train(n_samples=2000)

    groq_client = GroqReasoningClient(os.getenv("GROQ_API_KEY", ""), timeout_seconds=15)
    watsonx_client = None
    if all(
        os.getenv(name, "").strip()
        for name in ["WATSONX_API_KEY", "WATSONX_PROJECT_ID", "WATSONX_URL", "WATSONX_MODEL"]
    ):
        watsonx_client = WatsonxReasoningClient(
            api_key=os.getenv("WATSONX_API_KEY", ""),
            project_id=os.getenv("WATSONX_PROJECT_ID", ""),
            base_url=os.getenv("WATSONX_URL", ""),
            model=os.getenv("WATSONX_MODEL", ""),
            timeout_seconds=15,
        )

    inputs = {
        "company_profile": {
            "name": "Lehman Brothers",
            "ticker": "LEHMQ",
            "industry": "Capital Markets",
            "sector": "Financial Services",
        },
        "metrics": {
            "debt_to_equity": 3.2,
            "current_ratio": 0.78,
            "cash_burn": 240000000.0,
            "revenue_growth": -0.18,
            "revenue": 1200000000.0,
        },
        "peer_summary": {
            "metric_gaps": {
                "debt_to_equity_gap": 2.3,
                "current_ratio_gap": 1.6,
                "revenue_growth_gap": 0.21,
                "cash_burn_gap": 180000000.0,
            },
            "survivor_tickers": ["GS", "MS", "JPM"],
        },
        "simulation": {
            "original_score": 78.2,
            "adjusted_score": 41.7,
            "improvement_percentage": 46.68,
            "adjusted_metrics": {
                "debt_to_equity": 1.1,
                "current_ratio": 1.9,
                "cash_burn": 20000000.0,
                "revenue_growth": 0.03,
                "revenue": 1200000000.0,
            },
        },
        "recommendations": [
            "Reduce leverage toward survivor norms.",
            "Rebuild liquidity reserves before refinancing pressure escalates.",
            "Stabilize demand exposure and cut fixed-cost burn.",
        ],
        "failing_risk_score": 78.2,
        "macro_stress_score": 72.0,
        "qualitative_intensity": 3.0,
        "failure_year": 2008,
        "evidence_bundle": {
            "snippets": [
                {"id": 1, "label": "failure_check", "text": "The firm filed for Chapter 11 protection in 2008.", "source": "example://1"},
                {"id": 2, "label": "macro", "text": "Credit markets froze and refinancing conditions deteriorated sharply.", "source": "example://2"},
                {"id": 3, "label": "strategy", "text": "Peers that survived raised liquidity early and reduced leverage.", "source": "example://3"},
            ]
        },
        "groq_client": groq_client,
        "watsonx_client": watsonx_client,
        "local_model": local_model,
        "synthesis_provider": "watsonx" if watsonx_client is not None else "groq",
    }

    result = run_reasoning_council(inputs)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
