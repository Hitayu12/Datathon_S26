"""Tavily API client for web intelligence used by the analysis engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


@dataclass
class TavilySearchResult:
    query: str
    answer: str
    snippets: List[str]
    sources: List[str]


class TavilyClient:
    """Minimal Tavily REST client with safe fallbacks."""

    def __init__(self, api_key: Optional[str], timeout_seconds: int = 15) -> None:
        self.api_key = (api_key or "").strip()
        self.timeout_seconds = timeout_seconds
        self.search_endpoint = "https://api.tavily.com/search"

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def search(
        self,
        query: str,
        *,
        max_results: int = 5,
        search_depth: str = "advanced",
        include_answer: bool = True,
    ) -> TavilySearchResult:
        if not self.enabled:
            return TavilySearchResult(query=query, answer="", snippets=[], sources=[])

        payload: Dict[str, Any] = {
            "api_key": self.api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": search_depth,
            "include_answer": include_answer,
        }

        try:
            response = requests.post(
                self.search_endpoint,
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            body = response.json()
        except Exception:
            return TavilySearchResult(query=query, answer="", snippets=[], sources=[])

        answer = str(body.get("answer", "") or "")
        snippets: List[str] = []
        sources: List[str] = []
        for row in body.get("results", []) or []:
            content = str(row.get("content", "") or "").strip()
            url = str(row.get("url", "") or "").strip()
            if content:
                snippets.append(content)
            if url:
                sources.append(url)

        return TavilySearchResult(query=query, answer=answer, snippets=snippets, sources=sources)
