"""Data ingestion for financials, company profile, ticker resolution, and peer discovery."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import requests
import yfinance as yf

SECTOR_GROUPS: Dict[str, Dict[str, str]] = {
    "AAPL": {"sector": "Technology", "industry": "Consumer Electronics"},
    "MSFT": {"sector": "Technology", "industry": "Software"},
    "GOOGL": {"sector": "Technology", "industry": "Internet Content"},
    "AMZN": {"sector": "Consumer Cyclical", "industry": "Internet Retail"},
    "META": {"sector": "Technology", "industry": "Internet Content"},
    "NVDA": {"sector": "Technology", "industry": "Semiconductors"},
    "TSLA": {"sector": "Consumer Cyclical", "industry": "Auto Manufacturers"},
    "NFLX": {"sector": "Communication Services", "industry": "Entertainment"},
    "INTC": {"sector": "Technology", "industry": "Semiconductors"},
    "AMD": {"sector": "Technology", "industry": "Semiconductors"},
    "JPM": {"sector": "Financial Services", "industry": "Banks"},
    "BAC": {"sector": "Financial Services", "industry": "Banks"},
    "C": {"sector": "Financial Services", "industry": "Banks"},
    "WFC": {"sector": "Financial Services", "industry": "Banks"},
    "GS": {"sector": "Financial Services", "industry": "Capital Markets"},
    "MS": {"sector": "Financial Services", "industry": "Capital Markets"},
    "AXP": {"sector": "Financial Services", "industry": "Credit Services"},
    "BLK": {"sector": "Financial Services", "industry": "Asset Management"},
    "SCHW": {"sector": "Financial Services", "industry": "Capital Markets"},
    "USB": {"sector": "Financial Services", "industry": "Banks"},
    "WMT": {"sector": "Consumer Defensive", "industry": "Discount Stores"},
    "TGT": {"sector": "Consumer Defensive", "industry": "Discount Stores"},
    "COST": {"sector": "Consumer Defensive", "industry": "Discount Stores"},
    "HD": {"sector": "Consumer Cyclical", "industry": "Home Improvement"},
    "LOW": {"sector": "Consumer Cyclical", "industry": "Home Improvement"},
    "NKE": {"sector": "Consumer Cyclical", "industry": "Footwear"},
    "SBUX": {"sector": "Consumer Cyclical", "industry": "Restaurants"},
    "MCD": {"sector": "Consumer Cyclical", "industry": "Restaurants"},
    "DIS": {"sector": "Communication Services", "industry": "Entertainment"},
    "CMCSA": {"sector": "Communication Services", "industry": "Telecom Services"},
    "XOM": {"sector": "Energy", "industry": "Oil & Gas"},
    "CVX": {"sector": "Energy", "industry": "Oil & Gas"},
    "COP": {"sector": "Energy", "industry": "Oil & Gas"},
    "SLB": {"sector": "Energy", "industry": "Oil & Gas Equipment"},
    "OXY": {"sector": "Energy", "industry": "Oil & Gas"},
    "EOG": {"sector": "Energy", "industry": "Oil & Gas"},
    "MPC": {"sector": "Energy", "industry": "Oil & Gas Refining"},
    "PSX": {"sector": "Energy", "industry": "Oil & Gas Refining"},
    "VLO": {"sector": "Energy", "industry": "Oil & Gas Refining"},
    "KMI": {"sector": "Energy", "industry": "Midstream"},
    "PFE": {"sector": "Healthcare", "industry": "Drug Manufacturers"},
    "JNJ": {"sector": "Healthcare", "industry": "Drug Manufacturers"},
    "MRK": {"sector": "Healthcare", "industry": "Drug Manufacturers"},
    "ABBV": {"sector": "Healthcare", "industry": "Drug Manufacturers"},
    "UNH": {"sector": "Healthcare", "industry": "Health Plans"},
    "CVS": {"sector": "Healthcare", "industry": "Healthcare Plans"},
    "LLY": {"sector": "Healthcare", "industry": "Drug Manufacturers"},
    "BMY": {"sector": "Healthcare", "industry": "Drug Manufacturers"},
    "GILD": {"sector": "Healthcare", "industry": "Drug Manufacturers"},
    "MDT": {"sector": "Healthcare", "industry": "Medical Devices"},
    "CAT": {"sector": "Industrials", "industry": "Farm & Heavy Machinery"},
    "DE": {"sector": "Industrials", "industry": "Farm & Heavy Machinery"},
    "BA": {"sector": "Industrials", "industry": "Aerospace"},
    "GE": {"sector": "Industrials", "industry": "Conglomerates"},
    "MMM": {"sector": "Industrials", "industry": "Conglomerates"},
    "HON": {"sector": "Industrials", "industry": "Conglomerates"},
    "UPS": {"sector": "Industrials", "industry": "Integrated Freight"},
    "FDX": {"sector": "Industrials", "industry": "Integrated Freight"},
    "LMT": {"sector": "Industrials", "industry": "Aerospace"},
    "RTX": {"sector": "Industrials", "industry": "Aerospace"},
    "BK": {"sector": "Financial Services", "industry": "Asset Management"},
    "STT": {"sector": "Financial Services", "industry": "Asset Management"},
    "NTRS": {"sector": "Financial Services", "industry": "Asset Management"},
    "RJF": {"sector": "Financial Services", "industry": "Capital Markets"},
    "TROW": {"sector": "Financial Services", "industry": "Asset Management"},
    "BEN": {"sector": "Financial Services", "industry": "Asset Management"},
    "ICE": {"sector": "Financial Services", "industry": "Capital Markets"},
    "CME": {"sector": "Financial Services", "industry": "Capital Markets"},
    "LPLA": {"sector": "Financial Services", "industry": "Capital Markets"},
    "PYPL": {"sector": "Financial Services", "industry": "Credit Services"},
    "COF": {"sector": "Financial Services", "industry": "Credit Services"},
    "AIG": {"sector": "Financial Services", "industry": "Insurance"},
    "ALL": {"sector": "Financial Services", "industry": "Insurance"},
    "MET": {"sector": "Financial Services", "industry": "Insurance"},
    "PRU": {"sector": "Financial Services", "industry": "Insurance"},
    "LEHMQ": {"sector": "Financial Services", "industry": "Capital Markets"},
}

DEFAULT_UNIVERSE = list(SECTOR_GROUPS.keys())

PROFILE_HINTS: Dict[str, Tuple[str, str]] = {
    "lehman": ("Financial Services", "Capital Markets"),
    "bear stearns": ("Financial Services", "Capital Markets"),
    "countrywide": ("Financial Services", "Credit Services"),
    "washington mutual": ("Financial Services", "Banks"),
    "svb": ("Financial Services", "Banks"),
    "silicon valley bank": ("Financial Services", "Banks"),
}

INDUSTRY_HINTS: List[Tuple[List[str], Tuple[str, str]]] = [
    (["investment bank", "broker", "brokerage", "securities", "capital markets", "trading"], ("Financial Services", "Capital Markets")),
    (["commercial bank", "retail bank", "bank", "bancorp", "depository"], ("Financial Services", "Banks")),
    (["asset management", "wealth management", "fund manager"], ("Financial Services", "Asset Management")),
    (["credit card", "consumer finance", "lending", "loan servicing"], ("Financial Services", "Credit Services")),
    (["insurance", "underwriting", "reinsurance"], ("Financial Services", "Insurance")),
]


@dataclass
class CompanyProfile:
    ticker: str
    name: str
    sector: str
    industry: str


@dataclass
class ResolvedCompany:
    query: str
    ticker: str
    name: str
    confidence: float
    method: str


@lru_cache(maxsize=256)
def _safe_info(ticker: str) -> Dict[str, object]:
    try:
        return yf.Ticker(ticker).info or {}
    except Exception:
        return {}


def fetch_company_info(ticker: str) -> Dict[str, object]:
    return dict(_safe_info(ticker.upper()))


def _is_ticker_like(text: str) -> bool:
    raw = text.strip()
    return bool(re.fullmatch(r"[A-Z\.\-]{1,8}", raw)) and raw == raw.upper()


def _yahoo_search(query: str, max_results: int = 8) -> List[Dict[str, str]]:
    url = "https://query2.finance.yahoo.com/v1/finance/search"
    try:
        response = requests.get(
            url,
            params={"q": query, "quotesCount": max_results, "newsCount": 0},
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return []

    candidates: List[Dict[str, str]] = []
    for row in payload.get("quotes", []) or []:
        symbol = str(row.get("symbol") or "").strip().upper()
        shortname = str(row.get("shortname") or row.get("longname") or symbol).strip()
        quote_type = str(row.get("quoteType") or "").strip().upper()
        if not symbol or quote_type not in {"EQUITY", "ETF"}:
            continue
        candidates.append({"ticker": symbol, "name": shortname})
    return candidates


def resolve_company_input(company_input: str) -> Optional[ResolvedCompany]:
    query = (company_input or "").strip()
    if not query:
        return None

    if _is_ticker_like(query):
        ticker = query.upper()
        info = _safe_info(ticker)
        name = str(info.get("shortName") or info.get("longName") or ticker)
        return ResolvedCompany(query=query, ticker=ticker, name=name, confidence=0.95, method="ticker")

    candidates = _yahoo_search(query)
    if candidates:
        top = candidates[0]
        return ResolvedCompany(
            query=query,
            ticker=top["ticker"],
            name=top["name"],
            confidence=0.78,
            method="search",
        )

    synthetic = re.sub(r"[^A-Za-z]", "", query).upper()[:8]
    if not synthetic:
        return None
    return ResolvedCompany(
        query=query,
        ticker=synthetic,
        name=query,
        confidence=0.35,
        method="fallback_name",
    )


def fetch_company_profile(ticker: str) -> CompanyProfile:
    symbol = ticker.upper()
    info = _safe_info(symbol)
    static = SECTOR_GROUPS.get(symbol, {})
    return CompanyProfile(
        ticker=symbol,
        name=str(info.get("shortName") or info.get("longName") or symbol),
        sector=str(info.get("sector") or static.get("sector") or "Unknown"),
        industry=str(info.get("industry") or static.get("industry") or "Unknown"),
    )


def _norm_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()


def _token_set(value: str) -> Set[str]:
    return {tok for tok in _norm_label(value).split() if len(tok) > 2}


def _industry_family(industry: str) -> str:
    low = _norm_label(industry)
    if any(k in low for k in ["capital market", "broker", "securit", "investment"]):
        return "capital_markets"
    if any(k in low for k in ["bank", "depository"]):
        return "banks"
    if any(k in low for k in ["asset management", "wealth management", "fund"]):
        return "asset_management"
    if any(k in low for k in ["credit", "lending", "consumer finance"]):
        return "credit"
    if "insurance" in low:
        return "insurance"
    return "other"


def _infer_sector_industry(
    ticker: str,
    name: str,
    sector: str,
    industry: str,
    info: Dict[str, object],
) -> Tuple[str, str]:
    inferred_sector = str(sector or "Unknown")
    inferred_industry = str(industry or "Unknown")
    text = " ".join(
        [
            str(name or ""),
            str(info.get("shortName") or ""),
            str(info.get("longName") or ""),
            str(info.get("longBusinessSummary") or ""),
            str(info.get("industry") or ""),
            str(info.get("sector") or ""),
        ]
    ).lower()

    hint = PROFILE_HINTS.get(_norm_label(str(ticker)))
    if hint:
        inferred_sector, inferred_industry = hint

    for phrase, mapped in PROFILE_HINTS.items():
        if phrase in text:
            inferred_sector, inferred_industry = mapped
            break

    if inferred_sector == "Unknown" or inferred_industry == "Unknown":
        for terms, mapped in INDUSTRY_HINTS:
            if any(term in text for term in terms):
                inferred_sector, inferred_industry = mapped
                break

    return inferred_sector, inferred_industry


@lru_cache(maxsize=128)
def _fetch_financials_cached(ticker: str) -> Dict[str, pd.DataFrame]:
    stock = yf.Ticker(ticker)
    try:
        income = stock.financials
    except Exception:
        income = pd.DataFrame()
    try:
        balance = stock.balance_sheet
    except Exception:
        balance = pd.DataFrame()
    try:
        cash_flow = stock.cashflow
    except Exception:
        cash_flow = pd.DataFrame()

    return {
        "income_statement": income if isinstance(income, pd.DataFrame) else pd.DataFrame(),
        "balance_sheet": balance if isinstance(balance, pd.DataFrame) else pd.DataFrame(),
        "cash_flow": cash_flow if isinstance(cash_flow, pd.DataFrame) else pd.DataFrame(),
    }


def fetch_financials(ticker: str) -> Dict[str, pd.DataFrame]:
    raw = _fetch_financials_cached(ticker.upper())
    return {
        "income_statement": raw["income_statement"].copy(),
        "balance_sheet": raw["balance_sheet"].copy(),
        "cash_flow": raw["cash_flow"].copy(),
    }


def find_peer_companies(ticker: str, max_peers: int = 8) -> Dict[str, object]:
    base = fetch_company_profile(ticker)
    base_info = _safe_info(base.ticker)
    base_sector = base.sector
    base_industry = base.industry

    if base.ticker in SECTOR_GROUPS:
        base_sector = SECTOR_GROUPS[base.ticker]["sector"]
        base_industry = SECTOR_GROUPS[base.ticker]["industry"]

    base_sector, base_industry = _infer_sector_industry(
        base.ticker,
        base.name,
        base_sector,
        base_industry,
        base_info,
    )
    base_family = _industry_family(base_industry)
    base_tokens = _token_set(base_industry)

    scored: List[Tuple[int, Dict[str, object]]] = []
    for symbol in DEFAULT_UNIVERSE:
        if symbol == base.ticker:
            continue

        metadata = SECTOR_GROUPS.get(symbol, {"sector": "Unknown", "industry": "Unknown"})
        cand_sector = str(metadata.get("sector") or "Unknown")
        cand_industry = str(metadata.get("industry") or "Unknown")
        cand_family = _industry_family(cand_industry)
        cand_tokens = _token_set(cand_industry)

        score = 0
        match_type = "none"
        reason = "No match signal."

        if base_industry != "Unknown" and cand_industry == base_industry:
            score = 120
            match_type = "industry_exact"
            reason = f"Exact industry match: {cand_industry}"
        elif base_family != "other" and cand_family == base_family:
            score = 95
            match_type = "business_model"
            reason = f"Similar business model family: {cand_family.replace('_', ' ')}"
        elif base_sector != "Unknown" and cand_sector == base_sector:
            score = 65
            match_type = "sector_match"
            reason = f"Same sector: {cand_sector}"
        else:
            continue

        token_overlap = len(base_tokens.intersection(cand_tokens))
        score += min(12, token_overlap * 4)

        if base_family == "capital_markets" and cand_family in {"capital_markets", "asset_management"}:
            score += 8
            if match_type != "industry_exact":
                match_type = "business_model"
                reason = "Capital-markets business model overlap"

        row: Dict[str, object] = {
            "ticker": symbol,
            "name": symbol,
            "sector": cand_sector,
            "industry": cand_industry,
            "match_type": match_type,
            "match_reason": reason,
            "match_score": score,
        }
        scored.append((score, row))

    scored.sort(key=lambda x: (-x[0], x[1]["ticker"]))
    peers = [row for _, row in scored[:max_peers]]

    if not peers:
        fallback_rows: List[Dict[str, object]] = []
        for symbol in DEFAULT_UNIVERSE:
            if symbol == base.ticker:
                continue
            metadata = SECTOR_GROUPS.get(symbol, {"sector": "Unknown", "industry": "Unknown"})
            fallback_rows.append(
                {
                    "ticker": symbol,
                    "name": symbol,
                    "sector": metadata["sector"],
                    "industry": metadata["industry"],
                    "match_type": "fallback",
                    "match_reason": "Fallback universe used due limited profile metadata.",
                    "match_score": 1,
                }
            )
            if len(fallback_rows) >= max_peers:
                break
        peers = fallback_rows

    return {
        "sector": base_sector,
        "industry": base_industry,
        "industry_family": base_family,
        "peers": peers,
    }
