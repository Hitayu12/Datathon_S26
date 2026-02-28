"""Data ingestion for financials, company profile, ticker resolution, and peer discovery."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional

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
}

DEFAULT_UNIVERSE = list(SECTOR_GROUPS.keys())


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
    base_sector = base.sector
    base_industry = base.industry

    if base.ticker in SECTOR_GROUPS:
        base_sector = SECTOR_GROUPS[base.ticker]["sector"]
        base_industry = SECTOR_GROUPS[base.ticker]["industry"]

    peers: List[Dict[str, str]] = []
    for symbol in DEFAULT_UNIVERSE:
        if symbol == base.ticker:
            continue
        metadata = SECTOR_GROUPS.get(symbol, {"sector": "Unknown", "industry": "Unknown"})

        if metadata["industry"] == base_industry and base_industry != "Unknown":
            match_type = "industry"
        elif metadata["sector"] == base_sector and base_sector != "Unknown":
            match_type = "sector"
        else:
            continue

        peers.append(
            {
                "ticker": symbol,
                "name": symbol,
                "sector": metadata["sector"],
                "industry": metadata["industry"],
                "match_type": match_type,
            }
        )
        if len(peers) >= max_peers:
            break

    if not peers:
        for symbol in DEFAULT_UNIVERSE:
            if symbol == base.ticker:
                continue
            metadata = SECTOR_GROUPS.get(symbol, {"sector": "Unknown", "industry": "Unknown"})
            peers.append(
                {
                    "ticker": symbol,
                    "name": symbol,
                    "sector": metadata["sector"],
                    "industry": metadata["industry"],
                    "match_type": "fallback",
                }
            )
            if len(peers) >= max_peers:
                break

    return {
        "sector": base_sector,
        "industry": base_industry,
        "peers": peers,
    }
