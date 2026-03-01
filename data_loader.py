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
    "GM": {"sector": "Consumer Cyclical", "industry": "Auto Manufacturers"},
    "STLA": {"sector": "Consumer Cyclical", "industry": "Auto Manufacturers"},
    "TM": {"sector": "Consumer Cyclical", "industry": "Auto Manufacturers"},
    "HMC": {"sector": "Consumer Cyclical", "industry": "Auto Manufacturers"},
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
    "BBY": {"sector": "Consumer Cyclical", "industry": "Specialty Retail"},
    "ULTA": {"sector": "Consumer Cyclical", "industry": "Specialty Retail"},
    "ANF": {"sector": "Consumer Cyclical", "industry": "Specialty Retail"},
    "SBUX": {"sector": "Consumer Cyclical", "industry": "Restaurants"},
    "MCD": {"sector": "Consumer Cyclical", "industry": "Restaurants"},
    "DIS": {"sector": "Communication Services", "industry": "Entertainment"},
    "CMCSA": {"sector": "Communication Services", "industry": "Telecom Services"},
    "AAL": {"sector": "Industrials", "industry": "Airlines"},
    "DAL": {"sector": "Industrials", "industry": "Airlines"},
    "UAL": {"sector": "Industrials", "industry": "Airlines"},
    "LUV": {"sector": "Industrials", "industry": "Airlines"},
    "ALK": {"sector": "Industrials", "industry": "Airlines"},
    "JBLU": {"sector": "Industrials", "industry": "Airlines"},
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
    # ── Historically Failed / Distressed Companies ─────────────────────────────
    "LEHMQ": {"sector": "Financial Services", "industry": "Capital Markets"},
    "ENRNQ": {"sector": "Energy", "industry": "Oil & Gas"},
    "WCOEQ": {"sector": "Communication Services", "industry": "Telecom Services"},
    "WCOME": {"sector": "Communication Services", "industry": "Telecom Services"},
    "BLOAQ": {"sector": "Consumer Cyclical", "industry": "Video Rental"},
    "KODK":  {"sector": "Technology", "industry": "Photographic Equipment"},
    "SHLDQ": {"sector": "Consumer Cyclical", "industry": "Department Stores"},
    "RADIOSHACK": {"sector": "Consumer Cyclical", "industry": "Electronics Retail"},
    "RSHCQ": {"sector": "Consumer Cyclical", "industry": "Electronics Retail"},
    "CCTYQ": {"sector": "Consumer Cyclical", "industry": "Electronics Retail"},
    "GMGMQ": {"sector": "Consumer Cyclical", "industry": "Auto Manufacturers"},
    "CHRYSLER": {"sector": "Consumer Cyclical", "industry": "Auto Manufacturers"},
    "WAMUQ": {"sector": "Financial Services", "industry": "Banks"},
    "BSTRQ": {"sector": "Financial Services", "industry": "Capital Markets"},
    "MERQ":  {"sector": "Financial Services", "industry": "Capital Markets"},
    "GFGPQ": {"sector": "Consumer Defensive", "industry": "Food Distribution"},
    "NORTELQ": {"sector": "Technology", "industry": "Networking Equipment"},
    "NTIQ":  {"sector": "Technology", "industry": "Networking Equipment"},
    "SUNWQ": {"sector": "Technology", "industry": "Software"},
    "MYSPACEQ": {"sector": "Technology", "industry": "Social Media"},
    "YAHOOQ": {"sector": "Technology", "industry": "Internet Content"},
    "BBBYQ": {"sector": "Consumer Cyclical", "industry": "Specialty Retail"},
    "REVLONQ": {"sector": "Consumer Defensive", "industry": "Personal Care"},
    "REVNQ": {"sector": "Consumer Defensive", "industry": "Personal Care"},
    "HERTZ": {"sector": "Consumer Cyclical", "industry": "Car Rentals"},
    "HTZGQ": {"sector": "Consumer Cyclical", "industry": "Car Rentals"},
    "JCPNQ": {"sector": "Consumer Cyclical", "industry": "Department Stores"},
    "JCPENNEY": {"sector": "Consumer Cyclical", "industry": "Department Stores"},
    "NEIMQ": {"sector": "Consumer Cyclical", "industry": "Luxury Retail"},
    "CARESQ": {"sector": "Healthcare", "industry": "Healthcare Services"},
    "PMCQ":  {"sector": "Healthcare", "industry": "Drug Manufacturers"},
    "TXUQ":  {"sector": "Energy", "industry": "Electric Utilities"},
    "DYNQ":  {"sector": "Energy", "industry": "Oil & Gas"},
    "CHPTRQ": {"sector": "Consumer Cyclical", "industry": "Electric Vehicle Charging"},
    "SVBQ":  {"sector": "Financial Services", "industry": "Banks"},
    "SIVBQ": {"sector": "Financial Services", "industry": "Banks"},
    "SBNYQ": {"sector": "Financial Services", "industry": "Banks"},
    "FNMAQ": {"sector": "Financial Services", "industry": "Mortgage Finance"},
    "FMCCQ": {"sector": "Financial Services", "industry": "Mortgage Finance"},
    "AMRQ":  {"sector": "Industrials", "industry": "Airlines"},
    "DALRQ": {"sector": "Industrials", "industry": "Airlines"},
    "UAIRQ": {"sector": "Industrials", "industry": "Airlines"},
    "AAMRQ": {"sector": "Industrials", "industry": "Airlines"},
    "ZMTP":  {"sector": "Consumer Cyclical", "industry": "Internet Retail"},
    "PETS":  {"sector": "Consumer Cyclical", "industry": "Internet Retail"},
    "TOYSQ": {"sector": "Consumer Cyclical", "industry": "Specialty Retail"},
    "SRSCQ": {"sector": "Consumer Cyclical", "industry": "Specialty Retail"},
}

DEFAULT_UNIVERSE = list(SECTOR_GROUPS.keys())

PROFILE_HINTS: Dict[str, Tuple[str, str]] = {
    # Banks & Financial
    "lehman": ("Financial Services", "Capital Markets"),
    "lehman brothers": ("Financial Services", "Capital Markets"),
    "bear stearns": ("Financial Services", "Capital Markets"),
    "countrywide": ("Financial Services", "Credit Services"),
    "washington mutual": ("Financial Services", "Banks"),
    "wamu": ("Financial Services", "Banks"),
    "svb": ("Financial Services", "Banks"),
    "silicon valley bank": ("Financial Services", "Banks"),
    "signature bank": ("Financial Services", "Banks"),
    "first republic": ("Financial Services", "Banks"),
    "indymac": ("Financial Services", "Banks"),
    "merrill lynch": ("Financial Services", "Capital Markets"),
    "wachovia": ("Financial Services", "Banks"),
    "fannie mae": ("Financial Services", "Mortgage Finance"),
    "freddie mac": ("Financial Services", "Mortgage Finance"),
    "aig": ("Financial Services", "Insurance"),
    # Energy
    "enron": ("Energy", "Oil & Gas"),
    "dynegy": ("Energy", "Oil & Gas"),
    "txu": ("Energy", "Electric Utilities"),
    "energy future": ("Energy", "Electric Utilities"),
    "peabody energy": ("Energy", "Coal"),
    "chesapeake energy": ("Energy", "Oil & Gas"),
    "oasis petroleum": ("Energy", "Oil & Gas"),
    "denbury resources": ("Energy", "Oil & Gas"),
    # Telecom & Tech
    "worldcom": ("Communication Services", "Telecom Services"),
    "mci": ("Communication Services", "Telecom Services"),
    "nortel": ("Technology", "Networking Equipment"),
    "nortel networks": ("Technology", "Networking Equipment"),
    "lucent": ("Technology", "Networking Equipment"),
    "sun microsystems": ("Technology", "Software"),
    "compaq": ("Technology", "Personal Computers"),
    "palm": ("Technology", "Personal Computers"),
    "blackberry": ("Technology", "Consumer Electronics"),
    "myspace": ("Technology", "Social Media"),
    "digg": ("Technology", "Social Media"),
    "yahoo": ("Technology", "Internet Content"),
    "lycos": ("Technology", "Internet Content"),
    "altavista": ("Technology", "Internet Content"),
    "webvan": ("Consumer Cyclical", "Internet Retail"),
    "pets.com": ("Consumer Cyclical", "Internet Retail"),
    "kozmo": ("Consumer Cyclical", "Internet Retail"),
    "theglobe": ("Technology", "Internet Content"),
    "mp3.com": ("Communication Services", "Entertainment"),
    # Retail & Consumer
    "sears": ("Consumer Cyclical", "Department Stores"),
    "kmart": ("Consumer Cyclical", "Department Stores"),
    "jc penney": ("Consumer Cyclical", "Department Stores"),
    "jcpenney": ("Consumer Cyclical", "Department Stores"),
    "neiman marcus": ("Consumer Cyclical", "Luxury Retail"),
    "lord & taylor": ("Consumer Cyclical", "Department Stores"),
    "toys r us": ("Consumer Cyclical", "Specialty Retail"),
    "toysrus": ("Consumer Cyclical", "Specialty Retail"),
    "circuit city": ("Consumer Cyclical", "Electronics Retail"),
    "radioshack": ("Consumer Cyclical", "Electronics Retail"),
    "radio shack": ("Consumer Cyclical", "Electronics Retail"),
    "blockbuster": ("Consumer Cyclical", "Video Rental"),
    "borders": ("Consumer Cyclical", "Specialty Retail"),
    "bed bath beyond": ("Consumer Cyclical", "Specialty Retail"),
    "bed bath & beyond": ("Consumer Cyclical", "Specialty Retail"),
    "pier 1": ("Consumer Cyclical", "Specialty Retail"),
    "tuesday morning": ("Consumer Cyclical", "Specialty Retail"),
    "revlon": ("Consumer Defensive", "Personal Care"),
    "rite aid": ("Healthcare", "Drug Stores"),
    # Auto
    "general motors": ("Consumer Cyclical", "Auto Manufacturers"),
    "gm": ("Consumer Cyclical", "Auto Manufacturers"),
    "chrysler": ("Consumer Cyclical", "Auto Manufacturers"),
    "saab": ("Consumer Cyclical", "Auto Manufacturers"),
    "pontiac": ("Consumer Cyclical", "Auto Manufacturers"),
    # Travel & Hospitality
    "hertz": ("Consumer Cyclical", "Car Rentals"),
    "american airlines": ("Industrials", "Airlines"),
    "american air": ("Industrials", "Airlines"),
    "delta airlines": ("Industrials", "Airlines"),
    "united airlines": ("Industrials", "Airlines"),
    "pan am": ("Industrials", "Airlines"),
    "pan american": ("Industrials", "Airlines"),
    "eastern airlines": ("Industrials", "Airlines"),
    "twa": ("Industrials", "Airlines"),
    # Infra & Industrial
    "kodak": ("Technology", "Photographic Equipment"),
    "eastman kodak": ("Technology", "Photographic Equipment"),
    "polaroid": ("Technology", "Photographic Equipment"),
    "bethlehem steel": ("Industrials", "Steel"),
    "us steel": ("Industrials", "Steel"),
    "conseco": ("Financial Services", "Insurance"),
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
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "application/json",
            },
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
        candidates.append(
            {
                "ticker": symbol,
                "name": shortname,
                "exchange": str(row.get("exchange") or "").strip().upper(),
                "sector": str(row.get("sectorDisp") or "Unknown").strip() or "Unknown",
                "industry": str(row.get("industryDisp") or "Unknown").strip() or "Unknown",
                "quote_type": quote_type,
            }
        )
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


def _score_peer_match(
    *,
    base_ticker: str,
    base_sector: str,
    base_industry: str,
    base_family: str,
    base_tokens: Set[str],
    base_name_tokens: Set[str],
    symbol: str,
    cand_name: str,
    cand_sector: str,
    cand_industry: str,
    source: str,
) -> Optional[Dict[str, object]]:
    if symbol == base_ticker:
        return None

    cand_family = _industry_family(cand_industry)
    cand_tokens = _token_set(cand_industry)
    cand_name_tokens = _token_set(cand_name)

    score = 0
    match_type = "none"
    reason = "No match signal."

    base_unknown = base_industry == "Unknown" and base_sector == "Unknown"
    if base_unknown and source == "dynamic":
        if cand_industry != "Unknown":
            score = 76
            match_type = "query_competitor"
            reason = "Peer returned by competitor search for this company."
        elif cand_sector != "Unknown":
            score = 62
            match_type = "query_competitor"
            reason = "Peer returned by sector-level competitor search."
        elif cand_name and cand_name != symbol:
            score = 54
            match_type = "query_competitor"
            reason = "Peer returned by ticker/name search for this company."
        else:
            return None
    elif base_industry != "Unknown" and cand_industry == base_industry:
        score = 130
        match_type = "industry_exact"
        reason = f"Exact industry match: {cand_industry}"
    elif base_family != "other" and cand_family == base_family:
        score = 108
        match_type = "business_model"
        reason = f"Similar business model family: {cand_family.replace('_', ' ')}"
    elif base_sector != "Unknown" and cand_sector == base_sector:
        score = 74
        match_type = "sector_match"
        reason = f"Same sector: {cand_sector}"
    else:
        industry_overlap = len(base_tokens.intersection(cand_tokens))
        if industry_overlap > 0:
            score = 52 + min(16, industry_overlap * 4)
            match_type = "industry_keyword_overlap"
            reason = "Industry keyword overlap with target company."
        else:
            return None

    token_overlap = len(base_tokens.intersection(cand_tokens))
    name_overlap = len(base_name_tokens.intersection(cand_name_tokens))
    score += min(16, token_overlap * 4)
    score += min(10, name_overlap * 2)

    if base_family == "capital_markets" and cand_family in {"capital_markets", "asset_management"}:
        score += 8
        if match_type not in {"industry_exact", "business_model"}:
            match_type = "business_model"
            reason = "Capital-markets business model overlap."

    if source == "dynamic":
        score += 5

    if cand_sector == "Unknown" and cand_industry == "Unknown":
        score -= 18

    return {
        "ticker": symbol,
        "name": cand_name or symbol,
        "sector": cand_sector,
        "industry": cand_industry,
        "match_type": match_type,
        "match_reason": reason,
        "match_score": max(1, int(score)),
        "match_source": source,
    }


def find_peer_companies(ticker: str, max_peers: int = 8) -> Dict[str, object]:
    base = fetch_company_profile(ticker)
    base_info = _safe_info(base.ticker)
    base_sector = base.sector
    base_industry = base.industry
    base_exchange = str(base_info.get("exchange") or "").strip().upper()

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
    base_lookup = _yahoo_search(base.ticker, max_results=8)
    for row in base_lookup:
        if str(row.get("ticker") or "").upper() != base.ticker:
            continue
        base_exchange = str(row.get("exchange") or base_exchange).strip().upper()
        row_sector = str(row.get("sector") or "Unknown")
        row_industry = str(row.get("industry") or "Unknown")
        if base_sector == "Unknown" and row_sector != "Unknown":
            base_sector = row_sector
        if base_industry == "Unknown" and row_industry != "Unknown":
            base_industry = row_industry
        break

    base_family = _industry_family(base_industry)
    base_tokens = set() if base_industry == "Unknown" else _token_set(base_industry)
    base_name_tokens = _token_set(base.name)
    us_exchanges = {"NMS", "NYQ", "ASE", "PNK", "BTS"}

    dynamic_queries = [
        base.ticker,
        base.name,
        f"{base.name} stock",
        f"{base.name} competitors",
        f"{base.name} peers",
        f"{base.industry} public companies",
        f"{base.sector} {base_family.replace('_', ' ')} companies",
    ]
    if base_info.get("shortName") and str(base_info.get("shortName")) != base.name:
        dynamic_queries.append(f"{base_info.get('shortName')} competitors")

    deduped_queries: List[str] = []
    seen_queries: Set[str] = set()
    for query in dynamic_queries:
        q = str(query or "").strip()
        if not q or q.lower() == "unknown":
            continue
        key = q.lower()
        if key in seen_queries:
            continue
        seen_queries.add(key)
        deduped_queries.append(q)

    dynamic_candidates: List[Dict[str, str]] = []
    seen_dynamic: Set[str] = set()
    for query in deduped_queries:
        for row in _yahoo_search(query, max_results=14):
            symbol = str(row.get("ticker") or "").upper().strip()
            if not symbol or symbol == base.ticker or symbol in seen_dynamic:
                continue
            seen_dynamic.add(symbol)
            dynamic_candidates.append({"ticker": symbol, "name": str(row.get("name") or symbol)})
            if len(dynamic_candidates) >= 36:
                break
        if len(dynamic_candidates) >= 36:
            break

    scored_by_ticker: Dict[str, Dict[str, object]] = {}

    for symbol in DEFAULT_UNIVERSE:
        if symbol == base.ticker:
            continue
        metadata = SECTOR_GROUPS.get(symbol, {"sector": "Unknown", "industry": "Unknown"})
        row = _score_peer_match(
            base_ticker=base.ticker,
            base_sector=base_sector,
            base_industry=base_industry,
            base_family=base_family,
            base_tokens=base_tokens,
            base_name_tokens=base_name_tokens,
            symbol=symbol,
            cand_name=symbol,
            cand_sector=str(metadata.get("sector") or "Unknown"),
            cand_industry=str(metadata.get("industry") or "Unknown"),
            source="static",
        )
        if row is None:
            continue
        prev = scored_by_ticker.get(symbol)
        if prev is None or float(row.get("match_score", 0)) > float(prev.get("match_score", 0)):
            scored_by_ticker[symbol] = row

    for cand in dynamic_candidates:
        symbol = str(cand.get("ticker") or "").upper().strip()
        if not symbol or symbol == base.ticker:
            continue
        cand_exchange = str(cand.get("exchange") or "").strip().upper()
        if base_exchange in us_exchanges and cand_exchange and cand_exchange not in us_exchanges:
            continue
        if base_exchange in us_exchanges and "." in symbol and symbol not in SECTOR_GROUPS:
            continue

        info = {}
        if str(cand.get("sector") or "Unknown") == "Unknown" and str(cand.get("industry") or "Unknown") == "Unknown":
            info = _safe_info(symbol)
        metadata = SECTOR_GROUPS.get(symbol, {})
        cand_name = str(cand.get("name") or info.get("shortName") or info.get("longName") or symbol)
        cand_sector = str(cand.get("sector") or info.get("sector") or metadata.get("sector") or "Unknown")
        cand_industry = str(cand.get("industry") or info.get("industry") or metadata.get("industry") or "Unknown")
        cand_sector, cand_industry = _infer_sector_industry(symbol, cand_name, cand_sector, cand_industry, info)

        row = _score_peer_match(
            base_ticker=base.ticker,
            base_sector=base_sector,
            base_industry=base_industry,
            base_family=base_family,
            base_tokens=base_tokens,
            base_name_tokens=base_name_tokens,
            symbol=symbol,
            cand_name=cand_name,
            cand_sector=cand_sector,
            cand_industry=cand_industry,
            source="dynamic",
        )
        if row is None:
            continue
        prev = scored_by_ticker.get(symbol)
        if prev is None or float(row.get("match_score", 0)) > float(prev.get("match_score", 0)):
            scored_by_ticker[symbol] = row

    peers = sorted(
        scored_by_ticker.values(),
        key=lambda row: (-float(row.get("match_score", 0)), str(row.get("ticker", ""))),
    )[:max_peers]

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
                    "match_source": "fallback",
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
