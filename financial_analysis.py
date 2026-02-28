"""Financial statement parsing and metric computation with robust fallbacks."""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import pandas as pd


def _coerce_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return number


def _latest_from_rows(df: pd.DataFrame, row_names: Iterable[str]) -> Optional[float]:
    if df is None or df.empty:
        return None

    index_lookup = {str(idx).strip().lower(): idx for idx in df.index}

    for row_name in row_names:
        key = row_name.strip().lower()
        if key in index_lookup:
            idx = index_lookup[key]
            series = df.loc[idx]
            if isinstance(series, pd.Series):
                for value in series.tolist():
                    number = _coerce_float(value)
                    if number is not None:
                        return number
            else:
                number = _coerce_float(series)
                if number is not None:
                    return number

    # Contains-based fallback for schema variations.
    lower_index = [str(idx).lower() for idx in df.index]
    for row_name in row_names:
        target = row_name.lower()
        for pos, idx_lower in enumerate(lower_index):
            if target in idx_lower:
                series = df.loc[df.index[pos]]
                if isinstance(series, pd.Series):
                    for value in series.tolist():
                        number = _coerce_float(value)
                        if number is not None:
                            return number
                else:
                    number = _coerce_float(series)
                    if number is not None:
                        return number

    return None


def _safe_div(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def _normalize_info_dte(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    # Some providers report debt/equity as percentage points.
    if value > 20:
        return value / 100.0
    return value


def compute_metrics(
    financials: Dict[str, pd.DataFrame],
    company_info: Optional[Dict[str, object]] = None,
) -> Dict[str, Optional[float]]:
    info = company_info or {}

    income = financials.get("income_statement", pd.DataFrame())
    balance = financials.get("balance_sheet", pd.DataFrame())
    cash_flow = financials.get("cash_flow", pd.DataFrame())

    total_debt = _latest_from_rows(balance, ["total debt", "long term debt and capital lease obligation", "long term debt"]) 
    total_equity = _latest_from_rows(balance, ["stockholders equity", "total equity gross minority interest", "common stock equity"])
    current_assets = _latest_from_rows(balance, ["current assets", "cash cash equivalents and short term investments"])
    current_liabilities = _latest_from_rows(balance, ["current liabilities", "total liabilities net minority interest"])
    cash_and_equivalents = _latest_from_rows(balance, ["cash and cash equivalents", "cash cash equivalents and short term investments", "cash"])

    revenue_now = _latest_from_rows(income, ["total revenue", "revenue", "operating revenue"])
    revenue_prev = None
    if income is not None and not income.empty:
        for row in ["total revenue", "revenue", "operating revenue"]:
            value = _latest_from_rows(income, [row])
            if value is not None:
                series = income.loc[[idx for idx in income.index if row in str(idx).lower()][0]]
                if isinstance(series, pd.Series) and len(series.tolist()) > 1:
                    revenue_prev = _coerce_float(series.tolist()[1])
                break

    operating_expense_now = _latest_from_rows(income, ["operating expense", "total expenses", "selling general and administration"])
    operating_expense_prev = None
    if income is not None and not income.empty:
        for row in ["operating expense", "total expenses", "selling general and administration"]:
            matches = [idx for idx in income.index if row in str(idx).lower()]
            if matches:
                series = income.loc[matches[0]]
                if isinstance(series, pd.Series) and len(series.tolist()) > 1:
                    operating_expense_prev = _coerce_float(series.tolist()[1])
                break

    gross_profit = _latest_from_rows(income, ["gross profit"])
    operating_income = _latest_from_rows(income, ["operating income", "ebit", "total operating income as reported"])

    inventory_now = _latest_from_rows(balance, ["inventory"])
    inventory_prev = None
    if balance is not None and not balance.empty:
        inv_matches = [idx for idx in balance.index if "inventory" in str(idx).lower()]
        if inv_matches:
            inv_series = balance.loc[inv_matches[0]]
            if isinstance(inv_series, pd.Series) and len(inv_series.tolist()) > 1:
                inventory_prev = _coerce_float(inv_series.tolist()[1])

    operating_cash_flow = _latest_from_rows(cash_flow, ["operating cash flow", "cash flow from continuing operating activities"])

    # Fallbacks from quote info when statements are sparse or rate-limited.
    if total_debt is None:
        total_debt = _coerce_float(info.get("totalDebt"))
    if total_equity is None and total_debt is not None:
        dte_info = _normalize_info_dte(_coerce_float(info.get("debtToEquity")))
        if dte_info not in (None, 0):
            total_equity = total_debt / dte_info
    if current_assets is None or current_liabilities is None:
        cr_info = _coerce_float(info.get("currentRatio"))
        if cr_info is not None and current_assets is None and current_liabilities is None:
            current_assets = cr_info
            current_liabilities = 1.0
    if cash_and_equivalents is None:
        cash_and_equivalents = _coerce_float(info.get("totalCash"))

    if revenue_now is None:
        revenue_now = _coerce_float(info.get("totalRevenue"))
    if revenue_prev is None and revenue_now is not None:
        rg_info = _coerce_float(info.get("revenueGrowth"))
        if rg_info is not None and (1 + rg_info) != 0:
            revenue_prev = revenue_now / (1 + rg_info)

    if gross_profit is None and revenue_now is not None:
        gm_info = _coerce_float(info.get("grossMargins"))
        if gm_info is not None:
            gross_profit = revenue_now * gm_info

    if operating_income is None and revenue_now is not None:
        om_info = _coerce_float(info.get("operatingMargins"))
        if om_info is not None:
            operating_income = revenue_now * om_info

    if operating_cash_flow is None:
        operating_cash_flow = _coerce_float(info.get("operatingCashflow"))
    if operating_cash_flow is None:
        free_cf = _coerce_float(info.get("freeCashflow"))
        if free_cf is not None:
            operating_cash_flow = free_cf

    debt_to_equity = _safe_div(total_debt, total_equity)
    if debt_to_equity is None:
        debt_to_equity = _normalize_info_dte(_coerce_float(info.get("debtToEquity")))

    current_ratio = _safe_div(current_assets, current_liabilities)
    if current_ratio is None:
        current_ratio = _coerce_float(info.get("currentRatio"))

    cash_burn = None
    if operating_cash_flow is not None:
        cash_burn = -operating_cash_flow if operating_cash_flow < 0 else 0.0

    revenue_growth = None
    if revenue_now is not None and revenue_prev not in (None, 0):
        revenue_growth = (revenue_now - revenue_prev) / revenue_prev
    if revenue_growth is None:
        revenue_growth = _coerce_float(info.get("revenueGrowth"))

    expense_growth = None
    if operating_expense_now is not None and operating_expense_prev not in (None, 0):
        expense_growth = (operating_expense_now - operating_expense_prev) / operating_expense_prev

    inventory_growth = None
    if inventory_now is not None and inventory_prev not in (None, 0):
        inventory_growth = (inventory_now - inventory_prev) / inventory_prev

    gross_margin = _safe_div(gross_profit, revenue_now)
    if gross_margin is None:
        gross_margin = _coerce_float(info.get("grossMargins"))

    operating_margin = _safe_div(operating_income, revenue_now)
    if operating_margin is None:
        operating_margin = _coerce_float(info.get("operatingMargins"))

    return {
        "debt_to_equity": debt_to_equity,
        "current_ratio": current_ratio,
        "cash_burn": cash_burn,
        "revenue_growth": revenue_growth,
        "expense_growth": expense_growth,
        "inventory_growth": inventory_growth,
        "gross_margin": gross_margin,
        "operating_margin": operating_margin,
        "total_debt": total_debt,
        "cash_and_equivalents": cash_and_equivalents,
        "revenue": revenue_now,
        "operating_expense": operating_expense_now,
    }
