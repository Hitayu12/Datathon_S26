"""Risk scoring, layered analysis, survivor comparison, and simulation."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Dict, Iterable, List, Optional, Tuple


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _round(value: Optional[float], digits: int = 4) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


@dataclass
class RiskWeights:
    debt: float = 0.24
    liquidity: float = 0.22
    revenue: float = 0.20
    burn: float = 0.20
    macro: float = 0.14


class MultiFactorRiskEngine:
    """Composite risk score normalized to 0-100."""

    def __init__(
        self,
        metrics: Dict[str, Optional[float]],
        macro_stress_score: float,
        weights: Optional[RiskWeights] = None,
    ) -> None:
        self.metrics = metrics
        self.macro_stress_score = _clamp((macro_stress_score or 0.0) / 100.0)
        self.weights = weights or RiskWeights()

    def _debt_risk(self) -> float:
        dte = self.metrics.get("debt_to_equity")
        if dte is None:
            return 0.5
        return _clamp(float(dte) / 4.0)

    def _liquidity_risk(self) -> float:
        current_ratio = self.metrics.get("current_ratio")
        if current_ratio is None:
            return 0.5
        return _clamp((1.5 - float(current_ratio)) / 1.5)

    def _revenue_risk(self) -> float:
        growth = self.metrics.get("revenue_growth")
        if growth is None:
            return 0.5
        if growth >= 0:
            return _clamp(0.3 - float(growth))
        return _clamp(abs(float(growth)) / 0.4)

    def _burn_risk(self) -> float:
        burn = self.metrics.get("cash_burn")
        revenue = self.metrics.get("revenue")
        if burn is None or revenue in (None, 0):
            return 0.5
        burn_ratio = float(burn) / abs(float(revenue))
        return _clamp(burn_ratio / 0.35)

    def _component_scores(self) -> Dict[str, float]:
        return {
            "debt_risk": self._debt_risk(),
            "liquidity_risk": self._liquidity_risk(),
            "revenue_risk": self._revenue_risk(),
            "burn_risk": self._burn_risk(),
            "macro_risk": self.macro_stress_score,
        }

    def compute_score(self) -> Tuple[float, Dict[str, float]]:
        components = self._component_scores()
        score_0_1 = (
            self.weights.debt * components["debt_risk"]
            + self.weights.liquidity * components["liquidity_risk"]
            + self.weights.revenue * components["revenue_risk"]
            + self.weights.burn * components["burn_risk"]
            + self.weights.macro * components["macro_risk"]
        )
        return round(score_0_1 * 100, 2), components


class LayeredAnalysisEngine:
    """Produces stress signals across macro, business, financial, operational, and qualitative layers."""

    def __init__(self, metrics: Dict[str, Optional[float]], qualitative_themes: Dict[str, int], macro_stress_score: float):
        self.metrics = metrics
        self.qualitative_themes = qualitative_themes
        self.macro_stress_score = macro_stress_score

    def macro_layer(self) -> Dict[str, object]:
        signals: List[str] = []
        if self.macro_stress_score >= 65:
            signals.append("Elevated macro stress (rates/credit/demand)")
        if self.macro_stress_score >= 80:
            signals.append("Severe macro headwinds")
        return {
            "macro_stress_score": round(self.macro_stress_score, 2),
            "signals": signals,
        }

    def business_model_layer(self) -> Dict[str, object]:
        signals: List[str] = []
        rev_growth = self.metrics.get("revenue_growth")
        if rev_growth is not None and rev_growth < 0:
            signals.append("Contracting top-line demand")

        if self.qualitative_themes.get("demand_decline", 0) > 0:
            signals.append("Demand pressure mentioned in qualitative disclosures")

        return {"signals": signals}

    def financial_health_layer(self) -> Dict[str, object]:
        signals: List[str] = []
        if (self.metrics.get("debt_to_equity") or 0) > 2.5:
            signals.append("High leverage")
        if (self.metrics.get("current_ratio") or 2) < 1.0:
            signals.append("Weak short-term liquidity")
        if (self.metrics.get("cash_burn") or 0) > 0:
            signals.append("Operating cash burn")
        return {"signals": signals}

    def operational_layer(self) -> Dict[str, object]:
        signals: List[str] = []
        expense_growth = self.metrics.get("expense_growth")
        revenue_growth = self.metrics.get("revenue_growth")
        if expense_growth is not None and revenue_growth is not None and expense_growth > revenue_growth:
            signals.append("Expenses growing faster than revenue")
        operating_margin = self.metrics.get("operating_margin")
        if operating_margin is not None and operating_margin < 0:
            signals.append("Negative operating margin")
        inventory_growth = self.metrics.get("inventory_growth")
        if inventory_growth is not None and inventory_growth > 0.25:
            signals.append("Inventory build-up risk")
        return {"signals": signals}

    def qualitative_layer(self) -> Dict[str, object]:
        theme_map = {
            "liquidity_concerns": "Liquidity concerns in disclosures/news",
            "debt_stress": "Debt covenant or refinancing pressure",
            "demand_decline": "Declining demand language",
            "margin_pressure": "Margin compression language",
            "legal_regulatory": "Legal/regulatory overhang",
            "bankruptcy_language": "Distress/restructuring language",
        }
        signals = [label for key, label in theme_map.items() if self.qualitative_themes.get(key, 0) > 0]
        return {"signals": signals}

    def analyze_all_layers(self) -> Dict[str, Dict[str, object]]:
        return {
            "macro": self.macro_layer(),
            "business_model": self.business_model_layer(),
            "financial_health": self.financial_health_layer(),
            "operational": self.operational_layer(),
            "qualitative": self.qualitative_layer(),
        }


def average_metrics(metric_rows: Iterable[Dict[str, Optional[float]]]) -> Dict[str, Optional[float]]:
    rows = list(metric_rows)
    if not rows:
        return {}

    keys = set().union(*rows)
    output: Dict[str, Optional[float]] = {}
    for key in keys:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        output[key] = round(mean(values), 6) if values else None
    return output


def compare_failure_vs_survivors(
    failing_metrics: Dict[str, Optional[float]],
    survivor_metric_rows: List[Dict[str, Optional[float]]],
    macro_stress_score: float,
) -> Dict[str, object]:
    failing_score, failing_components = MultiFactorRiskEngine(failing_metrics, macro_stress_score).compute_score()
    survivor_avg = average_metrics(survivor_metric_rows)
    survivor_score, survivor_components = MultiFactorRiskEngine(survivor_avg, macro_stress_score).compute_score()

    metric_gaps = {
        "debt_to_equity_gap": _round((failing_metrics.get("debt_to_equity") or 0) - (survivor_avg.get("debt_to_equity") or 0)),
        "current_ratio_gap": _round((survivor_avg.get("current_ratio") or 0) - (failing_metrics.get("current_ratio") or 0)),
        "revenue_growth_gap": _round((survivor_avg.get("revenue_growth") or 0) - (failing_metrics.get("revenue_growth") or 0)),
        "cash_burn_gap": _round((failing_metrics.get("cash_burn") or 0) - (survivor_avg.get("cash_burn") or 0)),
    }

    return {
        "failing_score": failing_score,
        "survivor_score": survivor_score,
        "failing_components": failing_components,
        "survivor_components": survivor_components,
        "survivor_average_metrics": survivor_avg,
        "metric_gaps": metric_gaps,
    }


def simulate_counterfactual(
    failing_metrics: Dict[str, Optional[float]],
    survivor_average_metrics: Dict[str, Optional[float]],
    macro_stress_score: float,
) -> Dict[str, object]:
    original_score, _ = MultiFactorRiskEngine(failing_metrics, macro_stress_score).compute_score()

    adjusted_metrics = dict(failing_metrics)
    for key in ["debt_to_equity", "current_ratio", "cash_burn", "revenue_growth"]:
        if survivor_average_metrics.get(key) is not None:
            adjusted_metrics[key] = survivor_average_metrics[key]

    adjusted_score, _ = MultiFactorRiskEngine(adjusted_metrics, macro_stress_score).compute_score()

    improvement_pct = 0.0
    if original_score > 0:
        improvement_pct = round((original_score - adjusted_score) / original_score * 100, 2)

    return {
        "original_score": original_score,
        "adjusted_score": adjusted_score,
        "improvement_percentage": improvement_pct,
        "adjusted_metrics": adjusted_metrics,
    }


def generate_strategy_recommendations(
    failing_metrics: Dict[str, Optional[float]],
    survivor_avg_metrics: Dict[str, Optional[float]],
) -> List[str]:
    recommendations: List[str] = []

    fd, sd = failing_metrics.get("debt_to_equity"), survivor_avg_metrics.get("debt_to_equity")
    if fd is not None and sd is not None and fd > sd:
        reduction = max(0.0, (fd - sd) / max(fd, 1e-6) * 100)
        recommendations.append(f"Reduce leverage by about {reduction:.1f}% to approach survivor debt profile.")

    fc, sc = failing_metrics.get("current_ratio"), survivor_avg_metrics.get("current_ratio")
    if fc is not None and sc is not None and fc < sc:
        recommendations.append(f"Build liquidity buffer: current ratio from {fc:.2f} toward {sc:.2f}.")

    fr, sr = failing_metrics.get("revenue_growth"), survivor_avg_metrics.get("revenue_growth")
    if fr is not None and sr is not None and fr < sr:
        recommendations.append("Stabilize revenue mix and prioritize segments with resilient demand.")

    fb, sb = failing_metrics.get("cash_burn"), survivor_avg_metrics.get("cash_burn")
    if fb is not None and sb is not None and fb > sb:
        recommendations.append("Reduce fixed-cost base and tighten working-capital cycles to lower cash burn.")

    if not recommendations:
        recommendations.append("Maintain current capital and operating discipline; monitor macro stress weekly.")

    return recommendations[:4]
