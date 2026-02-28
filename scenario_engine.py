"""Scenario engine: generates multiple named counterfactual paths
with macro-micro linkage and company-specific feasibility scoring."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from risk_model import MultiFactorRiskEngine


# ── Macro-to-Micro transmission map ──────────────────────────────────────
# Shows which macro force affects which financial metric and in what direction.
# This is the bridge your current system is missing.

MACRO_MICRO_LINKAGE: Dict[str, Dict[str, object]] = {
    "high_interest_rates": {
        "affects": ["debt_to_equity", "cash_burn"],
        "direction": "worsens",
        "mechanism": "Refinancing costs rise; floating-rate debt burden increases",
        "policy_context": "Fed funds rate above 4% threshold",
    },
    "ev_subsidy_tailwind": {
        "affects": ["revenue_growth", "current_ratio"],
        "direction": "improves",
        "mechanism": "IRA tax credits reduce effective customer cost; demand pull-forward",
        "policy_context": "Inflation Reduction Act EV credit eligibility",
    },
    "supply_chain_shock": {
        "affects": ["operating_margin", "cash_burn", "inventory_growth"],
        "direction": "worsens",
        "mechanism": "Input cost inflation and delivery delays compress margin",
        "policy_context": "Semiconductor shortage / logistics disruption period",
    },
    "credit_tightening": {
        "affects": ["debt_to_equity", "liquidity_runway_months"],
        "direction": "worsens",
        "mechanism": "Capital markets close for distressed issuers; refinancing blocked",
        "policy_context": "HY spread above 500bps signals closed markets",
    },
    "demand_softening": {
        "affects": ["revenue_growth", "inventory_growth"],
        "direction": "worsens",
        "mechanism": "Consumer pullback reduces orders; inventory builds",
        "policy_context": "Consumer sentiment index below 65",
    },
    "tech_disruption_pressure": {
        "affects": ["gross_margin", "operating_margin"],
        "direction": "worsens",
        "mechanism": "Faster competitors compress pricing power and R&D spend",
        "policy_context": "Industry technology cycle inflection point",
    },
}


@dataclass
class ScenarioOutcome:
    """One named counterfactual path with full context."""
    name: str                          # e.g. "Early Capital Raise"
    strategy_description: str          # what the company would have done
    survivor_reference: str            # which survivor actually did this
    adjusted_metrics: Dict[str, Optional[float]]
    adjusted_score: float
    original_score: float
    risk_reduction: float              # points
    improvement_pct: float             # percentage
    feasibility: str                   # "high" / "medium" / "low"
    feasibility_score: float           # 0.0–1.0
    feasibility_reasoning: str         # why this rating for THIS company
    macro_enablers: List[str]          # macro conditions that would help
    macro_blockers: List[str]          # macro conditions working against it
    micro_impacts: List[str]           # specific metric changes and why
    what_had_to_be_true: List[str]     # conditions required for this to work
    likelihood_label: str              # "Plausible", "Difficult", "Unlikely"


@dataclass  
class ScenarioBundle:
    """Full multi-scenario output for one company analysis."""
    company_ticker: str
    original_score: float
    scenarios: List[ScenarioOutcome]
    active_macro_forces: List[str]     # which macro forces are live
    macro_micro_chains: List[str]      # plain-English transmission explanations
    best_scenario: str                 # name of highest-impact feasible path
    consensus_gap: str                 # the one gap all scenarios try to close


class MacroMicroBridge:
    """
    Translates macro stress signals into specific metric-level impacts.
    This is what connects policy/tech/demand forces to financial outcomes.
    """

    def __init__(
        self,
        macro_stress_score: float,
        tavily_macro_notes: List[str],
        industry: str,
    ):
        self.macro_stress_score = macro_stress_score
        self.notes_text = " ".join(tavily_macro_notes).lower()
        self.industry = industry.lower()

    def identify_active_forces(self) -> List[str]:
        """Detect which macro forces are live based on Tavily content."""
        active = []
        detection_rules = {
            "high_interest_rates":    ["rate", "fed", "interest", "yield", "tightening"],
            "ev_subsidy_tailwind":    ["ira", "subsidy", "tax credit", "ev credit", "incentive"],
            "supply_chain_shock":     ["supply chain", "shortage", "logistics", "semiconductor"],
            "credit_tightening":      ["credit", "spread", "default", "refinanc", "covenant"],
            "demand_softening":       ["demand", "slowdown", "consumer", "sentiment", "weak"],
            "tech_disruption_pressure": ["technology", "disruption", "competitor", "platform"],
        }
        for force, keywords in detection_rules.items():
            if any(kw in self.notes_text for kw in keywords):
                active.append(force)
        # Always include credit tightening if macro stress is high
        if self.macro_stress_score >= 65 and "credit_tightening" not in active:
            active.append("credit_tightening")
        return active

    def build_transmission_chains(
        self,
        active_forces: List[str],
        failing_metrics: Dict[str, Optional[float]],
    ) -> List[str]:
        """
        Build plain-English chains: macro force → metric impact → financial outcome.
        These are what Groq will use to explain WHY the gap matters.
        """
        chains = []
        for force in active_forces:
            linkage = MACRO_MICRO_LINKAGE.get(force, {})
            if not linkage:
                continue
            affected = linkage.get("affects", [])
            mechanism = linkage.get("mechanism", "")
            policy = linkage.get("policy_context", "")
            direction = linkage.get("direction", "")

            # Only include chain if the affected metrics are actually stressed
            relevant_metrics = []
            for metric in affected:
                val = failing_metrics.get(metric)
                if val is not None:
                    relevant_metrics.append(
                        f"{metric}={val:.3f}"
                    )

            if relevant_metrics:
                chain = (
                    f"{force.replace('_', ' ').title()} "
                    f"[{policy}] → {mechanism} → "
                    f"{direction}: {', '.join(relevant_metrics)}"
                )
                chains.append(chain)

        return chains


class ScenarioEngine:
    """
    Generates 3–4 named counterfactual scenarios, each grounded in
    a real survivor strategy with macro context and feasibility scoring.
    """

    # Named scenario templates — each maps to a real survivor pattern
    # This replaces generic metric swapping with strategy-named paths
    SCENARIO_TEMPLATES = [
        {
            "name": "Early Capital Raise",
            "strategy_description": (
                "Secure equity or debt financing 12–18 months before cash runway "
                "drops below 6 months. Reduces burn pressure and buys production time."
            ),
            "survivor_reference": "Rivian (raised $13.5B pre-production); Tesla (Series D timing)",
            "target_metrics": {
                "cash_burn": "reduce_by_pct",    # action type
                "current_ratio": "move_toward_survivor",
                "debt_to_equity": "hold",
            },
            "feasibility_factors": {
                "requires_market_access": True,
                "blocked_by": ["credit_tightening", "high_interest_rates"],
                "enabled_by": ["ev_subsidy_tailwind"],
                "company_prerequisites": ["current_ratio > 0.8", "no_covenant_breach"],
            },
        },
        {
            "name": "Supplier Diversification",
            "strategy_description": (
                "Reduce single-supplier dependency by qualifying 2+ alternative "
                "vendors for critical components. Reduces supply shock exposure."
            ),
            "survivor_reference": "Tesla (vertical integration + multi-source); NIO (supplier network)",
            "target_metrics": {
                "operating_margin": "move_toward_survivor",
                "cash_burn": "reduce_by_pct",
                "inventory_growth": "normalize",
            },
            "feasibility_factors": {
                "requires_market_access": False,
                "blocked_by": ["supply_chain_shock"],
                "enabled_by": [],
                "company_prerequisites": ["revenue > 100M", "operating_margin > -0.3"],
            },
        },
        {
            "name": "Revenue Mix Diversification",
            "strategy_description": (
                "Add a secondary revenue stream (fleet services, software/OTA, "
                "licensing) to reduce dependence on single product unit sales."
            ),
            "survivor_reference": "Tesla (energy + services); Rivian (Amazon fleet contract)",
            "target_metrics": {
                "revenue_growth": "move_toward_survivor",
                "gross_margin": "move_toward_survivor",
                "operating_margin": "improve_partial",
            },
            "feasibility_factors": {
                "requires_market_access": False,
                "blocked_by": ["demand_softening", "tech_disruption_pressure"],
                "enabled_by": ["ev_subsidy_tailwind"],
                "company_prerequisites": [
                    "commercialization_stage != pre-revenue",
                    "has_product_delivered"
                ],
            },
        },
        {
            "name": "Controlled Scale-Down",
            "strategy_description": (
                "Deliberately reduce production targets and headcount to extend "
                "runway. Accept lower volume in exchange for survival."
            ),
            "survivor_reference": "Canoo (restructuring pivot); Lucid (production guidance reset)",
            "target_metrics": {
                "cash_burn": "aggressive_reduce",
                "operating_expense": "reduce_by_pct",
                "current_ratio": "improve_partial",
            },
            "feasibility_factors": {
                "requires_market_access": False,
                "blocked_by": [],
                "enabled_by": [],
                "company_prerequisites": ["management_willingness", "no_bankruptcy_filing"],
            },
        },
    ]

    def __init__(
        self,
        failing_metrics: Dict[str, Optional[float]],
        survivor_average_metrics: Dict[str, Optional[float]],
        macro_stress_score: float,
        active_macro_forces: List[str],
        macro_micro_chains: List[str],
        industry: str,
    ):
        self.failing_metrics = failing_metrics
        self.survivor_avg = survivor_average_metrics
        self.macro_stress_score = macro_stress_score
        self.active_forces = active_macro_forces
        self.chains = macro_micro_chains
        self.industry = industry

    def _apply_scenario_metrics(
        self,
        template: Dict,
    ) -> Dict[str, Optional[float]]:
        """
        Apply each scenario's target metric changes to produce
        adjusted metrics. Uses survivor averages as targets where
        action type is 'move_toward_survivor'.
        """
        adjusted = dict(self.failing_metrics)
        targets = template.get("target_metrics", {})

        for metric, action in targets.items():
            failing_val = self.failing_metrics.get(metric)
            survivor_val = self.survivor_avg.get(metric)

            if action == "move_toward_survivor" and survivor_val is not None:
                # Full survivor target
                adjusted[metric] = survivor_val

            elif action == "reduce_by_pct" and failing_val is not None:
                # 40% reduction — conservative, not full survivor match
                adjusted[metric] = failing_val * 0.60

            elif action == "aggressive_reduce" and failing_val is not None:
                # 60% reduction — scale-down scenario
                adjusted[metric] = failing_val * 0.40

            elif action == "improve_partial" and failing_val is not None and survivor_val is not None:
                # Halfway between current and survivor
                adjusted[metric] = (failing_val + survivor_val) / 2.0

            elif action == "normalize" and failing_val is not None:
                # Bring toward zero (e.g. inventory_growth)
                adjusted[metric] = failing_val * 0.30

            elif action == "hold":
                pass  # No change for this metric

        return adjusted

    def _score_feasibility(
        self,
        template: Dict,
    ) -> Tuple[float, str, str]:
        """
        Returns (score 0-1, label, reasoning) based on:
        - Whether macro forces block or enable this scenario
        - Whether company metrics meet prerequisites
        - Whether capital markets are accessible
        """
        score = 0.70  # start optimistic
        reasons = []

        factors = template.get("feasibility_factors", {})

        # Check macro blockers
        blockers_active = [
            f for f in factors.get("blocked_by", [])
            if f in self.active_forces
        ]
        if blockers_active:
            score -= 0.15 * len(blockers_active)
            for b in blockers_active:
                reasons.append(
                    f"Blocked by active macro force: "
                    f"{b.replace('_', ' ')}"
                )

        # Check macro enablers
        enablers_active = [
            f for f in factors.get("enabled_by", [])
            if f in self.active_forces
        ]
        if enablers_active:
            score += 0.10 * len(enablers_active)
            for e in enablers_active:
                reasons.append(
                    f"Supported by: {e.replace('_', ' ')}"
                )

        # Check market access requirement
        if factors.get("requires_market_access"):
            if "credit_tightening" in self.active_forces:
                score -= 0.25
                reasons.append(
                    "Requires capital market access — "
                    "currently blocked by credit tightening"
                )

        # Check macro stress level
        if self.macro_stress_score >= 70:
            score -= 0.10
            reasons.append(
                f"High macro stress ({self.macro_stress_score:.0f}/100) "
                f"reduces execution window"
            )

        score = max(0.05, min(0.95, score))

        # Label
        if score >= 0.60:
            label = "high"
            likelihood = "Plausible"
        elif score >= 0.35:
            label = "medium"
            likelihood = "Difficult"
        else:
            label = "low"
            likelihood = "Unlikely"

        reasoning = "; ".join(reasons) if reasons else (
            "No major macro blockers identified for this strategy"
        )

        return score, label, likelihood, reasoning

    def _build_macro_context(
        self,
        template: Dict,
    ) -> Tuple[List[str], List[str], List[str]]:
        """Returns (enablers, blockers, what_had_to_be_true) for this scenario."""
        factors = template.get("feasibility_factors", {})

        enablers = []
        for f in factors.get("enabled_by", []):
            linkage = MACRO_MICRO_LINKAGE.get(f, {})
            enablers.append(
                f"{f.replace('_',' ').title()}: "
                f"{linkage.get('policy_context', '')}"
            )

        blockers = []
        for f in factors.get("blocked_by", []):
            linkage = MACRO_MICRO_LINKAGE.get(f, {})
            if f in self.active_forces:
                blockers.append(
                    f"ACTIVE — {f.replace('_',' ').title()}: "
                    f"{linkage.get('mechanism', '')}"
                )
            else:
                blockers.append(
                    f"Potential — {f.replace('_',' ').title()}"
                )

        what_had_to_be_true = [
            f"Company prerequisite: {p}"
            for p in factors.get("company_prerequisites", [])
        ]
        if "credit_tightening" not in self.active_forces and factors.get("requires_market_access"):
            what_had_to_be_true.append("Capital markets needed to remain accessible")

        return enablers, blockers, what_had_to_be_true

    def _build_micro_impacts(
        self,
        original: Dict,
        adjusted: Dict,
    ) -> List[str]:
        """Plain-English list of what each metric change means financially."""
        impacts = []
        for metric in ["debt_to_equity", "current_ratio", "cash_burn",
                       "revenue_growth", "operating_margin", "gross_margin"]:
            orig_val = original.get(metric)
            adj_val = adjusted.get(metric)
            if orig_val is None or adj_val is None:
                continue
            if abs(float(orig_val) - float(adj_val)) < 0.001:
                continue
            direction = "improves" if adj_val < orig_val else "increases"
            if metric == "current_ratio":
                direction = "improves" if adj_val > orig_val else "weakens"
            if metric == "revenue_growth":
                direction = "accelerates" if adj_val > orig_val else "slows"

            impacts.append(
                f"{metric.replace('_',' ').title()}: "
                f"{float(orig_val):.3f} → {float(adj_val):.3f} ({direction})"
            )
        return impacts[:4]

    def generate_all_scenarios(self) -> ScenarioBundle:
        """Generate all scenario outcomes and bundle with macro context."""
        original_score, _ = MultiFactorRiskEngine(
            self.failing_metrics, self.macro_stress_score
        ).compute_score()

        outcomes: List[ScenarioOutcome] = []

        for template in self.SCENARIO_TEMPLATES:
            adjusted_metrics = self._apply_scenario_metrics(template)
            adjusted_score, _ = MultiFactorRiskEngine(
                adjusted_metrics, self.macro_stress_score
            ).compute_score()

            risk_reduction = round(original_score - adjusted_score, 2)
            improvement_pct = round(
                (original_score - adjusted_score) / max(original_score, 1e-6) * 100, 2
            )

            feasibility_score, feasibility_label, likelihood, feasibility_reasoning = (
                self._score_feasibility(template)
            )

            enablers, blockers, what_had_to_be_true = (
                self._build_macro_context(template)
            )

            micro_impacts = self._build_micro_impacts(
                self.failing_metrics, adjusted_metrics
            )

            outcomes.append(ScenarioOutcome(
                name=template["name"],
                strategy_description=template["strategy_description"],
                survivor_reference=template["survivor_reference"],
                adjusted_metrics=adjusted_metrics,
                adjusted_score=round(adjusted_score, 2),
                original_score=round(original_score, 2),
                risk_reduction=risk_reduction,
                improvement_pct=improvement_pct,
                feasibility=feasibility_label,
                feasibility_score=feasibility_score,
                feasibility_reasoning=feasibility_reasoning,
                macro_enablers=enablers,
                macro_blockers=blockers,
                micro_impacts=micro_impacts,
                what_had_to_be_true=what_had_to_be_true,
                likelihood_label=likelihood,
            ))

        # Sort by feasibility-weighted impact
        # (high-impact but impossible scenario ranks below medium-impact plausible one)
        outcomes.sort(
            key=lambda x: x.risk_reduction * x.feasibility_score,
            reverse=True
        )

        best = outcomes[0].name if outcomes else "None"

        # Find the one gap that appears most often across scenarios
        all_metrics_changed = []
        for t in self.SCENARIO_TEMPLATES:
            all_metrics_changed.extend(t["target_metrics"].keys())
        from collections import Counter
        consensus_metric = Counter(all_metrics_changed).most_common(1)
        consensus_gap = (
            consensus_metric[0][0].replace("_", " ").title()
            if consensus_metric else "Cash Burn"
        )

        return ScenarioBundle(
            company_ticker=self.failing_metrics.get("_ticker", "unknown"),
            original_score=round(original_score, 2),
            scenarios=outcomes,
            active_macro_forces=self.active_forces,
            macro_micro_chains=self.chains,
            best_scenario=best,
            consensus_gap=f"{consensus_gap} — addressed by all scenarios",
        )
