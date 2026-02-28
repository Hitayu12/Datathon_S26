"""Local analyst model: trains a lightweight classifier for distress reasoning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "debt_to_equity",
    "current_ratio",
    "burn_ratio",
    "revenue_growth",
    "macro_stress",
    "qualitative_intensity",
]


@dataclass
class LocalReasoningResult:
    risk_probability: float
    label: str
    top_drivers: List[str]
    feature_values: Dict[str, float]


class LocalAnalystModel:
    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=500, random_state=random_state)),
            ]
        )
        self._is_fit = False

    def train(self, n_samples: int = 6000) -> None:
        rng = np.random.default_rng(self.random_state)

        debt_to_equity = rng.uniform(0.0, 6.0, n_samples)
        current_ratio = rng.uniform(0.3, 3.0, n_samples)
        burn_ratio = rng.uniform(0.0, 0.9, n_samples)
        revenue_growth = rng.uniform(-0.6, 0.4, n_samples)
        macro_stress = rng.uniform(15, 100, n_samples)
        qualitative_intensity = rng.integers(0, 7, n_samples)

        # Synthetic supervisory signal to emulate distressed-vs-survivor outcomes.
        latent = (
            1.25 * debt_to_equity
            - 1.10 * current_ratio
            + 1.45 * burn_ratio
            - 2.00 * revenue_growth
            + 0.022 * macro_stress
            + 0.35 * qualitative_intensity
            - 2.4
        )
        noise = rng.normal(0.0, 0.65, n_samples)
        probs = 1.0 / (1.0 + np.exp(-(latent + noise)))
        labels = (probs > 0.5).astype(int)

        x = np.column_stack(
            [
                debt_to_equity,
                current_ratio,
                burn_ratio,
                revenue_growth,
                macro_stress,
                qualitative_intensity,
            ]
        )

        self.pipeline.fit(x, labels)
        self._is_fit = True

    def _vectorize(
        self,
        metrics: Dict[str, Optional[float]],
        macro_stress_score: float,
        qualitative_intensity: float,
    ) -> np.ndarray:
        debt_to_equity = float(metrics.get("debt_to_equity") or 1.6)
        current_ratio = float(metrics.get("current_ratio") or 1.1)
        cash_burn = float(metrics.get("cash_burn") or 0.0)
        revenue = abs(float(metrics.get("revenue") or 1_000_000_000.0))
        burn_ratio = min(1.0, max(0.0, cash_burn / revenue)) if revenue > 0 else 0.5
        revenue_growth = float(metrics.get("revenue_growth") or 0.0)
        macro_stress = float(macro_stress_score)

        return np.array(
            [[debt_to_equity, current_ratio, burn_ratio, revenue_growth, macro_stress, float(qualitative_intensity)]],
            dtype=float,
        )

    def predict(
        self,
        metrics: Dict[str, Optional[float]],
        macro_stress_score: float,
        qualitative_intensity: float,
    ) -> LocalReasoningResult:
        if not self._is_fit:
            self.train()

        x = self._vectorize(metrics, macro_stress_score, qualitative_intensity)
        prob = float(self.pipeline.predict_proba(x)[0, 1])
        label = "High Distress" if prob >= 0.6 else "Moderate Distress" if prob >= 0.4 else "Lower Distress"

        scaler: StandardScaler = self.pipeline.named_steps["scaler"]
        clf: LogisticRegression = self.pipeline.named_steps["clf"]
        z = scaler.transform(x)[0]
        contributions = z * clf.coef_[0]

        ranked_idx = np.argsort(np.abs(contributions))[::-1]
        driver_map = {
            "debt_to_equity": "Leverage pressure",
            "current_ratio": "Liquidity cushion",
            "burn_ratio": "Cash burn intensity",
            "revenue_growth": "Revenue momentum",
            "macro_stress": "Macro pressure",
            "qualitative_intensity": "Distress language intensity",
        }
        top_drivers: List[str] = []
        for idx in ranked_idx[:3]:
            feat = FEATURES[idx]
            sign = "increasing" if contributions[idx] > 0 else "reducing"
            top_drivers.append(f"{driver_map[feat]} is {sign} risk.")

        feature_values = {
            "debt_to_equity": float(x[0, 0]),
            "current_ratio": float(x[0, 1]),
            "burn_ratio": float(x[0, 2]),
            "revenue_growth": float(x[0, 3]),
            "macro_stress": float(x[0, 4]),
            "qualitative_intensity": float(x[0, 5]),
        }

        return LocalReasoningResult(
            risk_probability=prob,
            label=label,
            top_drivers=top_drivers,
            feature_values=feature_values,
        )
