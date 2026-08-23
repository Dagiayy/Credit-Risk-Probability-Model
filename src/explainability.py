"""SHAP-based per-prediction explanations.

Credit scoring is a regulated domain — the project's own README cites Basel
II's expectation of explainable models. Returning a bare probability with no
justification is not production-usable for a real BNPL underwriting
decision, so every `/predict` response can optionally include the top
features that pushed a customer's score up or down.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from src.logging_utils import get_logger

logger = get_logger(__name__)


class RiskExplainer:
    def __init__(self, model, feature_names: list[str]):
        self.feature_names = feature_names
        try:
            self._explainer = shap.TreeExplainer(model)
            self._kind = "tree"
        except Exception as exc:  # model isn't tree-based (e.g. LogisticRegression)
            logger.warning("Falling back to model-agnostic SHAP explainer: %s", exc)
            self._explainer = shap.Explainer(model.predict_proba, feature_names=feature_names)
            self._kind = "generic"

    def top_factors(self, X_row: pd.DataFrame, top_n: int = 5) -> list[dict]:
        """Return the `top_n` features with the largest |SHAP value| for one row."""
        shap_values = self._explainer.shap_values(X_row)

        # TreeExplainer for a binary classifier can return either a single
        # (n_samples, n_features) array for the positive class, or a list of
        # per-class arrays — normalize to a single (n_features,) vector.
        if isinstance(shap_values, list):
            values = np.asarray(shap_values[-1])[0]
        else:
            values = np.asarray(shap_values)
            if values.ndim == 3:  # (n_samples, n_features, n_classes)
                values = values[0, :, -1]
            else:
                values = values[0]

        order = np.argsort(-np.abs(values))[:top_n]
        return [
            {
                "feature": self.feature_names[i],
                "shap_value": float(values[i]),
                "direction": "increases_risk" if values[i] > 0 else "decreases_risk",
            }
            for i in order
        ]
