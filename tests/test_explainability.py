from sklearn.ensemble import RandomForestClassifier

from src.explainability import RiskExplainer
from src.feature_eng_process import build_preprocessing_pipeline


def test_tree_explainer_top_factors(customer_features_df):
    X = customer_features_df.drop(columns=["CustomerId"])
    pipeline = build_preprocessing_pipeline()
    X_proc = pipeline.fit_transform(X)

    y = [i % 2 for i in range(len(X_proc))]
    model = RandomForestClassifier(n_estimators=20, max_depth=3, random_state=42)
    model.fit(X_proc, y)

    explainer = RiskExplainer(model, feature_names=list(X_proc.columns))
    factors = explainer.top_factors(X_proc.iloc[[0]], top_n=3)

    assert len(factors) == 3
    for factor in factors:
        assert factor["feature"] in X_proc.columns
        assert factor["direction"] in {"increases_risk", "decreases_risk"}
        assert isinstance(factor["shap_value"], float)
