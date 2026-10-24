import joblib
import pandas as pd
import shap
import numpy as np
import matplotlib.pyplot as plt

from src.features import extract_features


def load_model(model_path: str):
    return joblib.load(model_path)


def _get_shap_values(model, X):
    """Return SHAP values as a 2D array: samples x features."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        # Binary classifiers may return one array per class.
        shap_values = shap_values[-1]

    values = np.asarray(shap_values)

    # Handle newer SHAP output: samples x features x classes.
    if values.ndim == 3:
        values = values[:, :, -1]

    # Handle one prediction and one-dimensional output.
    if values.ndim == 1:
        values = values.reshape(1, -1)

    if values.ndim != 2:
        raise ValueError(
            f"Unexpected SHAP values shape: {values.shape}"
        )

    return values


def sample_prediction_explanation(
    model, X, feature_names, top_n=8
):
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=feature_names)

    values = _get_shap_values(model, X)

    # Explain the first URL in X.
    abs_values = pd.Series(
        np.abs(values[0]),
        index=feature_names
    )

    top_features = abs_values.sort_values(
        ascending=False
    ).head(top_n)

    fig, ax = plt.subplots(figsize=(8, 5))
    top_features.sort_values().plot(
        kind="barh",
        color="#2E86DE",
        ax=ax
    )
    ax.set_title("Top Explainable Features for This URL")
    ax.set_xlabel("Absolute SHAP Value")
    fig.tight_layout()

    return fig, top_features


def explain_text(model, row_dict, feature_names):
    X = pd.DataFrame(
        [row_dict],
        columns=feature_names
    )

    values = _get_shap_values(model, X)

    feature_effects = pd.Series(
        values[0],
        index=feature_names
    )

    top = feature_effects.abs().sort_values(
        ascending=False
    ).head(5)

    reasons = []

    for feature in top.index:
        score = feature_effects[feature]
        direction = (
            "increases"
            if score > 0
            else "decreases"
        )
        reasons.append(
            f"{feature} {direction} the model score"
        )

    return "Prediction rationale: " + "; ".join(reasons) + "."