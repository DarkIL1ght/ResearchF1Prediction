from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import f1_score

RANDOM_STATE = 42
SHAP_SUMMARY_PATH = "../figures/shap_summary_plot.png"
SHAP_BAR_PATH = "../figures/shap_feature_importance_bar.png"
SHAP_IMPORTANCE_PATH = "../figures/shap_feature_importance.csv"


def choose_best_tree_model(
    trained_models: Dict[str, object],
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Tuple[str, object]:
    """
    Выбирает лучшую tree-based модель для SHAP.
    Приоритет: RandomForest, XGBoost.
    LightGBM тоже поддерживается, если она лучшая из доступных.
    """
    candidate_names = ["RandomForest", "XGBoost", "LightGBM"]
    candidates = {
        name: model
        for name, model in trained_models.items()
        if name in candidate_names
    }

    if not candidates:
        raise ValueError("No tree-based models found for SHAP analysis.")

    scores = {}

    for model_name, model in candidates.items():
        predictions = model.predict(X_val)
        scores[model_name] = f1_score(y_val, predictions, zero_division=0)

    best_model_name = max(scores, key=scores.get)
    return best_model_name, candidates[best_model_name]


def normalize_shap_values(shap_values):
    """
    Приводит SHAP values к матрице для положительного класса.
    Разные версии SHAP возвращают разные форматы:
    - list[class_0, class_1]
    - array shape=(n_samples, n_features)
    - array shape=(n_samples, n_features, n_classes)
    """
    if isinstance(shap_values, list):
        if len(shap_values) > 1:
            return shap_values[1]
        return shap_values[0]

    shap_array = np.array(shap_values)

    if shap_array.ndim == 3:
        return shap_array[:, :, 1]

    return shap_array


def perform_shap_analysis(
    trained_models: Dict[str, object],
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: Optional[pd.Series] = None,
    max_samples: int = 100,
) -> pd.DataFrame:
    """
    Для лучшей tree-based модели строит:
    - SHAP summary plot
    - SHAP bar plot

    Сохраняет:
    - shap_summary_plot.png
    - shap_feature_importance_bar.png
    - shap_feature_importance.csv

    Выводит топ-5 важных признаков.
    """
    print("\n" + "=" * 70)
    print("SHAP analysis")
    print("=" * 70)

    if y_val is not None:
        model_name, model = choose_best_tree_model(trained_models, X_val, y_val)
    else:
        if "RandomForest" in trained_models:
            model_name = "RandomForest"
            model = trained_models["RandomForest"]
        elif "XGBoost" in trained_models:
            model_name = "XGBoost"
            model = trained_models["XGBoost"]
        elif "LightGBM" in trained_models:
            model_name = "LightGBM"
            model = trained_models["LightGBM"]
        else:
            raise ValueError("No supported model found for SHAP analysis.")

    print(f"Selected model for SHAP: {model_name}")

    if len(X_val) == 0:
        raise ValueError("X_val is empty. Cannot run SHAP analysis.")

    sample_size = min(max_samples, len(X_val))
    X_sample = X_val.sample(sample_size, random_state=RANDOM_STATE)

    explainer = shap.TreeExplainer(model)
    shap_values_raw = explainer.shap_values(X_sample)
    shap_values = normalize_shap_values(shap_values_raw)

    plt.figure(figsize=(12, 7))
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=X_sample.columns.tolist(),
        show=False,
    )
    plt.tight_layout()
    plt.savefig(SHAP_SUMMARY_PATH, dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(12, 7))
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=X_sample.columns.tolist(),
        plot_type="bar",
        show=False,
    )
    plt.tight_layout()
    plt.savefig(SHAP_BAR_PATH, dpi=200, bbox_inches="tight")
    plt.close()

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    feature_importance = pd.DataFrame(
        {
            "Feature": X_sample.columns,
            "MeanAbsoluteSHAP": mean_abs_shap,
        }
    ).sort_values("MeanAbsoluteSHAP", ascending=False).reset_index(drop=True)

    feature_importance.to_csv(SHAP_IMPORTANCE_PATH, index=False)

    print("\nTop-5 important features by SHAP:")
    for _, row in feature_importance.head(5).iterrows():
        print(f"  {row['Feature']}: {row['MeanAbsoluteSHAP']:.6f}")

    print("\nSaved SHAP outputs:")
    print(f"  Summary plot: {SHAP_SUMMARY_PATH}")
    print(f"  Bar plot: {SHAP_BAR_PATH}")
    print(f"  Feature importance CSV: {SHAP_IMPORTANCE_PATH}")

    return feature_importance


if __name__ == "__main__":
    from preprocess import preprocess_data
    from train_models import train_and_evaluate_models

    X_train_, y_train_, X_val_, y_val_, _ = preprocess_data()
    _, trained_models_ = train_and_evaluate_models(X_train_, y_train_, X_val_, y_val_)
    perform_shap_analysis(trained_models_, X_train_, X_val_, y_val_)