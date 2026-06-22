from typing import Dict, Tuple

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

RANDOM_STATE = 42

DRIVER_LEVEL_FEATURES = [
    "QualifyingPosition",
    "HistoricalWinRate",
    "AvgFinishPosition",
]


def safe_improvement(new_score: float, old_score: float) -> float:
    """Считает относительное улучшение в процентах, безопасно для old_score=0."""
    if old_score == 0:
        return 100.0 if new_score > 0 else 0.0

    return ((new_score - old_score) / old_score) * 100


def get_model_f1_scores(
    trained_models: Dict[str, object],
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Dict[str, float]:
    """Возвращает F1 для каждой обученной модели."""
    scores = {}

    for model_name, model in trained_models.items():
        y_pred = model.predict(X_val)
        scores[model_name] = f1_score(y_val, y_pred, zero_division=0)

    return scores


def test_hypotheses(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    trained_models: Dict[str, object],
) -> Tuple[float, float]:
    """
    Проверяет три гипотезы:

    H1: любая ML модель > DummyClassifier most_frequent по F1.
    H2: модель со всеми признаками > модель только с driver-level признаками.
    H3: лучшая tree-based модель RF/XGB/LGB > LogisticRegression.
    """
    print("\n" + "=" * 70)
    print("Hypothesis testing")
    print("=" * 70)

    model_scores = get_model_f1_scores(trained_models, X_val, y_val)

    best_model_name = max(model_scores, key=model_scores.get)
    best_f1 = model_scores[best_model_name]

    print("\n[H1] Any ML model > DummyClassifier by F1")

    dummy = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    dummy.fit(X_train, y_train)

    dummy_predictions = dummy.predict(X_val)
    dummy_f1 = f1_score(y_val, dummy_predictions, zero_division=0)

    h1_supported = best_f1 > dummy_f1

    print(f"  DummyClassifier F1: {dummy_f1:.4f}")
    print(f"  Best ML model: {best_model_name}")
    print(f"  Best ML F1: {best_f1:.4f}")
    print(f"  Improvement: {safe_improvement(best_f1, dummy_f1):.2f}%")
    print(f"  H1: {'SUPPORTED' if h1_supported else 'NOT SUPPORTED'}")

    print("\n[H2] Full-feature model > driver-level-only model by F1")

    missing_driver_features = [feature for feature in DRIVER_LEVEL_FEATURES if feature not in X_train.columns]
    if missing_driver_features:
        raise ValueError(f"Missing driver-level features: {missing_driver_features}")

    X_train_driver = X_train[DRIVER_LEVEL_FEATURES]
    X_val_driver = X_val[DRIVER_LEVEL_FEATURES]

    driver_only_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    driver_only_model.fit(X_train_driver, y_train)
    driver_only_predictions = driver_only_model.predict(X_val_driver)
    driver_only_f1 = f1_score(y_val, driver_only_predictions, zero_division=0)

    if "RandomForest" not in trained_models:
        raise ValueError("RandomForest model is required for H2 testing.")

    full_model_predictions = trained_models["RandomForest"].predict(X_val)
    full_model_f1 = f1_score(y_val, full_model_predictions, zero_division=0)

    h2_supported = full_model_f1 > driver_only_f1

    print(f"  Driver-only RandomForest F1: {driver_only_f1:.4f}")
    print(f"  Full-feature RandomForest F1: {full_model_f1:.4f}")
    print(f"  Improvement: {safe_improvement(full_model_f1, driver_only_f1):.2f}%")
    print(f"  H2: {'SUPPORTED' if h2_supported else 'NOT SUPPORTED'}")

    print("\n[H3] Best tree-based model > LogisticRegression by F1")

    tree_based_models = ["RandomForest", "XGBoost", "LightGBM"]

    if "LogisticRegression" not in model_scores:
        raise ValueError("LogisticRegression model is required for H3 testing.")

    available_tree_scores = {
        model_name: model_scores[model_name]
        for model_name in tree_based_models
        if model_name in model_scores
    }

    if not available_tree_scores:
        raise ValueError("At least one tree-based model is required for H3 testing.")

    best_tree_model_name = max(available_tree_scores, key=available_tree_scores.get)
    best_tree_f1 = available_tree_scores[best_tree_model_name]
    logistic_f1 = model_scores["LogisticRegression"]

    h3_supported = best_tree_f1 > logistic_f1

    for model_name, score in model_scores.items():
        print(f"  {model_name} F1: {score:.4f}")

    print(f"  Best tree-based model: {best_tree_model_name}")
    print(f"  Best tree-based F1: {best_tree_f1:.4f}")
    print(f"  LogisticRegression F1: {logistic_f1:.4f}")
    print(f"  Improvement: {safe_improvement(best_tree_f1, logistic_f1):.2f}%")
    print(f"  H3: {'SUPPORTED' if h3_supported else 'NOT SUPPORTED'}")

    print("\nHypothesis summary")
    print(f"  H1: {'SUPPORTED' if h1_supported else 'NOT SUPPORTED'}")
    print(f"  H2: {'SUPPORTED' if h2_supported else 'NOT SUPPORTED'}")
    print(f"  H3: {'SUPPORTED' if h3_supported else 'NOT SUPPORTED'}")

    return dummy_f1, best_f1


if __name__ == "__main__":
    from preprocess import preprocess_data
    from train_models import train_and_evaluate_models

    X_train_, y_train_, X_val_, y_val_, _ = preprocess_data()
    _, trained_models_ = train_and_evaluate_models(X_train_, y_train_, X_val_, y_val_)
    test_hypotheses(X_train_, y_train_, X_val_, y_val_, trained_models_)