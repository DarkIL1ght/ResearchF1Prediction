from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier

RANDOM_STATE = 42
MODEL_RESULTS_PATH = "../figures/model_results.csv"


def get_positive_class_weight(y_train: pd.Series) -> float:
    """Returns scale_pos_weight for XGBoost."""
    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())

    if positive_count == 0:
        return 1.0

    return negative_count / positive_count


def build_models(y_train: pd.Series) -> Dict[str, object]:
    """Creates all models with fixed random_state."""
    scale_pos_weight = get_positive_class_weight(y_train)

    return {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced",
            max_iter=3000,
            random_state=RANDOM_STATE,
            solver="lbfgs",
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
    }


def calculate_race_winner_accuracy(
    probabilities: np.ndarray,
    val_metadata: pd.DataFrame,
) -> Tuple[float, int]:
    """Checks whether the top predicted driver per race is the real winner."""
    if val_metadata is None or val_metadata.empty:
        return np.nan, 0

    evaluation_df = val_metadata.reset_index(drop=True).copy()
    evaluation_df["PredictedWinProbability"] = probabilities

    correct_predictions = 0
    races_evaluated = 0

    for _, race_df in evaluation_df.groupby(["Year", "Round"], sort=False):
        if race_df["Winner"].sum() == 0:
            continue

        predicted_winner = race_df.loc[race_df["PredictedWinProbability"].idxmax()]
        correct_predictions += int(predicted_winner["Winner"] == 1)
        races_evaluated += 1

    if races_evaluated == 0:
        return np.nan, 0

    return correct_predictions / races_evaluated, races_evaluated


def evaluate_model(
    model,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    val_metadata: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    """Calculates row-level metrics and optional race-level winner accuracy."""
    y_pred = model.predict(X_val)

    metrics = {
        "Accuracy": accuracy_score(y_val, y_pred),
        "Precision": precision_score(y_val, y_pred, zero_division=0),
        "Recall": recall_score(y_val, y_pred, zero_division=0),
        "F1": f1_score(y_val, y_pred, zero_division=0),
    }

    if val_metadata is not None and hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_val)[:, 1]
        race_accuracy, races_evaluated = calculate_race_winner_accuracy(
            probabilities,
            val_metadata,
        )
        metrics["RaceWinnerAccuracy"] = race_accuracy
        metrics["RacesEvaluated"] = races_evaluated

    return metrics


def train_and_evaluate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    val_metadata: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Trains models and returns metrics plus fitted estimators."""
    print("=" * 70)
    print("Training and evaluating models")
    print("=" * 70)

    print(f"Train shape: {X_train.shape}")
    print(f"Validation shape: {X_val.shape}")
    print(f"Train winner balance: {y_train.mean():.2%}")
    print(f"Validation winner balance: {y_val.mean():.2%}")

    models = build_models(y_train)
    trained_models = {}
    results = []

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")

        model.fit(X_train, y_train)
        trained_models[model_name] = model

        metrics = evaluate_model(model, X_val, y_val, val_metadata=val_metadata)
        results.append({"Model": model_name, **metrics})

        print(f"  Accuracy : {metrics['Accuracy']:.4f}")
        print(f"  Precision: {metrics['Precision']:.4f}")
        print(f"  Recall   : {metrics['Recall']:.4f}")
        print(f"  F1       : {metrics['F1']:.4f}")

        if "RaceWinnerAccuracy" in metrics:
            print(
                f"  Race winner accuracy: "
                f"{metrics['RaceWinnerAccuracy']:.4f} "
                f"({int(metrics['RacesEvaluated'])} races)"
            )

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X_val)[:, 1]
            print(
                f"  Probability range: "
                f"{probabilities.min():.4f} - {probabilities.max():.4f}"
            )

    results_df = (
        pd.DataFrame(results)
        .sort_values("F1", ascending=False)
        .reset_index(drop=True)
    )
    results_df.to_csv(MODEL_RESULTS_PATH, index=False)

    print("\n" + "=" * 70)
    print("Model comparison by F1")
    print("=" * 70)
    print(results_df.to_string(index=False))
    print(f"\nModel results saved to: {MODEL_RESULTS_PATH}")

    return results_df, trained_models


if __name__ == "__main__":
    from preprocess import preprocess_data

    X_train_, y_train_, X_val_, y_val_, _, _, val_metadata_ = preprocess_data(
        return_metadata=True
    )
    train_and_evaluate_models(
        X_train_,
        y_train_,
        X_val_,
        y_val_,
        val_metadata=val_metadata_,
    )
