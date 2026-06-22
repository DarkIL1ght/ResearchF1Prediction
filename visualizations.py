from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    confusion_matrix,
    f1_score,
    roc_curve,
)

from preprocess import FEATURE_COLUMNS, preprocess_data
from train_models import (
    build_models,
    calculate_race_winner_accuracy,
    train_and_evaluate_models,
)

RAW_DATA_PATH = "../figures/f1_race_data.csv"
OUTPUT_DIR = Path("../figures")


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)


def save_target_distribution(raw_df: pd.DataFrame, y_train: pd.Series, y_val: pd.Series) -> None:
    """Saves target distribution charts for overall/train/validation data."""
    overall = raw_df["Winner"].astype(int)
    distributions = {
        "Overall": overall,
        "Train 2014-2023": y_train,
        "Validation 2024": y_val,
    }

    labels = ["Non-winner", "Winner"]
    x = np.arange(len(distributions))
    width = 0.35

    non_winner_rates = []
    winner_rates = []
    winner_counts = []
    total_counts = []

    for values in distributions.values():
        total = len(values)
        winners = int(values.sum())
        non_winners = total - winners
        non_winner_rates.append(non_winners / total)
        winner_rates.append(winners / total)
        winner_counts.append(winners)
        total_counts.append(total)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, non_winner_rates, width, label=labels[0], color="#4c78a8")
    ax.bar(x + width / 2, winner_rates, width, label=labels[1], color="#f58518")

    for index, rate in enumerate(winner_rates):
        ax.text(
            index + width / 2,
            rate + 0.01,
            f"{rate:.2%}\n{winner_counts[index]}/{total_counts[index]}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(distributions.keys())
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Class share")
    ax.set_title("Target Distribution: Winner vs Non-winner")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "target_distribution.png", dpi=200)
    plt.close(fig)


def save_confusion_matrix(model, X_val: pd.DataFrame, y_val: pd.Series, model_name: str) -> None:
    y_pred = model.predict(X_val)
    matrix = confusion_matrix(y_val, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(6, 5))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Non-winner", "Winner"],
    )
    display.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title(f"Confusion Matrix: {model_name}")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"confusion_matrix_{model_name.lower()}.png", dpi=200)
    plt.close(fig)


def save_roc_curve(model, X_val: pd.DataFrame, y_val: pd.Series, model_name: str) -> float:
    probabilities = model.predict_proba(X_val)[:, 1]
    false_positive_rate, true_positive_rate, _ = roc_curve(y_val, probabilities)
    roc_auc = auc(false_positive_rate, true_positive_rate)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(
        false_positive_rate,
        true_positive_rate,
        color="#e45756",
        linewidth=2,
        label=f"{model_name} AUC = {roc_auc:.3f}",
    )
    ax.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve: {model_name}")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"roc_curve_{model_name.lower()}.png", dpi=200)
    plt.close(fig)

    return roc_auc


def get_feature_importance(model, feature_names) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_[0])
    else:
        raise ValueError("Model does not expose feature importance.")

    return (
        pd.DataFrame({"Feature": feature_names, "Importance": importance})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )


def save_feature_importance(model, feature_names, model_name: str) -> pd.DataFrame:
    importance_df = get_feature_importance(model, feature_names)

    fig, ax = plt.subplots(figsize=(9, 5))
    plot_df = importance_df.sort_values("Importance", ascending=True)
    ax.barh(plot_df["Feature"], plot_df["Importance"], color="#54a24b")
    ax.set_xlabel("Built-in feature importance")
    ax.set_title(f"Feature Importance: {model_name}")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"feature_importance_{model_name.lower()}.png", dpi=200)
    plt.close(fig)

    importance_df.to_csv(
        OUTPUT_DIR / f"feature_importance_{model_name.lower()}.csv",
        index=False,
    )
    return importance_df


def get_race_predictions(
    trained_models: Dict[str, object],
    X_val: pd.DataFrame,
    val_metadata: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    summary_rows = []

    for model_name, model in trained_models.items():
        probabilities = model.predict_proba(X_val)[:, 1]
        race_accuracy, races_evaluated = calculate_race_winner_accuracy(
            probabilities,
            val_metadata,
        )
        summary_rows.append(
            {
                "Model": model_name,
                "RaceWinnerAccuracy": race_accuracy,
                "RacesEvaluated": races_evaluated,
            }
        )

        eval_df = val_metadata.reset_index(drop=True).copy()
        eval_df["PredictedWinProbability"] = probabilities

        for (year, round_number), race_df in eval_df.groupby(["Year", "Round"], sort=True):
            predicted = race_df.loc[race_df["PredictedWinProbability"].idxmax()]
            actual = race_df.loc[race_df["Winner"].idxmax()]
            rows.append(
                {
                    "Model": model_name,
                    "Year": int(year),
                    "Round": int(round_number),
                    "Race": actual["Race"],
                    "ActualWinner": actual["Driver"],
                    "PredictedWinner": predicted["Driver"],
                    "Correct": int(predicted["Driver"] == actual["Driver"]),
                    "PredictedWinProbability": predicted["PredictedWinProbability"],
                }
            )

    predictions_df = pd.DataFrame(rows)
    summary_df = pd.DataFrame(summary_rows).sort_values(
        "RaceWinnerAccuracy",
        ascending=False,
    )

    predictions_df.to_csv(OUTPUT_DIR / "race_winner_predictions_2024.csv", index=False)
    summary_df.to_csv(OUTPUT_DIR / "race_winner_accuracy_by_model.csv", index=False)

    return predictions_df, summary_df


def save_race_winner_timeline(predictions_df: pd.DataFrame, model_name: str) -> None:
    model_df = predictions_df[predictions_df["Model"] == model_name].sort_values("Round")
    colors = np.where(model_df["Correct"] == 1, "#54a24b", "#e45756")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(model_df["Round"], model_df["Correct"], color=colors)
    ax.set_xticks(model_df["Round"])
    ax.set_ylim(0, 1.15)
    ax.set_xlabel("2024 race round")
    ax.set_ylabel("Correct winner prediction")
    ax.set_title(f"Race-level Winner Prediction by Round: {model_name}")
    ax.grid(axis="y", alpha=0.25)

    cumulative_accuracy = model_df["Correct"].expanding().mean()
    ax2 = ax.twinx()
    ax2.plot(
        model_df["Round"],
        cumulative_accuracy,
        color="#4c78a8",
        marker="o",
        linewidth=2,
        label="Cumulative accuracy",
    )
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Cumulative accuracy")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"race_winner_timeline_{model_name.lower()}.png", dpi=200)
    plt.close(fig)


def save_race_accuracy_comparison(summary_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        summary_df["Model"],
        summary_df["RaceWinnerAccuracy"],
        color=["#4c78a8", "#f58518", "#54a24b", "#e45756"][: len(summary_df)],
    )

    for index, row in summary_df.reset_index(drop=True).iterrows():
        correct = int(round(row["RaceWinnerAccuracy"] * row["RacesEvaluated"]))
        ax.text(
            index,
            row["RaceWinnerAccuracy"] + 0.015,
            f"{row['RaceWinnerAccuracy']:.3f}\n{correct}/{int(row['RacesEvaluated'])}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylim(0, 0.55)
    ax.set_ylabel("RaceWinnerAccuracy")
    ax.set_title("Race-level Winner Accuracy by Model")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "race_winner_accuracy_by_model.png", dpi=200)
    plt.close(fig)


def save_temporal_learning_curve(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    train_metadata: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    val_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Trains XGBoost on increasing chronological train fractions."""
    fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
    train_order = train_metadata.sort_values(["Year", "Round"]).index.to_numpy()
    rows = []

    for fraction in fractions:
        sample_size = max(50, int(len(train_order) * fraction))
        selected_index = train_order[:sample_size]

        X_subset = X_train.loc[selected_index]
        y_subset = y_train.loc[selected_index]

        if y_subset.nunique() < 2:
            continue

        model = build_models(y_subset)["XGBoost"]
        model.fit(X_subset, y_subset)

        train_predictions = model.predict(X_subset)
        val_predictions = model.predict(X_val)
        val_probabilities = model.predict_proba(X_val)[:, 1]
        race_accuracy, races_evaluated = calculate_race_winner_accuracy(
            val_probabilities,
            val_metadata,
        )

        rows.append(
            {
                "TrainFraction": fraction,
                "TrainRows": len(X_subset),
                "TrainF1": f1_score(y_subset, train_predictions, zero_division=0),
                "ValidationF1": f1_score(y_val, val_predictions, zero_division=0),
                "RaceWinnerAccuracy": race_accuracy,
                "RacesEvaluated": races_evaluated,
            }
        )

    curve_df = pd.DataFrame(rows)
    curve_df.to_csv(OUTPUT_DIR / "temporal_learning_curve_xgboost.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(curve_df["TrainRows"], curve_df["TrainF1"], marker="o", label="Train F1")
    ax.plot(
        curve_df["TrainRows"],
        curve_df["ValidationF1"],
        marker="o",
        label="Validation F1",
    )
    ax.plot(
        curve_df["TrainRows"],
        curve_df["RaceWinnerAccuracy"],
        marker="o",
        label="RaceWinnerAccuracy",
    )
    ax.set_xlabel("Chronological training rows")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Temporal Learning Curve: XGBoost")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "temporal_learning_curve_xgboost.png", dpi=200)
    plt.close(fig)

    return curve_df


def main() -> None:
    ensure_output_dir()

    raw_df = pd.read_csv(RAW_DATA_PATH)
    (
        X_train,
        y_train,
        X_val,
        y_val,
        _,
        train_metadata,
        val_metadata,
    ) = preprocess_data(raw_df, return_metadata=True)

    results_df, trained_models = train_and_evaluate_models(
        X_train,
        y_train,
        X_val,
        y_val,
        val_metadata=val_metadata,
    )

    best_f1_model_name = results_df.iloc[0]["Model"]
    best_f1_model = trained_models[best_f1_model_name]

    save_target_distribution(raw_df, y_train, y_val)
    save_confusion_matrix(best_f1_model, X_val, y_val, best_f1_model_name)
    roc_auc = save_roc_curve(best_f1_model, X_val, y_val, best_f1_model_name)
    save_feature_importance(best_f1_model, FEATURE_COLUMNS, best_f1_model_name)

    predictions_df, race_summary_df = get_race_predictions(
        trained_models,
        X_val,
        val_metadata,
    )
    best_race_model_name = race_summary_df.iloc[0]["Model"]
    save_race_winner_timeline(predictions_df, best_race_model_name)
    save_race_accuracy_comparison(race_summary_df)
    save_temporal_learning_curve(
        X_train,
        y_train,
        train_metadata,
        X_val,
        y_val,
        val_metadata,
    )

    print("\nSaved visualization files to:", OUTPUT_DIR)
    print(f"ROC AUC ({best_f1_model_name}): {roc_auc:.4f}")
    print("Best row-level F1 model:", best_f1_model_name)
    print("Best race-level model:", best_race_model_name)
    print(race_summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
