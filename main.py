import argparse
import os
import random

import numpy as np

from collect_data import main as collect_data
from preprocess import preprocess_data
from shap_analysis import perform_shap_analysis
from test_hypotheses import test_hypotheses
from train_models import train_and_evaluate_models

RANDOM_STATE = 42


def set_global_seed(seed: int = RANDOM_STATE) -> None:
    """Фиксирует random_state для воспроизводимости."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def run_full_pipeline(force_reload_data: bool = False, resume_data: bool = True) -> None:
    """Запускает полный ML-пайплайн проекта."""
    set_global_seed()

    print("=" * 70)
    print("F1 RACE WINNER PREDICTION PIPELINE")
    print("=" * 70)

    print("\n[Step 1] Collecting raw FastF1 data")
    df = collect_data(force_reload=force_reload_data, resume=resume_data)

    print("\n[Step 2] Preprocessing data")
    (
        X_train,
        y_train,
        X_val,
        y_val,
        scaler,
        train_metadata,
        val_metadata,
    ) = preprocess_data(df, return_metadata=True)

    print("\n[Step 3] Training and evaluating models")
    results_df, trained_models = train_and_evaluate_models(
        X_train,
        y_train,
        X_val,
        y_val,
        val_metadata=val_metadata,
    )

    print("\n[Step 4] Testing hypotheses")
    dummy_f1, best_f1 = test_hypotheses(
        X_train,
        y_train,
        X_val,
        y_val,
        trained_models,
    )

    print("\n[Step 5] Running SHAP analysis")
    shap_importance = perform_shap_analysis(
        trained_models=trained_models,
        X_train=X_train,
        X_val=X_val,
        y_val=y_val,
    )

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(f"Train size: {X_train.shape}")
    print(f"Validation size: {X_val.shape}")
    print(f"Train winner balance: {y_train.mean():.2%}")
    print(f"Validation winner balance: {y_val.mean():.2%}")

    print("\nModel comparison by F1:")
    summary_columns = ["Model", "F1", "Accuracy", "Precision", "Recall"]
    if "RaceWinnerAccuracy" in results_df.columns:
        summary_columns.append("RaceWinnerAccuracy")
    print(results_df[summary_columns].to_string(index=False))

    print("\nHypothesis baseline summary:")
    print(f"DummyClassifier F1: {dummy_f1:.4f}")
    print(f"Best ML model F1: {best_f1:.4f}")

    print("\nTop-5 SHAP features:")
    print(shap_importance.head(5).to_string(index=False))

    print("\nSaved files:")
    print("  Raw dataset: f1_race_data.csv")
    print("  Processed dataset: f1_processed_data.csv")
    print("  Model results: model_results.csv")
    print("  SHAP summary plot: shap_summary_plot.png")
    print("  SHAP bar plot: shap_feature_importance_bar.png")
    print("  SHAP importance CSV: shap_feature_importance.csv")

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the full F1 winner prediction pipeline."
    )
    parser.add_argument(
        "--force-reload-data",
        action="store_true",
        help="Ignore existing CSV files and collect data from scratch.",
    )
    parser.add_argument(
        "--no-resume-data",
        action="store_true",
        help="Do not continue from existing partial/raw data.",
    )
    args = parser.parse_args()

    run_full_pipeline(
        force_reload_data=args.force_reload_data,
        resume_data=not args.no_resume_data,
    )
