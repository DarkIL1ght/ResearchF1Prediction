from typing import Tuple

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

RANDOM_STATE = 42
RAW_DATA_PATH = "../figures/f1_race_data.csv"
PROCESSED_DATA_PATH = "../figures/f1_processed_data.csv"

NUMERIC_FEATURES = [
    "QualifyingPosition",
    "HistoricalWinRate",
    "AvgFinishPosition",
    "ConstructorWinRate",
    "PrevSeasonConstructorRank",
]

WINSORIZE_FEATURES = [
    "QualifyingPosition",
    "HistoricalWinRate",
    "AvgFinishPosition",
    "ConstructorWinRate",
]

CATEGORICAL_FEATURES = [
    "CircuitType",
    "Weather",
]

CATEGORY_VALUES = {
    "CircuitType": ["Permanent", "Street"],
    "Weather": ["Dry", "Wet"],
}

FEATURE_COLUMNS = [
    "QualifyingPosition",
    "HistoricalWinRate",
    "AvgFinishPosition",
    "ConstructorWinRate",
    "PrevSeasonConstructorRank",
    "CircuitTypeEncoded",
    "WeatherEncoded",
    "StartPositionNormalized",
    "ConstructorStrength",
]

METADATA_COLUMNS = [
    "Year",
    "Round",
    "Race",
    "Driver",
    "DriverName",
    "Constructor",
    "FinalPosition",
    "Winner",
]


def winsorize_with_iqr(df: pd.DataFrame, columns: list, multiplier: float = 1.5) -> pd.DataFrame:
    """
    Обрабатывает выбросы через IQR + winsorizing:
    значения ниже/выше границ обрезаются до границ.
    """
    df = df.copy()

    for column in columns:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1

        if iqr == 0 or pd.isna(iqr):
            continue

        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr

        outliers_count = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()

        if outliers_count > 0:
            print(f"  Winsorizing {outliers_count} outliers in {column}")
            df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)

    return df


def fit_iqr_bounds(df: pd.DataFrame, columns: list, multiplier: float = 1.5) -> dict:
    """Fits IQR clipping bounds on train data only."""
    bounds = {}

    for column in columns:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1

        if iqr == 0 or pd.isna(iqr):
            continue

        bounds[column] = (q1 - multiplier * iqr, q3 + multiplier * iqr)

    return bounds


def apply_iqr_winsorizing(df: pd.DataFrame, bounds: dict, label: str) -> pd.DataFrame:
    """Applies pre-fitted IQR clipping bounds."""
    df = df.copy()

    for column, (lower_bound, upper_bound) in bounds.items():
        outliers_count = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()

        if outliers_count > 0:
            print(f"  Winsorizing {outliers_count} outliers in {column} ({label})")
            df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)

    return df


def add_interactive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds derived features after numeric scaling."""
    df = df.copy()
    df["StartPositionNormalized"] = 1 - df["QualifyingPosition"]
    df["ConstructorStrength"] = (
        df["ConstructorWinRate"] * (1 - df["PrevSeasonConstructorRank"])
    )
    return df


def preprocess_data(
    df: pd.DataFrame = None,
    return_metadata: bool = False,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, MinMaxScaler]:
    """
    Полная предобработка:
    - читает CSV, если df не передан;
    - обрабатывает пропуски;
    - кодирует категориальные признаки через LabelEncoder;
    - удаляет выбросы через IQR + winsorizing;
    - нормализует числовые признаки MinMaxScaler;
    - добавляет интерактивные признаки;
    - делит данные: train <= 2023, validation = 2024.
    """
    print("=" * 70)
    print("Preprocessing data")
    print("=" * 70)

    if df is None:
        df = pd.read_csv(RAW_DATA_PATH)

    df_processed = df.copy()

    required_columns = [
        "Year",
        "Round",
        "Winner",
        "QualifyingPosition",
        "FinalPosition",
        "HistoricalWinRate",
        "AvgFinishPosition",
        "ConstructorWinRate",
        "PrevSeasonConstructorRank",
        "CircuitType",
        "Weather",
        "IsDNS",
    ]

    missing_columns = [column for column in required_columns if column not in df_processed.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df_processed = df_processed[df_processed["IsDNS"] == False].copy()

    df_processed["QualifyingPosition"] = pd.to_numeric(
        df_processed["QualifyingPosition"], errors="coerce"
    ).fillna(20)

    df_processed["FinalPosition"] = pd.to_numeric(
        df_processed["FinalPosition"], errors="coerce"
    )

    df_processed = df_processed.dropna(subset=["FinalPosition"])

    df_processed["HistoricalWinRate"] = pd.to_numeric(
        df_processed["HistoricalWinRate"], errors="coerce"
    ).fillna(0)

    df_processed["AvgFinishPosition"] = pd.to_numeric(
        df_processed["AvgFinishPosition"], errors="coerce"
    ).fillna(0)

    df_processed["ConstructorWinRate"] = pd.to_numeric(
        df_processed["ConstructorWinRate"], errors="coerce"
    ).fillna(0)

    df_processed["PrevSeasonConstructorRank"] = pd.to_numeric(
        df_processed["PrevSeasonConstructorRank"], errors="coerce"
    ).fillna(20)

    df_processed["CircuitType"] = df_processed["CircuitType"].fillna("Permanent").astype(str)
    df_processed["Weather"] = df_processed["Weather"].fillna("Dry").astype(str)

    df_processed["Winner"] = pd.to_numeric(df_processed["Winner"], errors="coerce").fillna(0).astype(int)

    print(f"Rows after cleaning: {len(df_processed)}")
    print(f"Overall winner balance: {df_processed['Winner'].mean():.2%}")

    for column in CATEGORICAL_FEATURES:
        encoder = LabelEncoder()
        encoded_column = f"{column}Encoded"
        encoder.fit(CATEGORY_VALUES[column])
        df_processed[column] = df_processed[column].where(
            df_processed[column].isin(CATEGORY_VALUES[column]),
            CATEGORY_VALUES[column][0],
        )
        df_processed[encoded_column] = encoder.transform(df_processed[column])

        mapping = {
            str(category): int(encoded_value)
            for category, encoded_value in zip(
                encoder.classes_,
                encoder.transform(encoder.classes_),
            )
        }
        print(f"  {column} encoding: {mapping}")

    df_processed = df_processed.sort_values(["Year", "Round"]).reset_index(drop=True)

    train_data = df_processed[df_processed["Year"] <= 2023].copy()
    val_data = df_processed[df_processed["Year"] == 2024].copy()

    if train_data.empty:
        raise ValueError("Training data is empty. Check Year column and collected data.")

    if val_data.empty:
        latest_year = int(df_processed["Year"].max())
        latest_round = int(
            df_processed[df_processed["Year"] == latest_year]["Round"].max()
        )
        raise ValueError(
            "Validation data for 2024 is empty. "
            f"Current dataset ends at {latest_year} round {latest_round}. "
            "Run collect_data.py again to resume collection."
        )

    iqr_bounds = fit_iqr_bounds(train_data, WINSORIZE_FEATURES, multiplier=1.5)
    train_data = apply_iqr_winsorizing(train_data, iqr_bounds, label="train")
    val_data = apply_iqr_winsorizing(val_data, iqr_bounds, label="validation")

    scaler = MinMaxScaler()
    train_data[NUMERIC_FEATURES] = scaler.fit_transform(train_data[NUMERIC_FEATURES])
    val_data[NUMERIC_FEATURES] = scaler.transform(val_data[NUMERIC_FEATURES])

    train_data = add_interactive_features(train_data)
    val_data = add_interactive_features(val_data)

    X_train = train_data[FEATURE_COLUMNS].copy()
    y_train = train_data["Winner"].copy()

    X_val = val_data[FEATURE_COLUMNS].copy()
    y_val = val_data["Winner"].copy()

    df_processed = pd.concat([train_data, val_data], ignore_index=True)
    df_processed.to_csv(PROCESSED_DATA_PATH, index=False)

    print("\nDataset split")
    print(f"  Train shape: {X_train.shape}")
    print(f"  Validation shape: {X_val.shape}")
    print(f"  Train winners share: {y_train.mean():.2%}")
    print(f"  Validation winners share: {y_val.mean():.2%}")
    print(f"  Processed dataset saved to: {PROCESSED_DATA_PATH}")

    if return_metadata:
        train_metadata = train_data[METADATA_COLUMNS].copy()
        val_metadata = val_data[METADATA_COLUMNS].copy()
        return X_train, y_train, X_val, y_val, scaler, train_metadata, val_metadata

    return X_train, y_train, X_val, y_val, scaler


if __name__ == "__main__":
    preprocess_data()
