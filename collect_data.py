import argparse
import os
import time
import warnings
from typing import Dict, List, Tuple

import fastf1
import numpy as np
import pandas as pd
from fastf1.ergast import Ergast

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
START_YEAR = 2014
END_YEAR = 2024
CACHE_DIR = "../cache"
RAW_DATA_PATH = "../figures/f1_race_data.csv"
PARTIAL_DATA_PATH = "../figures/f1_race_data_partial.csv"
HISTORICAL_WINDOW = 10
LOAD_RACE_WEATHER = True
EXPECTED_RACE_COUNTS = {
    2014: 19,
    2015: 19,
    2016: 21,
    2017: 20,
    2018: 21,
    2019: 21,
    2020: 17,
    2021: 22,
    2022: 22,
    2023: 22,
    2024: 24,
}

os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

ergast = Ergast()


def is_rate_limit_error(error: Exception) -> bool:
    """Проверяет, является ли ошибка лимитом API FastF1."""
    message = str(error).lower()
    return (
        "500 calls/h" in message
        or "rate limit" in message
        or "too many requests" in message
        or "429" in message
    )


def load_session_with_retry(
    year: int,
    round_number: int,
    session_type: str,
    retries: int = 3,
    sleep_seconds: int = 5,
):
    """
    Загружает сессию FastF1 с повторными попытками.

    session_type:
    - 'R' — race
    - 'Q' — qualifying

    Если достигнут лимит API, функция сразу пробрасывает ошибку наверх,
    чтобы сохранить прогресс и остановиться.
    """
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            session = fastf1.get_session(year, round_number, session_type)

            session.load(
                telemetry=False,
                laps=False,
                weather=LOAD_RACE_WEATHER and session_type == "R",
                messages=False,
            )

            return session

        except Exception as error:
            last_error = error

            if is_rate_limit_error(error):
                print(
                    f"  [RATE LIMIT] FastF1/API limit reached while loading "
                    f"{year} round {round_number} session {session_type}: {error}"
                )
                raise error

            print(
                f"  [WARN] Failed to load {year} round {round_number} "
                f"session {session_type} "
                f"(attempt {attempt}/{retries}): {error}"
            )

            if attempt < retries:
                time.sleep(sleep_seconds)

    print(
        f"  [ERROR] Skipping {year} round {round_number} "
        f"session {session_type}: {last_error}"
    )
    return None


def classify_circuit(event_name: str) -> str:
    """Классифицирует трассу как Street или Permanent."""
    if not isinstance(event_name, str):
        return "Permanent"

    street_keywords = [
        "Monaco",
        "Singapore",
        "Azerbaijan",
        "Baku",
        "Las Vegas",
        "Miami",
        "Saudi Arabian",
        "Jeddah",
        "Australian",
        "Melbourne",
        "Canadian",
        "Montreal",
    ]

    event_name_lower = event_name.lower()

    for keyword in street_keywords:
        if keyword.lower() in event_name_lower:
            return "Street"

    return "Permanent"


def detect_weather(session) -> str:
    """
    Определяет погоду по данным FastF1.

    Так как для старых сезонов погода часто недоступна,
    по умолчанию возвращаем Dry.
    """
    try:
        weather_data = getattr(session, "weather_data", None)

        if weather_data is not None and not weather_data.empty:
            if "Rainfall" in weather_data.columns:
                rainfall = weather_data["Rainfall"].fillna(False)

                if rainfall.astype(bool).any():
                    return "Wet"

    except Exception:
        pass

    return "Dry"


def get_quali_positions(year: int, round_number: int) -> Dict[str, int]:
    """
    Возвращает словарь:
    Driver Abbreviation -> qualifying position.

    Если квалификация недоступна, дальше используется позиция 20.
    """
    quali_session = load_session_with_retry(year, round_number, "Q")
    quali_positions = {}

    if quali_session is None:
        return quali_positions

    try:
        results = getattr(quali_session, "results", None)

        if results is None or results.empty:
            return quali_positions

        for _, row in results.iterrows():
            abbreviation = row.get("Abbreviation")
            position = row.get("Position")

            if pd.notna(abbreviation) and pd.notna(position):
                quali_positions[str(abbreviation)] = int(position)

    except Exception as error:
        print(
            f"  [WARN] Could not parse qualifying results "
            f"for {year} round {round_number}: {error}"
        )

    return quali_positions


def get_historical_stats(
    driver_id: str,
    current_year: int,
    current_round: int,
    historical_records: List[Dict],
    window: int = HISTORICAL_WINDOW,
) -> Tuple[float, float]:
    """
    Возвращает:
    - HistoricalWinRate
    - AvgFinishPosition

    Для rookie drivers:
    - HistoricalWinRate = 0
    - AvgFinishPosition = 0
    """
    if not historical_records:
        return 0.0, 0.0

    historical_df = pd.DataFrame(historical_records)

    if historical_df.empty:
        return 0.0, 0.0

    driver_history = historical_df[
        (historical_df["Driver"] == driver_id)
        & (
            (historical_df["Year"] < current_year)
            | (
                (historical_df["Year"] == current_year)
                & (historical_df["Round"] < current_round)
            )
        )
    ].sort_values(["Year", "Round"]).tail(window)

    if driver_history.empty:
        return 0.0, 0.0

    historical_win_rate = float(driver_history["Winner"].mean())
    avg_finish_position = float(driver_history["FinalPosition"].mean())

    if np.isnan(historical_win_rate):
        historical_win_rate = 0.0

    if np.isnan(avg_finish_position):
        avg_finish_position = 0.0

    return historical_win_rate, avg_finish_position


def get_constructor_historical_stats(
    constructor_name: str,
    current_year: int,
    current_round: int,
    historical_records: List[Dict],
    window: int = HISTORICAL_WINDOW,
) -> float:
    """Возвращает процент побед команды за последние N гонок."""
    if not historical_records:
        return 0.0

    historical_df = pd.DataFrame(historical_records)

    if historical_df.empty:
        return 0.0

    constructor_history = historical_df[
        (historical_df["Constructor"] == constructor_name)
        & (
            (historical_df["Year"] < current_year)
            | (
                (historical_df["Year"] == current_year)
                & (historical_df["Round"] < current_round)
            )
        )
    ].sort_values(["Year", "Round"])

    if constructor_history.empty:
        return 0.0

    constructor_races = (
        constructor_history.groupby(["Year", "Round"], as_index=False)["Winner"]
        .max()
        .sort_values(["Year", "Round"])
        .tail(window)
    )

    constructor_win_rate = float(constructor_races["Winner"].mean())

    if np.isnan(constructor_win_rate):
        return 0.0

    return constructor_win_rate


def recompute_historical_features(records: List[Dict]) -> List[Dict]:
    """Recomputes rolling historical features for loaded partial/raw records."""
    if not records:
        return records

    recomputed_records = []
    driver_history = {}
    constructor_history = {}

    sorted_records = sorted(
        records,
        key=lambda item: (
            int(item["Year"]),
            int(item["Round"]),
            int(item["FinalPosition"]),
        ),
    )

    current_event = None
    event_records = []

    def flush_event(group: List[Dict]) -> None:
        if not group:
            return

        constructor_event_winners = {}

        for record in group:
            driver_id = str(record["Driver"])
            constructor = str(record["Constructor"])
            driver_recent = driver_history.get(driver_id, [])[-HISTORICAL_WINDOW:]
            constructor_recent = constructor_history.get(constructor, [])[
                -HISTORICAL_WINDOW:
            ]

            if driver_recent:
                historical_win_rate = float(
                    np.mean([item["Winner"] for item in driver_recent])
                )
                avg_finish_position = float(
                    np.mean([item["FinalPosition"] for item in driver_recent])
                )
            else:
                historical_win_rate = 0.0
                avg_finish_position = 0.0

            constructor_win_rate = (
                float(np.mean(constructor_recent)) if constructor_recent else 0.0
            )

            updated_record = dict(record)
            updated_record["HistoricalWinRate"] = historical_win_rate
            updated_record["AvgFinishPosition"] = avg_finish_position
            updated_record["ConstructorWinRate"] = constructor_win_rate
            recomputed_records.append(updated_record)

            constructor_event_winners[constructor] = max(
                int(record["Winner"]),
                constructor_event_winners.get(constructor, 0),
            )

        for record in group:
            driver_id = str(record["Driver"])
            driver_history.setdefault(driver_id, []).append(
                {
                    "FinalPosition": int(record["FinalPosition"]),
                    "Winner": int(record["Winner"]),
                }
            )

        for constructor, constructor_winner in constructor_event_winners.items():
            constructor_history.setdefault(constructor, []).append(constructor_winner)

    for record in sorted_records:
        event_key = (int(record["Year"]), int(record["Round"]))

        if current_event is None:
            current_event = event_key

        if event_key != current_event:
            flush_event(event_records)
            event_records = []
            current_event = event_key

        event_records.append(record)

    flush_event(event_records)

    return recomputed_records


def normalize_constructor_name(name: str) -> str:
    """Нормализует названия команд для сопоставления с Ergast/FastF1."""
    if not isinstance(name, str):
        return ""

    lowered = name.lower().strip()

    aliases = {
        "red bull racing": "red bull",
        "red bull": "red bull",
        "mercedes": "mercedes",
        "ferrari": "ferrari",
        "mclaren": "mclaren",
        "williams": "williams",
        "force india": "force india",
        "aston martin": "aston martin",
        "racing point": "racing point",
        "renault": "renault",
        "alpine": "alpine",
        "haas": "haas",
        "sauber": "sauber",
        "alfa romeo": "alfa romeo",
        "rb": "rb",
        "visa cash app rb": "rb",
        "alphatauri": "alphatauri",
        "alpha tauri": "alphatauri",
        "toro rosso": "toro rosso",
        "lotus f1": "lotus",
        "lotus": "lotus",
        "caterham": "caterham",
        "marussia": "marussia",
        "manor": "manor",
    }

    for key, normalized in aliases.items():
        if key in lowered:
            return normalized

    return lowered


def extract_standings_dataframe(response) -> pd.DataFrame:
    """
    Достаёт DataFrame из ответа Ergast.

    В разных версиях FastF1 Ergast может возвращать:
    - ErgastMultiResponse с .content;
    - DataFrame напрямую;
    - список DataFrame.
    """
    if response is None:
        return pd.DataFrame()

    if isinstance(response, pd.DataFrame):
        return response

    if hasattr(response, "content"):
        content = response.content

        if isinstance(content, list) and len(content) > 0:
            if isinstance(content[0], pd.DataFrame):
                return content[0]

        if isinstance(content, dict):
            for value in content.values():
                if isinstance(value, pd.DataFrame):
                    return value

    if isinstance(response, list) and len(response) > 0:
        if isinstance(response[0], pd.DataFrame):
            return response[0]

    return pd.DataFrame()


def get_constructor_prev_season_rank(
    constructor_name: str,
    year: int,
    standings_cache: Dict[int, pd.DataFrame],
) -> int:
    """
    Возвращает ранг команды в Кубке конструкторов прошлого сезона.
    Если данные недоступны — возвращает 20.
    """
    previous_year = year - 1

    if previous_year < 1950:
        return 20

    try:
        if previous_year not in standings_cache:
            response = ergast.get_constructor_standings(
                season=previous_year,
                result_type="pandas",
            )

            standings_df = extract_standings_dataframe(response)

            if standings_df is None or standings_df.empty:
                print(f"  [WARN] Empty constructor standings for {previous_year}")
                standings_cache[previous_year] = pd.DataFrame()
            else:
                standings_cache[previous_year] = standings_df

        standings_df = standings_cache[previous_year]

        if standings_df is None or standings_df.empty:
            return 20

        constructor_normalized = normalize_constructor_name(constructor_name)

        possible_name_columns = [
            "constructorName",
            "ConstructorName",
            "constructor_name",
            "constructorId",
            "name",
            "Name",
        ]

        possible_position_columns = [
            "position",
            "Position",
            "positionText",
        ]

        name_column = next(
            (
                column
                for column in possible_name_columns
                if column in standings_df.columns
            ),
            None,
        )

        position_column = next(
            (
                column
                for column in possible_position_columns
                if column in standings_df.columns
            ),
            None,
        )

        if name_column is None:
            constructor_related_columns = [
                column
                for column in standings_df.columns
                if "constructor" in column.lower() or "name" in column.lower()
            ]

            if constructor_related_columns:
                name_column = constructor_related_columns[0]

        if position_column is None:
            position_related_columns = [
                column
                for column in standings_df.columns
                if "position" in column.lower()
            ]

            if position_related_columns:
                position_column = position_related_columns[0]

        if name_column is None or position_column is None:
            print(
                f"  [WARN] Could not identify standings columns for {previous_year}. "
                f"Available columns: {list(standings_df.columns)}"
            )
            return 20

        for _, row in standings_df.iterrows():
            standing_constructor = normalize_constructor_name(
                str(row.get(name_column, ""))
            )

            names_match = (
                constructor_normalized
                and standing_constructor
                and (
                    constructor_normalized in standing_constructor
                    or standing_constructor in constructor_normalized
                )
            )

            if names_match:
                position = row.get(position_column)

                if pd.notna(position):
                    try:
                        return int(position)
                    except ValueError:
                        return int(float(position))

    except Exception as error:
        if is_rate_limit_error(error):
            raise error

        print(
            f"  [WARN] Could not get constructor standings for "
            f"{constructor_name}, {previous_year}: {error}"
        )

    return 20


def parse_position_value(value):
    """Parses FastF1/Ergast position values such as 1, 1.0, '1', or '\\N'."""
    if pd.isna(value):
        return None

    text_value = str(value).strip()

    if not text_value or text_value.upper() in {"R", "D", "E", "W", "N", "\\N", "NC"}:
        return None

    try:
        return int(float(text_value))
    except (TypeError, ValueError):
        return None


def is_valid_race_records(records: List[Dict]) -> bool:
    """A race is usable only when it has rows and exactly one winner."""
    if not records:
        return False

    return sum(int(record.get("Winner", 0)) for record in records) == 1


def remove_invalid_races(df: pd.DataFrame) -> pd.DataFrame:
    """Drops already saved races that have no winner or multiple winners."""
    if df is None or df.empty:
        return df

    winner_counts = df.groupby(["Year", "Round"])["Winner"].sum()
    invalid_events = set(winner_counts[winner_counts != 1].index)

    if not invalid_events:
        return df

    print(
        f"[WARN] Removing {len(invalid_events)} invalid saved races "
        "from resume state:"
    )

    for year, round_number in sorted(invalid_events):
        race_name = df[(df["Year"] == year) & (df["Round"] == round_number)][
            "Race"
        ].iloc[0]
        race_name = str(race_name).encode("ascii", errors="replace").decode("ascii")
        print(f"  {int(year)} round {int(round_number)}: {race_name}")

    valid_mask = ~df.set_index(["Year", "Round"]).index.isin(invalid_events)
    return df[valid_mask].copy()


def extract_race_records(
    year: int,
    round_number: int,
    event_name: str,
    session,
    quali_positions: Dict[str, int],
    historical_records: List[Dict],
    standings_cache: Dict[int, pd.DataFrame],
) -> List[Dict]:
    """Формирует строки датасета для одной гонки."""
    records = []

    results = getattr(session, "results", None)

    if results is None or results.empty:
        print(f"  [WARN] No race results for {year} round {round_number}")
        return records

    weather = detect_weather(session)
    circuit_type = classify_circuit(event_name)

    for _, row in results.iterrows():
        try:
            driver_id = str(row.get("Abbreviation", "")).strip()

            if not driver_id:
                continue

            constructor = str(row.get("TeamName", "Unknown")).strip()

            position_value = row.get("Position")
            classified_position = row.get("ClassifiedPosition")

            final_position = None
            is_dns = False

            final_position = parse_position_value(position_value)

            if final_position is None:
                final_position = parse_position_value(classified_position)

            if final_position is None:
                is_dns = True

            if final_position is None:
                final_position = 20

            first_name = str(row.get("FirstName", "")).strip()
            last_name = str(row.get("LastName", "")).strip()
            driver_name = f"{first_name} {last_name}".strip()

            qualifying_position = quali_positions.get(driver_id, 20)

            if qualifying_position is None or pd.isna(qualifying_position):
                qualifying_position = 20

            historical_win_rate, avg_finish_position = get_historical_stats(
                driver_id=driver_id,
                current_year=year,
                current_round=round_number,
                historical_records=historical_records,
            )

            constructor_win_rate = get_constructor_historical_stats(
                constructor_name=constructor,
                current_year=year,
                current_round=round_number,
                historical_records=historical_records,
            )

            prev_season_constructor_rank = get_constructor_prev_season_rank(
                constructor_name=constructor,
                year=year,
                standings_cache=standings_cache,
            )

            winner = int(final_position == 1)

            record = {
                "Year": year,
                "Round": round_number,
                "Race": event_name,
                "Driver": driver_id,
                "DriverName": driver_name,
                "Constructor": constructor,
                "QualifyingPosition": int(qualifying_position),
                "FinalPosition": int(final_position),
                "Winner": winner,
                "HistoricalWinRate": historical_win_rate,
                "AvgFinishPosition": avg_finish_position,
                "ConstructorWinRate": constructor_win_rate,
                "PrevSeasonConstructorRank": int(prev_season_constructor_rank),
                "CircuitType": circuit_type,
                "Weather": weather,
                "IsDNS": bool(is_dns),
            }

            records.append(record)

        except Exception as error:
            if is_rate_limit_error(error):
                raise error

            print(
                f"  [WARN] Could not parse driver row for "
                f"{year} round {round_number}: {error}"
            )

    if not is_valid_race_records(records):
        print(
            f"  [WARN] Invalid race result for {year} round {round_number}: "
            f"winner count = {sum(record['Winner'] for record in records)}. "
            "Race will be retried on the next run."
        )
        return []

    return records


def build_history_from_records(records: List[Dict]) -> List[Dict]:
    """Восстанавливает historical_records из уже собранных строк."""
    historical_records = []

    sorted_records = sorted(
        records,
        key=lambda item: (int(item["Year"]), int(item["Round"])),
    )

    for record in sorted_records:
        historical_records.append(
            {
                "Year": int(record["Year"]),
                "Round": int(record["Round"]),
                "Driver": record["Driver"],
                "Constructor": record["Constructor"],
                "FinalPosition": int(record["FinalPosition"]),
                "Winner": int(record["Winner"]),
            }
        )

    return historical_records


def is_dataset_complete(df: pd.DataFrame) -> bool:
    """Returns True when all expected seasons and races are present."""
    if df is None or df.empty or "Year" not in df.columns or "Round" not in df.columns:
        return False

    winner_counts = df.groupby(["Year", "Round"])["Winner"].sum()
    if (winner_counts != 1).any():
        return False

    races_by_year = df.groupby("Year")["Round"].nunique()

    for year in range(START_YEAR, END_YEAR + 1):
        expected_count = EXPECTED_RACE_COUNTS.get(year)

        if expected_count is None:
            return False

        if int(races_by_year.get(year, 0)) < expected_count:
            return False

    return True


def describe_dataset_coverage(df: pd.DataFrame) -> str:
    """Builds a short human-readable coverage string for a collected dataset."""
    if df is None or df.empty or "Year" not in df.columns or "Round" not in df.columns:
        return "empty dataset"

    sorted_df = df.sort_values(["Year", "Round"])
    first = sorted_df.iloc[0]
    last = sorted_df.iloc[-1]

    return (
        f"{int(first['Year'])} round {int(first['Round'])} -> "
        f"{int(last['Year'])} round {int(last['Round'])}, rows={len(df)}"
    )


def progress_from_dataframe(df: pd.DataFrame) -> Tuple[List[Dict], List[Dict], set]:
    """Converts an existing CSV dataframe into collector resume state."""
    if df is None or df.empty:
        return [], [], set()

    df = remove_invalid_races(df)
    df = df.sort_values(["Year", "Round", "FinalPosition"]).reset_index(drop=True)
    all_records = df.to_dict("records")
    all_records = recompute_historical_features(all_records)
    historical_records = build_history_from_records(all_records)
    processed_events = set(zip(df["Year"].astype(int), df["Round"].astype(int)))

    return all_records, historical_records, processed_events


def load_partial_progress(force_reload: bool = False) -> Tuple[List[Dict], List[Dict], set]:
    """
    Загружает partial CSV, если он есть.

    Возвращает:
    - all_records;
    - historical_records;
    - processed_events.
    """
    all_records = []
    historical_records = []
    processed_events = set()

    if force_reload:
        return all_records, historical_records, processed_events

    candidates = []

    for path in [PARTIAL_DATA_PATH, RAW_DATA_PATH]:
        if not os.path.exists(path):
            continue

        candidate_df = pd.read_csv(path)

        if candidate_df.empty:
            continue

        last_year = int(candidate_df["Year"].max()) if "Year" in candidate_df.columns else 0
        last_round = int(
            candidate_df[candidate_df["Year"] == last_year]["Round"].max()
        ) if "Round" in candidate_df.columns else 0

        candidates.append((last_year, last_round, len(candidate_df), path, candidate_df))

    if candidates:
        _, _, _, path, progress_df = max(candidates, key=lambda item: item[:3])
        all_records, historical_records, processed_events = progress_from_dataframe(
            progress_df
        )
        resume_df = pd.DataFrame(all_records)

        print(f"[INFO] Loaded resume dataset: {path}")
        print(f"[INFO] Resume coverage: {describe_dataset_coverage(resume_df)}")
        print(f"[INFO] Already processed races: {len(processed_events)}")

    return all_records, historical_records, processed_events


def save_partial_progress(records: List[Dict], processed_events: set) -> None:
    """Сохраняет промежуточный прогресс."""
    if not records:
        return

    partial_df = pd.DataFrame(records)
    partial_df = partial_df.sort_values(["Year", "Round", "FinalPosition"])
    partial_df.to_csv(PARTIAL_DATA_PATH, index=False)

    print(f"    Progress saved to: {PARTIAL_DATA_PATH}")
    print(f"    Processed races so far: {len(processed_events)}")


def save_final_dataset(records: List[Dict]) -> pd.DataFrame:
    """Сохраняет финальный датасет."""
    df = pd.DataFrame(records)

    if df.empty:
        raise RuntimeError(
            "No data was collected. Check FastF1 connection/cache and try again."
        )

    df = df.sort_values(["Year", "Round", "FinalPosition"]).reset_index(drop=True)
    df.to_csv(RAW_DATA_PATH, index=False)

    print("\n" + "=" * 70)
    print("Raw dataset saved")
    print("=" * 70)
    print(f"Path: {RAW_DATA_PATH}")
    print(f"Shape: {df.shape}")
    print(
        f"Winners: {df['Winner'].sum()} / {len(df)} "
        f"({df['Winner'].mean():.2%})"
    )
    print(f"Columns: {list(df.columns)}")

    return df


def main(force_reload: bool = False, resume: bool = True) -> pd.DataFrame:
    """
    Загружает данные FastF1 за 2014–2024,
    считает признаки и сохраняет f1_race_data.csv.

    Логика:
    - если есть финальный f1_race_data.csv и force_reload=False — читает его;
    - если есть f1_race_data_partial.csv — продолжает с него;
    - при лимите 500 calls/h сохраняет partial и аккуратно завершает сбор.
    """
    if os.path.exists(RAW_DATA_PATH) and not force_reload:
        print(f"[INFO] Existing raw dataset found: {RAW_DATA_PATH}")
        df = pd.read_csv(RAW_DATA_PATH)
        print(f"[INFO] Loaded dataset shape: {df.shape}")
        print(f"[INFO] Dataset coverage: {describe_dataset_coverage(df)}")

        if is_dataset_complete(df):
            return df

        if not resume:
            print("[WARN] Dataset is incomplete, but resume=False. Returning existing CSV.")
            return df

        print("[WARN] Dataset is incomplete. Continuing collection from existing progress.")

    all_records, historical_records, processed_events = load_partial_progress(
        force_reload=force_reload or not resume
    )

    standings_cache = {}

    print("=" * 70)
    print("Collecting Formula 1 race data with FastF1")
    print("=" * 70)

    for year in range(START_YEAR, END_YEAR + 1):
        print(f"\n[Season {year}] Loading schedule...")

        try:
            schedule = fastf1.get_event_schedule(year)

        except Exception as error:
            if is_rate_limit_error(error):
                if all_records:
                    save_partial_progress(all_records, processed_events)

                print("\n[RATE LIMIT] FastF1/API limit reached while loading schedule.")
                print("Wait about 60 minutes, then run the script again.")
                print("The collector will continue from partial CSV.")
                return pd.DataFrame(all_records)

            print(f"  [ERROR] Could not load schedule for {year}: {error}")
            continue

        for _, event in schedule.iterrows():
            try:
                round_number = int(event["RoundNumber"])
                event_name = str(event["EventName"])

                if round_number <= 0:
                    continue

                if (year, round_number) in processed_events:
                    print(
                        f"  Skipping already processed round "
                        f"{round_number}: {event_name}"
                    )
                    continue

                session5 = event.get("Session5")

                if pd.isna(session5):
                    continue

                print(f"  Processing round {round_number}: {event_name}")

                race_session = load_session_with_retry(year, round_number, "R")

                if race_session is None:
                    continue

                quali_positions = get_quali_positions(year, round_number)

                race_records = extract_race_records(
                    year=year,
                    round_number=round_number,
                    event_name=event_name,
                    session=race_session,
                    quali_positions=quali_positions,
                    historical_records=historical_records,
                    standings_cache=standings_cache,
                )

                if not race_records:
                    print(
                        f"    No valid rows added for round {round_number}. "
                        "It will be retried on the next run."
                    )
                    continue

                all_records.extend(race_records)

                for record in race_records:
                    historical_records.append(
                        {
                            "Year": record["Year"],
                            "Round": record["Round"],
                            "Driver": record["Driver"],
                            "Constructor": record["Constructor"],
                            "FinalPosition": record["FinalPosition"],
                            "Winner": record["Winner"],
                        }
                    )

                processed_events.add((year, round_number))

                print(f"    Added rows: {len(race_records)}")
                save_partial_progress(all_records, processed_events)

            except Exception as error:
                if is_rate_limit_error(error):
                    if all_records:
                        save_partial_progress(all_records, processed_events)

                    print("\n" + "=" * 70)
                    print("[RATE LIMIT] FastF1/API limit reached: 500 calls/h")
                    print("=" * 70)
                    print(f"Progress saved to: {PARTIAL_DATA_PATH}")
                    print("Wait about 60 minutes, then run again:")
                    print("  python collect_data.py")
                    print("The collector will continue from partial CSV.")

                    return pd.DataFrame(all_records)

                print(f"  [ERROR] Failed processing event in {year}: {error}")

    df = save_final_dataset(all_records)

    if os.path.exists(PARTIAL_DATA_PATH):
        print(f"[INFO] Partial file kept as backup: {PARTIAL_DATA_PATH}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect Formula 1 race data with FastF1."
    )
    parser.add_argument(
        "--force-reload",
        action="store_true",
        help="Ignore existing CSV files and collect everything again.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not continue from f1_race_data.csv or f1_race_data_partial.csv.",
    )
    args = parser.parse_args()

    main(force_reload=args.force_reload, resume=not args.no_resume)
