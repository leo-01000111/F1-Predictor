"""
Open-Meteo weather collector.
No API key required.

Historical archive: https://archive-api.open-meteo.com/v1/archive
Forecast:          https://api.open-meteo.com/v1/forecast

Outputs:
  - data/raw/weather_history.parquet — historical race-day weather per circuit/date
"""
from __future__ import annotations

import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import requests
import pandas as pd
from loguru import logger

ROOT = Path(__file__).parent.parent.parent
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

HOURLY_VARS = [
    "temperature_2m",
    "precipitation",
    "windspeed_10m",
    "relativehumidity_2m",
    "cloudcover",
]

# Circuit coordinates (lat/lon). Matches config.yaml but kept here for standalone use.
CIRCUIT_COORDS: dict[str, tuple[float, float]] = {
    "bahrain": (26.0325, 50.5106),
    "saudi_arabia": (21.4858, 39.1925),
    "australia": (-37.8497, 144.9680),
    "japan": (34.8431, 136.5419),
    "china": (31.3389, 121.2197),
    "miami": (25.9581, -80.2389),
    "emilia_romagna": (44.3439, 11.7139),
    "monaco": (43.7347, 7.4206),
    "canada": (45.5000, -73.5228),
    "spain": (41.5700, 2.2611),
    "austria": (47.2197, 14.7647),
    "great_britain": (52.0733, -1.0169),
    "hungary": (47.5789, 19.2486),
    "belgium": (50.4372, 5.9714),
    "netherlands": (52.3888, 4.5409),
    "italy": (45.6156, 9.2811),
    "azerbaijan": (40.3725, 49.8533),
    "singapore": (1.2914, 103.8639),
    "united_states": (30.1328, -97.6411),
    "mexico": (19.4042, -99.0907),
    "brazil": (-23.7036, -46.6997),
    "las_vegas": (36.1147, -115.1728),
    "qatar": (25.4897, 51.4531),
    "abu_dhabi": (24.4672, 54.6031),
    # Historical circuits (no longer on calendar)
    "malaysia": (2.7606, 101.7379),    # Sepang
    "germany": (50.3356, 8.5697),      # Hockenheim / Nurburgring (approx)
    "russia": (43.4057, 39.9719),      # Sochi
    "turkey": (40.9517, 29.4050),      # Istanbul Park
    "portugal": (38.7597, -9.1491),    # Algarve International Circuit (Portimao)
    "france": (43.2506, 5.7917),       # Circuit Paul Ricard
}

WET_RACE_PRECIP_THRESHOLD = 2.0  # mm — cumulative race-day precipitation


def _fetch_hourly_weather(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    forecast: bool = False,
    retries: int = 5,
) -> Optional[pd.DataFrame]:
    """Fetch hourly weather for a lat/lon and date range.

    Retries up to `retries` times with exponential backoff on connection errors
    (e.g. Windows ConnectionResetError 10054 from rapid Open-Meteo requests).
    Uses ``Connection: close`` to avoid reusing a stale socket.
    """
    url = FORECAST_URL if forecast else ARCHIVE_URL
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "UTC",
    }
    # Close the connection after each request — prevents Windows socket reset
    # errors when many requests are fired in quick succession.
    headers = {"Connection": "close"}

    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            hourly = data.get("hourly", {})
            if not hourly or "time" not in hourly:
                return None
            df = pd.DataFrame(hourly)
            df["time"] = pd.to_datetime(df["time"])
            return df
        except Exception as exc:
            last_exc = exc
            wait = 2 ** attempt  # 1, 2, 4, 8, 16 s
            logger.debug(
                f"Weather fetch attempt {attempt + 1}/{retries} failed "
                f"({lat},{lon}) {start_date}: {exc}  — retrying in {wait}s"
            )
            time.sleep(wait)

    logger.warning(f"Weather fetch failed ({lat},{lon}) {start_date}: {last_exc}")
    return None


def _aggregate_race_day_weather(hourly_df: pd.DataFrame, race_date: str) -> dict:
    """
    Aggregate hourly data to race-day summary statistics.
    Race typically runs from ~13:00–16:00 local time (rough heuristic).
    We use the full day to keep it simple and timezone-agnostic.
    """
    day = pd.to_datetime(race_date).date()
    mask = hourly_df["time"].dt.date == day
    day_df = hourly_df[mask]

    if day_df.empty:
        return {}

    return {
        "temp_c_mean": day_df["temperature_2m"].mean(),
        "temp_c_max": day_df["temperature_2m"].max(),
        "precip_mm_total": day_df["precipitation"].sum(),
        "precip_mm_max_hourly": day_df["precipitation"].max(),
        "windspeed_ms_mean": day_df["windspeed_10m"].mean(),
        "windspeed_ms_max": day_df["windspeed_10m"].max(),
        "humidity_pct_mean": day_df["relativehumidity_2m"].mean(),
        "cloudcover_pct_mean": day_df["cloudcover"].mean(),
        "is_wet_race": int(day_df["precipitation"].sum() >= WET_RACE_PRECIP_THRESHOLD),
    }


# --------------------------------------------------------------------------- #
# Historical collection
# --------------------------------------------------------------------------- #

def collect_historical_weather(race_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given the race results DataFrame (with circuit_key + race_date columns),
    fetch and aggregate race-day weather for each race.

    race_df must have columns: year, round, circuit_key, race_date
    """
    required = {"year", "round", "circuit_key", "race_date"}
    missing = required - set(race_df.columns)
    if missing:
        raise ValueError(f"race_df missing columns: {missing}")

    # Deduplicate: one row per unique race
    races = race_df[["year", "round", "circuit_key", "race_date"]].drop_duplicates()
    logger.info(f"Fetching weather for {len(races)} race events")

    weather_rows = []
    for i, (_, row) in enumerate(races.iterrows()):
        circuit_key = str(row["circuit_key"])
        race_date = str(row["race_date"])

        # Try to resolve circuit coordinates
        coords = _resolve_circuit_coords(circuit_key)
        if coords is None:
            logger.warning(f"Unknown circuit '{circuit_key}', skipping weather")
            continue

        lat, lon = coords

        # Fetch ±1 day window to ensure we capture the full race day
        race_dt = pd.to_datetime(race_date)
        start = (race_dt - timedelta(days=1)).strftime("%Y-%m-%d")
        end = (race_dt + timedelta(days=1)).strftime("%Y-%m-%d")

        hourly_df = _fetch_hourly_weather(lat, lon, start, end, forecast=False)

        # Polite inter-request pause — prevents ConnectionResetError(10054) on
        # Windows when Open-Meteo rejects rapid-fire connections.
        time.sleep(0.5)

        if hourly_df is None:
            continue

        agg = _aggregate_race_day_weather(hourly_df, race_date)
        if not agg:
            continue

        weather_rows.append(
            {
                "year": int(row["year"]),
                "round": int(row["round"]),
                "circuit_key": circuit_key,
                "race_date": race_date,
                **agg,
            }
        )

        if (i + 1) % 20 == 0:
            logger.info(f"  Weather progress: {i + 1}/{len(races)} races fetched so far...")

    df = pd.DataFrame(weather_rows)
    logger.info(f"Collected weather for {len(df)}/{len(races)} races")
    return df


def _resolve_circuit_coords(circuit_key: str) -> Optional[tuple[float, float]]:
    """Resolve a circuit_key string to (lat, lon). Fuzzy match if needed."""
    if circuit_key in CIRCUIT_COORDS:
        return CIRCUIT_COORDS[circuit_key]

    # Fuzzy: check if any key is a substring
    for key, coords in CIRCUIT_COORDS.items():
        if key in circuit_key or circuit_key in key:
            return coords

    # Common aliases — also covers GP-suffix circuit_keys (e.g. "italian_gp" → "italy")
    aliases = {
        # Circuit name → coord key
        "monza": "italy",
        "silverstone": "great_britain",
        "spa": "belgium",
        "suzuka": "japan",
        "interlagos": "brazil",
        "yas_marina": "abu_dhabi",
        "marina_bay": "singapore",
        "hungaroring": "hungary",
        "red_bull_ring": "austria",
        "circuit_de_barcelona": "spain",
        "baku": "azerbaijan",
        "imola": "emilia_romagna",
        "zandvoort": "netherlands",
        "jeddah": "saudi_arabia",
        "lusail": "qatar",
        "cota": "united_states",
        "autodromo_hermanos_rodriguez": "mexico",
        "albert_park": "australia",
        "melbourne": "australia",
        "bahrain_international_circuit": "bahrain",
        # GP suffix patterns (circuit_key format from FastF1)
        "italian_gp": "italy",
        "italian_grand_prix": "italy",
        "belgian_gp": "belgium",
        "belgian_grand_prix": "belgium",
        "hungarian_gp": "hungary",
        "hungarian_grand_prix": "hungary",
        "chinese_gp": "china",
        "chinese_grand_prix": "china",
        "british_gp": "great_britain",
        "british_grand_prix": "great_britain",
        "japanese_gp": "japan",
        "japanese_grand_prix": "japan",
        "australian_gp": "australia",
        "australian_grand_prix": "australia",
        "bahrain_gp": "bahrain",
        "bahrain_grand_prix": "bahrain",
        "spanish_gp": "spain",
        "spanish_grand_prix": "spain",
        "monaco_gp": "monaco",
        "monaco_grand_prix": "monaco",
        "canadian_gp": "canada",
        "canadian_grand_prix": "canada",
        "austrian_gp": "austria",
        "austrian_grand_prix": "austria",
        "german_gp": "germany",
        "german_grand_prix": "germany",
        "dutch_gp": "netherlands",
        "dutch_grand_prix": "netherlands",
        "singapore_gp": "singapore",
        "singapore_grand_prix": "singapore",
        "russian_gp": "russia",
        "russian_grand_prix": "russia",
        "united_states_gp": "united_states",
        "us_grand_prix": "united_states",
        "mexican_gp": "mexico",
        "mexico_city_gp": "mexico",
        "mexican_grand_prix": "mexico",
        "brazil_gp": "brazil",
        "sao_paulo_gp": "brazil",
        "brazilian_grand_prix": "brazil",
        "abu_dhabi_gp": "abu_dhabi",
        "abu_dhabi_grand_prix": "abu_dhabi",
        "azerbaijan_gp": "azerbaijan",
        "azerbaijan_grand_prix": "azerbaijan",
        "saudi_arabian_gp": "saudi_arabia",
        "saudi_arabian_grand_prix": "saudi_arabia",
        "miami_gp": "miami",
        "miami_grand_prix": "miami",
        "emilia_romagna_gp": "emilia_romagna",
        "emilia-romagna_gp": "emilia_romagna",
        "qatar_gp": "qatar",
        "qatar_grand_prix": "qatar",
        "las_vegas_gp": "las_vegas",
        "las_vegas_grand_prix": "las_vegas",
        "malaysian_gp": "malaysia",
        "malaysian_grand_prix": "malaysia",
        "turkish_gp": "turkey",
        "turkish_grand_prix": "turkey",
        "portuguese_gp": "portugal",
        "portuguese_grand_prix": "portugal",
        "styrian_gp": "austria",
        "70th_anniversary_gp": "great_britain",
        "eifel_gp": "germany",
        "tuscan_gp": "italy",
        "bahrain_-_sakhir_gp": "bahrain",
        "sakhir_gp": "bahrain",             # 2020 Sakhir GP (Bahrain outer loop)
        "sakhir_grand_prix": "bahrain",
        # São Paulo GP — circuit_key uses accented ã, plain alias won't fuzzy-match
        "são_paulo_gp": "brazil",
        "são_paulo_grand_prix": "brazil",
        "french_gp": "france",
        "french_grand_prix": "france",
        "european_gp": "germany",           # approx - held at Nurburgring/Valencia
        "european_grand_prix": "germany",
    }
    for alias, mapped in aliases.items():
        if alias in circuit_key or circuit_key == alias:
            coords = CIRCUIT_COORDS.get(mapped)
            if coords:
                return coords

    return None


def save_weather(df: pd.DataFrame) -> None:
    path = RAW_DIR / "weather_history.parquet"
    df.to_parquet(path, index=False)
    logger.info(f"Saved {len(df)} weather rows → {path}")


def load_weather_history() -> pd.DataFrame:
    path = RAW_DIR / "weather_history.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Run weather collection first. Expected: {path}")
    return pd.read_parquet(path)


# --------------------------------------------------------------------------- #
# Forecast (used at inference time)
# --------------------------------------------------------------------------- #

def get_race_day_forecast(circuit_key: str, race_date: str) -> dict:
    """
    Fetch weather forecast for an upcoming race.
    Returns aggregated race-day weather dict.
    """
    coords = _resolve_circuit_coords(circuit_key)
    if coords is None:
        logger.error(f"Cannot find coords for circuit '{circuit_key}'")
        return {}

    lat, lon = coords
    race_dt = pd.to_datetime(race_date)
    start = race_dt.strftime("%Y-%m-%d")
    end = (race_dt + timedelta(days=1)).strftime("%Y-%m-%d")

    hourly_df = _fetch_hourly_weather(lat, lon, start, end, forecast=True)
    if hourly_df is None:
        return {}

    return _aggregate_race_day_weather(hourly_df, race_date)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import argparse
    from src.data_collection.f1_historical import load_race_results

    parser = argparse.ArgumentParser(description="Collect historical weather data for F1 races")
    parser.add_argument("--from-saved", action="store_true", help="Load race data from saved parquet")
    args = parser.parse_args()

    race_df = load_race_results()
    weather_df = collect_historical_weather(race_df)
    save_weather(weather_df)
    logger.info("Done.")
