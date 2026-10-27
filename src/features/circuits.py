"""
Canonical circuit IDs.

Circuit names arrive in several forms that never matched each other:
  - FastF1 event names in raw parquets      ("united_states_gp", "spanish_gp")
  - FastF1 / OpenF1 locations               ("Austin", "Marina Bay", "Montréal")
  - OpenF1 circuit_short_name at inference  ("austin", "catalunya", "madring")
  - track IDs in the feature lookup tables  ("cota", "albert_park")

canonical_circuit() maps all of them to the track IDs used by STREET_CIRCUITS,
CIRCUIT_SC_RATE and CIRCUIT_TYPICAL_COMPOUND, so circuit history, grid
conversion and the circuit lookups agree between training and inference.

Location wins over event name: event names move between venues (Spanish GP:
Barcelona -> Madrid in 2026; 2026 "Bahrain GP" held in Kuala Lumpur).
"""
from __future__ import annotations

import unicodedata
from typing import Optional

import pandas as pd

# Location / circuit short name -> track ID
_LOCATION_TO_CIRCUIT: dict[str, str] = {
    "melbourne": "albert_park",
    "albert_park": "albert_park",
    "shanghai": "shanghai",
    "suzuka": "suzuka",
    "sakhir": "bahrain",
    "bahrain": "bahrain",
    "jeddah": "jeddah",
    "miami": "miami",
    "miami_gardens": "miami",
    "montreal": "montreal",
    "monaco": "monaco",
    "monte_carlo": "monaco",
    "barcelona": "barcelona",
    "catalunya": "barcelona",
    "madrid": "madrid",
    "madring": "madrid",
    "spielberg": "red_bull_ring",
    "silverstone": "silverstone",
    "spa": "spa",
    "spa_francorchamps": "spa",
    "budapest": "hungaroring",
    "hungaroring": "hungaroring",
    "zandvoort": "zandvoort",
    "monza": "monza",
    "baku": "baku",
    "singapore": "singapore",
    "marina_bay": "singapore",
    "austin": "cota",
    "mexico_city": "mexico_city",
    "sao_paulo": "interlagos",
    "interlagos": "interlagos",
    "las_vegas": "las_vegas",
    "lusail": "losail",
    "abu_dhabi": "abu_dhabi",
    "yas_island": "abu_dhabi",
    "yas_marina": "abu_dhabi",
    "yas_marina_circuit": "abu_dhabi",
    "imola": "imola",
    # Sepang is listed as Kuala Lumpur (Malaysian GP 2014-2017, 2026 R16)
    "kuala_lumpur": "sepang",
    "sepang": "sepang",
    "sochi": "sochi",
    "istanbul": "istanbul",
    "portimao": "portimao",
    "le_castellet": "paul_ricard",
    "hockenheim": "hockenheim",
    "nurburgring": "nurburgring",
    "mugello": "mugello",
}

# FastF1 event-name key -> track ID (used when no location is available)
_EVENT_TO_CIRCUIT: dict[str, str] = {
    "australian_gp": "albert_park",
    "chinese_gp": "shanghai",
    "japanese_gp": "suzuka",
    "bahrain_gp": "bahrain",
    "sakhir_gp": "bahrain",
    "saudi_arabian_gp": "jeddah",
    "miami_gp": "miami",
    "canadian_gp": "montreal",
    "monaco_gp": "monaco",
    "spanish_gp": "barcelona",          # Madrid from 2026, see canonical_circuit()
    "barcelona_gp": "barcelona",
    "austrian_gp": "red_bull_ring",
    "styrian_gp": "red_bull_ring",
    "british_gp": "silverstone",
    "70th_anniversary_gp": "silverstone",
    "belgian_gp": "spa",
    "hungarian_gp": "hungaroring",
    "dutch_gp": "zandvoort",
    "italian_gp": "monza",
    "azerbaijan_gp": "baku",
    "european_gp": "baku",              # 2016 European GP was in Baku
    "singapore_gp": "singapore",
    "united_states_gp": "cota",
    "mexican_gp": "mexico_city",
    "mexico_city_gp": "mexico_city",
    "brazilian_gp": "interlagos",
    "sao_paulo_gp": "interlagos",
    "las_vegas_gp": "las_vegas",
    "qatar_gp": "losail",
    "abu_dhabi_gp": "abu_dhabi",
    "emilia_romagna_gp": "imola",
    "portuguese_gp": "portimao",
    "turkish_gp": "istanbul",
    "russian_gp": "sochi",
    "malaysian_gp": "sepang",
    "french_gp": "paul_ricard",
    "german_gp": "hockenheim",
    "eifel_gp": "nurburgring",
    "tuscan_gp": "mugello",
}


def _norm(name: object) -> str:
    """Lowercase ASCII with underscores: "São Paulo" -> "sao_paulo"."""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    s = s.strip().lower().replace("-", "_").replace(" ", "_")
    return s.replace("grand_prix", "gp")


def canonical_circuit(
    circuit_key: Optional[object] = None,
    location: Optional[object] = None,
    year: Optional[int] = None,
) -> str:
    """Return the canonical track ID for a circuit key and/or location."""
    if location is not None and not pd.isna(location):
        loc = _norm(location)
        if loc in _LOCATION_TO_CIRCUIT:
            return _LOCATION_TO_CIRCUIT[loc]

    key = _norm(circuit_key) if circuit_key is not None and not pd.isna(circuit_key) else ""
    if key == "spanish_gp" and year is not None and int(year) >= 2026:
        return "madrid"
    if key in _EVENT_TO_CIRCUIT:
        return _EVENT_TO_CIRCUIT[key]
    if key in _LOCATION_TO_CIRCUIT:
        return _LOCATION_TO_CIRCUIT[key]
    return key or "unknown"


def canonicalize_circuits(df: pd.DataFrame) -> pd.DataFrame:
    """Replace df["circuit_key"] with canonical track IDs (uses "circuit" location and "year" if present)."""
    if "circuit_key" not in df.columns:
        return df
    out = df.copy()
    locs = out["circuit"] if "circuit" in out.columns else pd.Series(None, index=out.index)
    years = out["year"] if "year" in out.columns else pd.Series(None, index=out.index)
    out["circuit_key"] = [
        canonical_circuit(k, l, y) for k, l, y in zip(out["circuit_key"], locs, years)
    ]
    return out
