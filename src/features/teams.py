"""
Canonical team names.

Team names arrive in several forms that never matched each other:
  - FastF1 live timing                 ("Red Bull Racing", "Racing Bulls", "Alpine")
  - FastF1's Jolpica fallback / Jolpica ("Red Bull", "RB F1 Team", "Alpine F1 Team")
  - Jolpica constructorId              ("red_bull", "rb", "alfa")
  - OpenF1 team_name at inference      (same style as live timing)

In 2026 rounds 1-2 came from live timing and rounds 3-18 from Jolpica, so team
form, reliability and standings were split in two for four teams, and live
inference (OpenF1 names) would have matched only rounds 1-2.

canonical_team() maps every name of a team lineage to one name, the 2026
branding, so team features carry across rebrands:
  Toro Rosso -> AlphaTauri -> RB -> Racing Bulls
  Force India -> Racing Point -> Aston Martin
  Lotus F1 -> Renault -> Alpine
  Sauber -> Alfa Romeo -> Kick Sauber -> Audi
The website colours teams by these names (substring match), so keep them.
"""
from __future__ import annotations

import re

import pandas as pd

_LINEAGE: dict[str, tuple[str, ...]] = {
    "Mercedes": ("mercedes", "mercedes_amg_petronas", "mercedes_amg_petronas_f1_team"),
    "Ferrari": ("ferrari", "scuderia_ferrari", "scuderia_ferrari_hp"),
    "McLaren": ("mclaren", "mclaren_f1_team", "mclaren_formula_1_team"),
    "Williams": ("williams", "williams_racing", "atlassian_williams_racing"),
    "Haas F1 Team": ("haas", "haas_f1_team", "moneygram_haas_f1_team"),
    "Red Bull Racing": ("red_bull", "red_bull_racing", "oracle_red_bull_racing"),
    "Racing Bulls": (
        "toro_rosso", "scuderia_toro_rosso", "alphatauri", "scuderia_alphatauri",
        "rb", "rb_f1_team", "visa_cash_app_rb", "racing_bulls", "visa_cash_app_racing_bulls",
    ),
    "Aston Martin": (
        "force_india", "sahara_force_india", "racing_point", "racing_point_force_india",
        "bwt_racing_point", "aston_martin", "aston_martin_aramco",
        "aston_martin_aramco_f1_team", "aston_martin_f1_team",
    ),
    "Alpine": ("lotus", "lotus_f1", "lotus_f1_team", "renault", "alpine", "alpine_f1_team", "bwt_alpine_f1_team"),
    "Audi": (
        "sauber", "alfa", "alfa_romeo", "alfa_romeo_racing", "alfa_romeo_f1_team",
        "kick_sauber", "stake_f1_team_kick_sauber", "audi", "audi_f1_team",
    ),
    "Cadillac": ("cadillac", "cadillac_f1_team"),
    "Manor": ("marussia", "manor", "manor_marussia", "manor_racing"),
    "Caterham": ("caterham",),
}

_ALIASES: dict[str, str] = {alias: name for name, aliases in _LINEAGE.items() for alias in aliases}


def _norm(name: object) -> str:
    """Lowercase, non-alphanumerics to underscores: "RB F1 Team" -> "rb_f1_team"."""
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def canonical_team(name: object) -> str:
    """Return the canonical team name; unknown names come back stripped, unchanged."""
    if name is None or (not isinstance(name, str) and pd.isna(name)):
        return "Unknown"
    return _ALIASES.get(_norm(name), str(name).strip())


def canonicalize_teams(df: pd.DataFrame) -> pd.DataFrame:
    """Replace df["team"] with canonical team names."""
    if "team" not in df.columns:
        return df
    out = df.copy()
    out["team"] = out["team"].map(canonical_team)
    return out
