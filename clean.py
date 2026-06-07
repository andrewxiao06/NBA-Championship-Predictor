"""Clean and normalize raw NBA game data.

Produces one row per team per game (two rows per game) with a home/away flag,
normalized team abbreviations, parsed dates, and a season-type label. Dedup to a
single matchup row happens later, at the feature-join step in features.py.

Run:
    python clean.py
"""

import logging
from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/raw/games_raw.csv")
CURRENT_PATH = Path("data/raw/games_current.csv")
CLEAN_PATH = Path("data/processed/games_clean.csv")
CURRENT_CLEAN_PATH = Path("data/processed/games_current_clean.csv")

# SEASON_ID is a 5-char string: first char is the game-type prefix, last four
# are the season's starting year. Plan B02 only knew 2/4; the real data has six.
SEASON_TYPES_BY_PREFIX = {
    "1": "preseason",
    "2": "regular_season",
    "3": "allstar",
    "4": "playoffs",
    "5": "playin",
    "6": "tournament",
}

# Preseason and all-star games are not competitive, so we drop them.
KEEP_SEASON_TYPES = {"regular_season", "playoffs", "playin", "tournament"}

# Bug B11: team abbreviations drift across seasons. Map historical/alternate
# codes onto each franchise's current abbreviation so joins never KeyError.
TEAM_ABBREVIATION_MAP = {
    "NOH": "NOP",  # New Orleans Hornets -> Pelicans
    "NJN": "BKN",  # New Jersey Nets -> Brooklyn
    "NOK": "NOP",  # New Orleans/Oklahoma City Hornets -> Pelicans
    "SEA": "OKC",  # Seattle SuperSonics -> Oklahoma City Thunder
    "VAN": "MEM",  # Vancouver Grizzlies -> Memphis
    "CHH": "CHA",  # Charlotte Hornets (original) -> Charlotte
}

# Columns that must be present and non-null for a row to be usable.
KEY_COLUMNS = ["GAME_ID", "GAME_DATE", "TEAM_ABBREVIATION", "MATCHUP", "WL", "PTS"]

logger = logging.getLogger(__name__)


def load_raw(path: Path) -> pd.DataFrame:
    """Load raw game data from CSV. Raises FileNotFoundError if missing."""
    logger.info("Loading raw data from %s", path)
    if not path.exists():
        raise FileNotFoundError(f"Expected raw data at {path}; run fetch_games.py first.")
    # GAME_ID and SEASON_ID are zero-padded identifiers, not numbers.
    return pd.read_csv(path, dtype={"GAME_ID": str, "SEASON_ID": str})


def add_season_type(df: pd.DataFrame) -> pd.DataFrame:
    """Add SEASON_TYPE (regular_season/playoffs/...), IS_PLAYOFF, and SEASON year."""
    df = df.copy()
    prefix = df["SEASON_ID"].str[0]
    df["SEASON_TYPE"] = prefix.map(SEASON_TYPES_BY_PREFIX).fillna("unknown")
    df["IS_PLAYOFF"] = (df["SEASON_TYPE"] == "playoffs").astype(int)
    df["SEASON"] = df["SEASON_ID"].str[1:].astype(int)  # starting year, e.g. 2015
    return df


def filter_game_types(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only competitive game types (drop preseason and all-star)."""
    before = len(df)
    df = df[df["SEASON_TYPE"].isin(KEEP_SEASON_TYPES)].copy()
    logger.info("Filtered game types: %d -> %d rows", before, len(df))
    return df


def normalize_teams(df: pd.DataFrame) -> pd.DataFrame:
    """Map alternate/historical team abbreviations onto current ones (B11)."""
    df = df.copy()
    df["TEAM_ABBREVIATION"] = df["TEAM_ABBREVIATION"].replace(TEAM_ABBREVIATION_MAP)
    return df


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse GAME_DATE to datetime; drop rows that fail to parse."""
    df = df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    bad = df["GAME_DATE"].isna().sum()
    if bad:
        logger.warning("Dropping %d rows with unparseable GAME_DATE", bad)
        df = df[df["GAME_DATE"].notna()].copy()
    return df


def add_home_away(df: pd.DataFrame) -> pd.DataFrame:
    """Add IS_HOME (1 home / 0 away) and OPPONENT from the MATCHUP string.

    MATCHUP looks like "GSW vs. CLE" (home) or "GSW @ CLE" (away).
    """
    df = df.copy()
    df["IS_HOME"] = df["MATCHUP"].str.contains("vs.", regex=False).astype(int)
    # Opponent is the token after "vs." or "@".
    df["OPPONENT"] = (
        df["MATCHUP"].str.split(r"\s+(?:vs\.|@)\s+", regex=True).str[-1].str.strip()
    )
    df["OPPONENT"] = df["OPPONENT"].replace(TEAM_ABBREVIATION_MAP)
    return df


def drop_bad_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with nulls in key columns (data integrity)."""
    before = len(df)
    df = df.dropna(subset=KEY_COLUMNS).copy()
    dropped = before - len(df)
    if dropped:
        logger.warning("Dropped %d rows with nulls in key columns", dropped)
    return df


def keep_complete_games(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only games that have both team rows (exactly 2 rows per GAME_ID)."""
    counts = df.groupby("GAME_ID")["GAME_ID"].transform("size")
    complete = df[counts == 2].copy()
    dropped = (counts != 2).sum()
    if dropped:
        logger.warning("Dropped %d rows from games without exactly 2 team rows", dropped)
    return complete


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full cleaning pipeline on a raw games DataFrame."""
    df = df.drop_duplicates()
    df = add_season_type(df)
    df = filter_game_types(df)
    df = normalize_teams(df)
    df = parse_dates(df)
    df = add_home_away(df)
    df = drop_bad_rows(df)
    df = keep_complete_games(df)
    df = df.sort_values(["GAME_DATE", "GAME_ID", "IS_HOME"]).reset_index(drop=True)
    return df


def verify(df: pd.DataFrame) -> None:
    """Assert the Week-1 exit criteria hold for cleaned data."""
    assert df["WL"].isna().sum() == 0, "WL has nulls"
    assert df["GAME_DATE"].notna().all(), "GAME_DATE has unparsed values"
    assert (df.groupby("GAME_ID").size() == 2).all(), "not exactly 2 rows per game"
    assert set(df["IS_HOME"].unique()) <= {0, 1}, "IS_HOME not binary"
    logger.info(
        "Verified: %d games, %d rows, seasons %d-%d, types %s",
        df["GAME_ID"].nunique(),
        len(df),
        df["SEASON"].min(),
        df["SEASON"].max(),
        sorted(df["SEASON_TYPE"].unique()),
    )


def run(src: Path, dst: Path) -> pd.DataFrame:
    """Clean one raw file and write the result to dst."""
    df = clean(load_raw(src))
    verify(df)
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst, index=False)
    logger.info("Wrote %d rows to %s", len(df), dst)
    return df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(RAW_PATH, CLEAN_PATH)
    if CURRENT_PATH.exists():
        run(CURRENT_PATH, CURRENT_CLEAN_PATH)


if __name__ == "__main__":
    main()
