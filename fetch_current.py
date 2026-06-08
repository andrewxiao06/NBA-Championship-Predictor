"""Fetch the current NBA season (2025-26) into games_current.csv.

Run this before each prediction session so tonight's completed games are
included (Bug B06). Same shape as fetch_games.py, just a different season set
and output file.

Run:
    python fetch_current.py
"""

import time

import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder

SEASONS = ["2025-26"]
OUTPUT_PATH = "data/raw/games_current.csv"


def fetch_seasons(seasons: list[str]) -> pd.DataFrame:
    """Fetch all games for each season and return them concatenated."""
    all_games = []
    for season in seasons:
        print(f"Fetching {season}...")
        finder = leaguegamefinder.LeagueGameFinder(season_nullable=season)
        df = finder.get_data_frames()[0]
        print(f"  {len(df)} rows")
        all_games.append(df)
        time.sleep(1)  # respect rate limits (B03)
    return pd.concat(all_games, ignore_index=True)


def main() -> None:
    games = fetch_seasons(SEASONS)
    games.to_csv(OUTPUT_PATH, index=False)
    print(f"Done. Wrote {games.shape[0]} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
