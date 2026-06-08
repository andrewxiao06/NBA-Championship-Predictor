"""Fetch historical NBA games (2015-16 through 2024-25) into games_raw.csv.

One LeagueGameFinder call per season, with a sleep between calls to respect
nba_api rate limits (Bug B03). Each season returns two rows per game (one per
team) and all game types; cleaning/filtering happens later in clean.py.

Run:
    python fetch_games.py
"""

import time

import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder

# Historical training range. Current season (2025-26) is fetched separately by
# fetch_current.py into games_current.csv.
SEASONS = [
    "2015-16",
    "2016-17",
    "2017-18",
    "2018-19",
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
]

OUTPUT_PATH = "data/raw/games_raw.csv"


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
