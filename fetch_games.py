from nba_api.stats.endpoints import leaguegamefinder
import pandas as pd
import time

seasons = [
    '2025-26'
]

all_games = []

for season in seasons:
    print(f"Fetching {season}...")
    finder = leaguegamefinder.LeagueGameFinder(season_nullable=season)
    df = finder.get_data_frames()[0]
    all_games.append(df)
    time.sleep(1)  # respect rate limits

games = pd.concat(all_games, ignore_index=True)
games.to_csv('data/raw/games_current.csv', index=False)
print(f"Done. Total rows: {games.shape[0]}")