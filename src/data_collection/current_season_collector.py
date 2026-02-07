"""
Collect current season FPL data from the official API
"""
import os
import sys
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from configs.config import RAW_DATA_DIR
from src.data_collection.getters import (
    get_data,
    get_individual_player_data,
    get_fixtures_data
)
from src.data_collection.parsers import (
    parse_players,
    parse_fixtures,
    parse_team_data
)


class CurrentSeasonCollector:
    """Collector for current FPL season data"""

    def __init__(self, season='2025-26'):
        """Initialize collector

        Args:
            season (str): Season identifier (e.g., '2025-26')
        """
        self.season = season
        self.season_dir = RAW_DATA_DIR / season
        self.season_dir.mkdir(parents=True, exist_ok=True)
        self.gws_dir = self.season_dir / 'gws'
        self.gws_dir.mkdir(parents=True, exist_ok=True)
        self.players_dir = self.season_dir / 'players'
        self.players_dir.mkdir(parents=True, exist_ok=True)

    def collect_bootstrap_data(self):
        """Collect and parse bootstrap-static data (main FPL data)

        Returns:
            dict: The bootstrap data
        """
        print("Collecting bootstrap data from FPL API...")

        try:
            data = get_data()
            print(f"  ✓ Retrieved bootstrap data successfully")

            # Save raw JSON
            raw_json_path = self.season_dir / 'bootstrap_static.json'
            with open(raw_json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            print(f"  ✓ Saved raw JSON to {raw_json_path.name}")

            # Parse and save players data
            if 'elements' in data:
                players_df = pd.DataFrame(data['elements'])
                players_path = self.season_dir / 'players_raw.csv'
                players_df.to_csv(players_path, index=False)
                print(f"  ✓ Saved {len(players_df)} players to {players_path.name}")

            # Parse and save teams data
            if 'teams' in data:
                teams_df = pd.DataFrame(data['teams'])
                teams_path = self.season_dir / 'teams.csv'
                teams_df.to_csv(teams_path, index=False)
                print(f"  ✓ Saved {len(teams_df)} teams to {teams_path.name}")

            # Parse and save gameweek data
            if 'events' in data:
                events_df = pd.DataFrame(data['events'])
                events_path = self.season_dir / 'events.csv'
                events_df.to_csv(events_path, index=False)
                print(f"  ✓ Saved {len(events_df)} gameweeks to {events_path.name}")

            # Parse and save player types
            if 'element_types' in data:
                types_df = pd.DataFrame(data['element_types'])
                types_path = self.season_dir / 'element_types.csv'
                types_df.to_csv(types_path, index=False)
                print(f"  ✓ Saved {len(types_df)} player types to {types_path.name}")

            return data

        except Exception as e:
            print(f"  ✗ Error collecting bootstrap data: {str(e)}")
            return None

    def collect_fixtures(self):
        """Collect fixtures data

        Returns:
            pd.DataFrame: Fixtures data
        """
        print("\nCollecting fixtures data...")

        try:
            fixtures_data = get_fixtures_data()
            fixtures_df = pd.DataFrame(fixtures_data)

            fixtures_path = self.season_dir / 'fixtures.csv'
            fixtures_df.to_csv(fixtures_path, index=False)
            print(f"  ✓ Saved {len(fixtures_df)} fixtures to {fixtures_path.name}")

            return fixtures_df

        except Exception as e:
            print(f"  ✗ Error collecting fixtures: {str(e)}")
            return None

    def collect_player_details(self, player_ids=None, max_players=None):
        """Collect detailed data for individual players

        Args:
            player_ids (list): Specific player IDs to collect. If None, collect all
            max_players (int): Maximum number of players to collect (for testing)

        Returns:
            int: Number of players successfully collected
        """
        print("\nCollecting individual player data...")

        # If no player IDs specified, get all from bootstrap data
        if player_ids is None:
            bootstrap_path = self.season_dir / 'players_raw.csv'
            if not bootstrap_path.exists():
                print("  ! Bootstrap data not found. Run collect_bootstrap_data() first.")
                return 0

            players_df = pd.read_csv(bootstrap_path)
            player_ids = players_df['id'].tolist()

        if max_players:
            player_ids = player_ids[:max_players]

        print(f"  → Collecting data for {len(player_ids)} players...")

        success_count = 0
        failed_count = 0

        for player_id in tqdm(player_ids, desc="Players"):
            try:
                player_data = get_individual_player_data(player_id)

                # Create player directory
                player_dir = self.players_dir / f"player_{player_id}"
                player_dir.mkdir(parents=True, exist_ok=True)

                # Save gameweek history
                if 'history' in player_data and len(player_data['history']) > 0:
                    gw_df = pd.DataFrame(player_data['history'])
                    gw_path = player_dir / 'gw_history.csv'
                    gw_df.to_csv(gw_path, index=False)

                # Save past season history
                if 'history_past' in player_data and len(player_data['history_past']) > 0:
                    past_df = pd.DataFrame(player_data['history_past'])
                    past_path = player_dir / 'season_history.csv'
                    past_df.to_csv(past_path, index=False)

                # Save fixtures
                if 'fixtures' in player_data and len(player_data['fixtures']) > 0:
                    fixtures_df = pd.DataFrame(player_data['fixtures'])
                    fixtures_path = player_dir / 'fixtures.csv'
                    fixtures_df.to_csv(fixtures_path, index=False)

                success_count += 1

            except Exception as e:
                failed_count += 1
                # Don't print every error, just continue

        print(f"  ✓ Successfully collected {success_count} players")
        if failed_count > 0:
            print(f"  ✗ Failed to collect {failed_count} players")

        return success_count

    def create_merged_gw_file(self):
        """Merge all player gameweek data into a single file

        Returns:
            pd.DataFrame: Merged gameweek data
        """
        print("\nCreating merged gameweek file...")

        all_gw_data = []

        # Iterate through all player directories
        player_dirs = list(self.players_dir.glob('player_*'))

        for player_dir in tqdm(player_dirs, desc="Merging"):
            gw_file = player_dir / 'gw_history.csv'
            if gw_file.exists():
                try:
                    df = pd.read_csv(gw_file)
                    # Add player_id from directory name
                    player_id = int(player_dir.name.replace('player_', ''))
                    df['element'] = player_id
                    all_gw_data.append(df)
                except Exception as e:
                    pass  # Skip files that can't be read

        if all_gw_data:
            merged_df = pd.concat(all_gw_data, ignore_index=True)
            merged_path = self.gws_dir / 'merged_gw.csv'
            merged_df.to_csv(merged_path, index=False)
            print(f"  ✓ Saved merged data: {len(merged_df)} records from {len(all_gw_data)} players")
            return merged_df
        else:
            print("  ! No gameweek data found to merge")
            return None

    def collect_all(self, include_player_details=True, max_players=None):
        """Collect all available current season data

        Args:
            include_player_details (bool): Whether to collect detailed player data
            max_players (int): Limit number of players (for testing)

        Returns:
            dict: Collection summary
        """
        print(f"\n{'='*60}")
        print(f"Collecting Current Season Data: {self.season}")
        print(f"Target directory: {self.season_dir}")
        print(f"{'='*60}\n")

        summary = {
            'season': self.season,
            'timestamp': datetime.now().isoformat(),
            'bootstrap': False,
            'fixtures': False,
            'players_detailed': 0,
            'merged_gw': False
        }

        # Collect bootstrap data
        bootstrap = self.collect_bootstrap_data()
        summary['bootstrap'] = bootstrap is not None

        # Collect fixtures
        fixtures = self.collect_fixtures()
        summary['fixtures'] = fixtures is not None

        # Collect detailed player data
        if include_player_details:
            player_count = self.collect_player_details(max_players=max_players)
            summary['players_detailed'] = player_count

            # Create merged gameweek file
            if player_count > 0:
                merged = self.create_merged_gw_file()
                summary['merged_gw'] = merged is not None

        print(f"\n{'='*60}")
        print(f"Collection Complete!")
        print(f"  Bootstrap data: {'✓' if summary['bootstrap'] else '✗'}")
        print(f"  Fixtures: {'✓' if summary['fixtures'] else '✗'}")
        print(f"  Player details: {summary['players_detailed']}")
        print(f"  Merged GW file: {'✓' if summary['merged_gw'] else '✗'}")
        print(f"{'='*60}\n")

        # Save summary
        summary_path = self.season_dir / 'collection_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        return summary


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Collect current FPL season data')
    parser.add_argument('--season', type=str, default='2025-26',
                       help='Season identifier (default: 2025-26)')
    parser.add_argument('--bootstrap-only', action='store_true',
                       help='Only collect bootstrap data')
    parser.add_argument('--max-players', type=int,
                       help='Limit number of players to collect (for testing)')

    args = parser.parse_args()

    collector = CurrentSeasonCollector(season=args.season)

    if args.bootstrap_only:
        collector.collect_bootstrap_data()
        collector.collect_fixtures()
    else:
        collector.collect_all(
            include_player_details=True,
            max_players=args.max_players
        )
