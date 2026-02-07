"""
Test script for data collection pipeline
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data_collection.historical_data_downloader import download_season_data, verify_downloads
from src.data_collection.current_season_collector import CurrentSeasonCollector
from src.preprocessing.data_loader import FPLDataLoader


def test_historical_download(test_season='2023-24'):
    """Test historical data download for one season

    Args:
        test_season (str): Season to test with
    """
    print("\n" + "="*60)
    print("TEST 1: Historical Data Download")
    print("="*60)

    print(f"\nDownloading sample season: {test_season}")
    stats = download_season_data(test_season)

    if stats['success'] > 0:
        print("\n✓ Historical download test PASSED")
        return True
    else:
        print("\n✗ Historical download test FAILED")
        return False


def test_current_season_collection(max_players=10):
    """Test current season data collection

    Args:
        max_players (int): Number of players to test with
    """
    print("\n" + "="*60)
    print("TEST 2: Current Season Data Collection")
    print("="*60)

    print(f"\nCollecting current season data (limited to {max_players} players)...")

    try:
        collector = CurrentSeasonCollector(season='2025-26')

        # Test bootstrap data
        print("\n→ Testing bootstrap data collection...")
        bootstrap = collector.collect_bootstrap_data()

        if bootstrap:
            print("✓ Bootstrap data collected successfully")
        else:
            print("✗ Failed to collect bootstrap data")
            return False

        # Test fixtures
        print("\n→ Testing fixtures collection...")
        fixtures = collector.collect_fixtures()

        if fixtures is not None:
            print("✓ Fixtures collected successfully")
        else:
            print("✗ Failed to collect fixtures")
            return False

        # Test player details (limited)
        print(f"\n→ Testing player details collection ({max_players} players)...")
        player_count = collector.collect_player_details(max_players=max_players)

        if player_count > 0:
            print(f"✓ Collected {player_count} player details")
        else:
            print("✗ Failed to collect player details")
            return False

        print("\n✓ Current season collection test PASSED")
        return True

    except Exception as e:
        print(f"\n✗ Current season collection test FAILED: {str(e)}")
        return False


def test_data_loading():
    """Test data loading functionality"""
    print("\n" + "="*60)
    print("TEST 3: Data Loading")
    print("="*60)

    loader = FPLDataLoader()

    # Get available seasons
    print("\n→ Checking available seasons...")
    seasons = loader.get_available_seasons()

    if not seasons:
        print("⊙ No seasons available yet - run data collection first")
        return None

    print(f"✓ Found {len(seasons)} seasons: {seasons}")

    # Try loading data from the first available season
    test_season = seasons[0]
    print(f"\n→ Testing data load for season: {test_season}")

    summary = loader.get_data_summary(test_season)
    print(f"\nData summary for {test_season}:")
    print(f"  Files available: {list(summary['files'].keys())}")
    print(f"  Gameweeks: {summary['gameweeks']}")
    print(f"  Players: {summary['players']}")
    print(f"  Teams: {summary['teams']}")

    # Try loading each type of data
    tests_passed = 0
    tests_total = 0

    # Test players
    if 'players' in summary['files']:
        tests_total += 1
        try:
            players_df = loader.load_players(test_season)
            print(f"\n✓ Loaded players: {len(players_df)} rows, {len(players_df.columns)} columns")
            print(f"  Sample columns: {list(players_df.columns[:5])}")
            tests_passed += 1
        except Exception as e:
            print(f"\n✗ Failed to load players: {str(e)}")

    # Test teams
    if 'teams' in summary['files']:
        tests_total += 1
        try:
            teams_df = loader.load_teams(test_season)
            print(f"\n✓ Loaded teams: {len(teams_df)} rows")
            tests_passed += 1
        except Exception as e:
            print(f"\n✗ Failed to load teams: {str(e)}")

    # Test fixtures
    if 'fixtures' in summary['files']:
        tests_total += 1
        try:
            fixtures_df = loader.load_fixtures(test_season)
            print(f"\n✓ Loaded fixtures: {len(fixtures_df)} rows")
            tests_passed += 1
        except Exception as e:
            print(f"\n✗ Failed to load fixtures: {str(e)}")

    # Test gameweeks
    if summary['gameweeks'] > 0 or 'merged_gw' in summary['files']:
        tests_total += 1
        try:
            gw_df = loader.load_gameweeks(test_season)
            print(f"\n✓ Loaded gameweeks: {len(gw_df)} rows, {len(gw_df.columns)} columns")
            print(f"  Sample columns: {list(gw_df.columns[:8])}")
            tests_passed += 1
        except Exception as e:
            print(f"\n✗ Failed to load gameweeks: {str(e)}")

    print(f"\n→ Data loading tests: {tests_passed}/{tests_total} passed")

    if tests_passed == tests_total and tests_total > 0:
        print("✓ Data loading test PASSED")
        return True
    else:
        print("⊙ Data loading test PARTIAL")
        return None


def run_all_tests():
    """Run all data collection tests"""
    print("\n" + "#"*60)
    print("# FPL Data Collection Pipeline Test Suite")
    print("#"*60)

    results = {}

    # Test 1: Historical download (single season)
    results['historical'] = test_historical_download('2023-24')

    # Test 2: Current season collection (limited)
    results['current'] = test_current_season_collection(max_players=10)

    # Test 3: Data loading
    results['loading'] = test_data_loading()

    # Summary
    print("\n" + "#"*60)
    print("# Test Summary")
    print("#"*60)

    for test_name, result in results.items():
        status = "✓ PASS" if result is True else ("⊙ SKIP" if result is None else "✗ FAIL")
        print(f"{test_name.capitalize():20} {status}")

    passed = sum(1 for r in results.values() if r is True)
    total = len([r for r in results.values() if r is not None])

    print(f"\nOverall: {passed}/{total} tests passed")
    print("#"*60 + "\n")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test FPL data collection pipeline')
    parser.add_argument('--test', choices=['historical', 'current', 'loading', 'all'],
                       default='all', help='Which test to run')
    parser.add_argument('--season', type=str, default='2023-24',
                       help='Season to test with (for historical)')
    parser.add_argument('--max-players', type=int, default=10,
                       help='Max players to collect (for current)')

    args = parser.parse_args()

    if args.test == 'historical':
        test_historical_download(args.season)
    elif args.test == 'current':
        test_current_season_collection(args.max_players)
    elif args.test == 'loading':
        test_data_loading()
    else:
        run_all_tests()
