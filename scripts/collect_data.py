"""
Simple script to collect all FPL data
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data_collection.historical_data_downloader import download_all_seasons
from src.data_collection.current_season_collector import CurrentSeasonCollector


def main():
    """Collect all available FPL data"""
    print("\n" + "="*60)
    print("FPL Data Collection Script")
    print("="*60)

    # Ask user what to collect
    print("\nWhat would you like to collect?")
    print("1. Historical data only (past seasons from GitHub)")
    print("2. Current season data only (2025-26 from FPL API)")
    print("3. Both historical and current season")
    print("4. Quick test (one historical season + limited current)")

    choice = input("\nEnter choice (1-4): ").strip()

    if choice == '1':
        print("\nDownloading historical data...")
        download_all_seasons()

    elif choice == '2':
        print("\nCollecting current season data...")
        collector = CurrentSeasonCollector(season='2025-26')
        collector.collect_all(include_player_details=True)

    elif choice == '3':
        print("\nDownloading historical data...")
        download_all_seasons()

        print("\n" + "="*60)
        print("\nCollecting current season data...")
        collector = CurrentSeasonCollector(season='2025-26')
        collector.collect_all(include_player_details=True)

    elif choice == '4':
        print("\nQuick test mode...")

        # Download one historical season
        print("\n→ Downloading 2023-24 season as test...")
        from src.data_collection.historical_data_downloader import download_season_data
        download_season_data('2023-24')

        # Collect limited current season
        print("\n→ Collecting current season (50 players only)...")
        collector = CurrentSeasonCollector(season='2025-26')
        collector.collect_all(include_player_details=True, max_players=50)

    else:
        print("Invalid choice. Exiting.")
        return

    print("\n" + "="*60)
    print("Data collection complete!")
    print("="*60)

    # Show next steps
    print("\nNext steps:")
    print("1. Verify data: python scripts/test_data_collection.py --test loading")
    print("2. Explore data: jupyter notebook")
    print("3. See guide: DATA_COLLECTION_GUIDE.md")
    print()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nData collection cancelled by user.")
    except Exception as e:
        print(f"\n\nError during data collection: {str(e)}")
        import traceback
        traceback.print_exc()
