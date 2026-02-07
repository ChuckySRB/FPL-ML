"""Quick test to verify the schema fix"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.preprocessing import FPLDataLoader

print("Testing data loader with fixed schemas...")

loader = FPLDataLoader()
seasons = loader.get_available_seasons()

if seasons:
    test_season = seasons[0]
    print(f"\nTesting with season: {test_season}")

    try:
        # This was failing before
        gw_df = loader.load_gameweeks(test_season)
        print(f"✓ Successfully loaded {len(gw_df):,} gameweek records")
        print(f"✓ Columns: {len(gw_df.columns)}")
        print(f"✓ Fix verified!")

    except Exception as e:
        print(f"✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
else:
    print("No seasons available. Run data collection first.")
