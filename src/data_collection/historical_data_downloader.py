"""
Download historical FPL data from the vaastav/Fantasy-Premier-League repository
"""
import os
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from configs.config import RAW_DATA_DIR, HISTORICAL_DATA_BASE_URL, SEASONS


def download_file(url, save_path, description=""):
    """Download a file from URL to save_path with progress bar

    Args:
        url (str): URL to download from
        save_path (Path): Path to save the file
        description (str): Description for progress bar

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        response = requests.get(url, stream=True, timeout=30)
        if response.status_code == 200:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            total_size = int(response.headers.get('content-length', 0))

            with open(save_path, 'wb') as f:
                if total_size == 0:
                    f.write(response.content)
                else:
                    with tqdm(total=total_size, unit='B', unit_scale=True,
                             desc=description, leave=False) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                            pbar.update(len(chunk))
            return True
        else:
            print(f"  ✗ Failed to download {url} (Status: {response.status_code})")
            return False
    except Exception as e:
        print(f"  ✗ Error downloading {url}: {str(e)}")
        return False


def download_season_data(season, base_url=HISTORICAL_DATA_BASE_URL):
    """Download all available data for a specific season

    Args:
        season (str): Season identifier (e.g., '2023-24')
        base_url (str): Base URL for historical data

    Returns:
        dict: Download statistics
    """
    print(f"\n{'='*60}")
    print(f"Downloading data for season {season}")
    print(f"{'='*60}")

    season_dir = RAW_DATA_DIR / season
    season_dir.mkdir(parents=True, exist_ok=True)

    stats = {'success': 0, 'failed': 0, 'skipped': 0}

    # Files to download at season level
    season_files = [
        'cleaned_players.csv',
        'players_raw.csv',
        'teams.csv',
        'fixtures.csv',
    ]

    # Download season-level files
    for filename in season_files:
        url = f"{base_url}/{season}/{filename}"
        save_path = season_dir / filename

        if save_path.exists():
            print(f"  ⊙ Skipping {filename} (already exists)")
            stats['skipped'] += 1
            continue

        print(f"  → Downloading {filename}...")
        if download_file(url, save_path, filename):
            stats['success'] += 1
        else:
            stats['failed'] += 1

    # Download gameweek data
    gws_dir = season_dir / 'gws'
    gws_dir.mkdir(parents=True, exist_ok=True)

    # Try to download merged_gw.csv first
    merged_url = f"{base_url}/{season}/gws/merged_gw.csv"
    merged_path = gws_dir / 'merged_gw.csv'

    if not merged_path.exists():
        print(f"  → Downloading merged gameweek data...")
        if download_file(merged_url, merged_path, "merged_gw.csv"):
            stats['success'] += 1
        else:
            stats['failed'] += 1
    else:
        print(f"  ⊙ Skipping merged_gw.csv (already exists)")
        stats['skipped'] += 1

    # Download individual gameweek files (GW 1-38)
    print(f"  → Downloading individual gameweek files...")
    gw_stats = {'success': 0, 'failed': 0, 'skipped': 0}

    for gw in range(1, 39):
        gw_filename = f"gw{gw}.csv"
        url = f"{base_url}/{season}/gws/{gw_filename}"
        save_path = gws_dir / gw_filename

        if save_path.exists():
            gw_stats['skipped'] += 1
            continue

        if download_file(url, save_path, f"GW{gw}"):
            gw_stats['success'] += 1
        else:
            gw_stats['failed'] += 1
            # If we get multiple failures in a row, stop trying
            if gw_stats['failed'] > 3:
                print(f"  ⊙ Stopping GW downloads after {gw} (likely end of available data)")
                break

        time.sleep(0.1)  # Be nice to the server

    stats['success'] += gw_stats['success']
    stats['failed'] += gw_stats['failed']
    stats['skipped'] += gw_stats['skipped']

    print(f"\nSeason {season} summary:")
    print(f"  ✓ Downloaded: {stats['success']}")
    print(f"  ⊙ Skipped: {stats['skipped']}")
    print(f"  ✗ Failed: {stats['failed']}")

    return stats


def download_all_seasons(seasons=None):
    """Download data for all specified seasons

    Args:
        seasons (list): List of seasons to download. If None, use config default

    Returns:
        dict: Overall statistics
    """
    if seasons is None:
        seasons = SEASONS

    print(f"\n{'#'*60}")
    print(f"# FPL Historical Data Downloader")
    print(f"# Downloading {len(seasons)} seasons")
    print(f"# Target directory: {RAW_DATA_DIR}")
    print(f"{'#'*60}")

    overall_stats = {'success': 0, 'failed': 0, 'skipped': 0}

    for season in seasons:
        stats = download_season_data(season)
        overall_stats['success'] += stats['success']
        overall_stats['failed'] += stats['failed']
        overall_stats['skipped'] += stats['skipped']
        time.sleep(0.5)  # Be nice to the server

    print(f"\n{'#'*60}")
    print(f"# Download Complete!")
    print(f"# Total downloaded: {overall_stats['success']}")
    print(f"# Total skipped: {overall_stats['skipped']}")
    print(f"# Total failed: {overall_stats['failed']}")
    print(f"{'#'*60}\n")

    return overall_stats


def verify_downloads(season=None):
    """Verify that downloaded files exist and are not empty

    Args:
        season (str): Specific season to verify, or None for all seasons

    Returns:
        dict: Verification results
    """
    if season:
        seasons_to_check = [season]
    else:
        seasons_to_check = SEASONS

    results = {}

    for s in seasons_to_check:
        season_dir = RAW_DATA_DIR / s
        if not season_dir.exists():
            results[s] = {'status': 'missing', 'files': []}
            continue

        files_found = list(season_dir.rglob('*.csv'))
        total_size = sum(f.stat().st_size for f in files_found)

        results[s] = {
            'status': 'exists',
            'files': len(files_found),
            'total_size_mb': round(total_size / (1024 * 1024), 2)
        }

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Download historical FPL data')
    parser.add_argument('--season', type=str, help='Download specific season (e.g., 2023-24)')
    parser.add_argument('--verify', action='store_true', help='Verify downloaded data')
    parser.add_argument('--all', action='store_true', help='Download all configured seasons')

    args = parser.parse_args()

    if args.verify:
        print("\nVerifying downloads...")
        results = verify_downloads()
        for season, info in results.items():
            if info['status'] == 'exists':
                print(f"  {season}: {info['files']} files, {info['total_size_mb']} MB")
            else:
                print(f"  {season}: NOT DOWNLOADED")
    elif args.season:
        download_season_data(args.season)
    else:
        download_all_seasons()
