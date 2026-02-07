"""Data collection modules for FPL API"""

from .getters import (
    get_data,
    get_individual_player_data,
    get_entry_data,
    get_entry_personal_data,
    get_entry_gws_data,
    get_entry_transfers_data,
    get_fixtures_data
)

from .parsers import (
    parse_players,
    parse_player_history,
    parse_player_gw_history,
    parse_entry_history,
    parse_entry_leagues,
    parse_transfer_history,
    parse_fixtures,
    parse_team_data
)

from .historical_data_downloader import (
    download_season_data,
    download_all_seasons,
    verify_downloads
)

from .current_season_collector import CurrentSeasonCollector

__all__ = [
    # API getters
    'get_data',
    'get_individual_player_data',
    'get_entry_data',
    'get_entry_personal_data',
    'get_entry_gws_data',
    'get_entry_transfers_data',
    'get_fixtures_data',
    # Parsers
    'parse_players',
    'parse_player_history',
    'parse_player_gw_history',
    'parse_entry_history',
    'parse_entry_leagues',
    'parse_transfer_history',
    'parse_fixtures',
    'parse_team_data',
    # Data downloaders
    'download_season_data',
    'download_all_seasons',
    'verify_downloads',
    'CurrentSeasonCollector',
]
