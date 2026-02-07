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

__all__ = [
    'get_data',
    'get_individual_player_data',
    'get_entry_data',
    'get_entry_personal_data',
    'get_entry_gws_data',
    'get_entry_transfers_data',
    'get_fixtures_data',
    'parse_players',
    'parse_player_history',
    'parse_player_gw_history',
    'parse_entry_history',
    'parse_entry_leagues',
    'parse_transfer_history',
    'parse_fixtures',
    'parse_team_data'
]
