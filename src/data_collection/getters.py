import requests
import json
import time


def _get_json(url, timeout=30, max_retries=3):
    '''Fetch JSON with bounded retries and an explicit timeout.'''
    last_error = None
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt + 1 < max_retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(
        f'FPL request failed after {max_retries} attempts: {url}'
    ) from last_error

def get_data():
    """ Retrieve the fpl player data from the hard-coded url
    """
    return _get_json(
        'https://fantasy.premierleague.com/api/bootstrap-static/')

def get_individual_player_data(player_id):
    """ Retrieve the player-specific detailed data

    Args:
        player_id (int): ID of the player whose data is to be retrieved
    """
    base_url = "https://fantasy.premierleague.com/api/element-summary/"
    full_url = base_url + str(player_id) + "/"
    return _get_json(full_url)

def get_entry_data(entry_id):
    """ Retrieve the summary/history data for a specific entry/team

    Args:
        entry_id (int) : ID of the team whose data is to be retrieved
    """
    base_url = "https://fantasy.premierleague.com/api/entry/"
    full_url = base_url + str(entry_id) + "/history/"
    return _get_json(full_url)

def get_entry_personal_data(entry_id):
    """ Retrieve the summary/history data for a specific entry/team

    Args:
        entry_id (int) : ID of the team whose data is to be retrieved
    """
    base_url = "https://fantasy.premierleague.com/api/entry/"
    full_url = base_url + str(entry_id) + "/"
    return _get_json(full_url)

def get_entry_gws_data(entry_id,num_gws,start_gw=1):
    """ Retrieve the gw-by-gw data for a specific entry/team

    Args:
        entry_id (int) : ID of the team whose data is to be retrieved
    """
    base_url = "https://fantasy.premierleague.com/api/entry/"
    gw_data = []
    for i in range(start_gw, num_gws+1):
        full_url = base_url + str(entry_id) + "/event/" + str(i) + "/picks/"
        data = _get_json(full_url)
        gw_data += [data]
    return gw_data

def get_entry_transfers_data(entry_id):
    """ Retrieve the transfer data for a specific entry/team

    Args:
        entry_id (int) : ID of the team whose data is to be retrieved
    """
    base_url = "https://fantasy.premierleague.com/api/entry/"
    full_url = base_url + str(entry_id) + "/transfers/"
    return _get_json(full_url)

def get_fixtures_data():
    """ Retrieve the fixtures data for the season
    """
    url = "https://fantasy.premierleague.com/api/fixtures/"
    return _get_json(url)

def main():
    data = get_data()
    with open('raw.json', 'w') as outf:
        json.dump(data, outf)

if __name__ == '__main__':
    main()
