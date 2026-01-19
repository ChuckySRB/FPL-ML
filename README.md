# Fantasy Premier League Machine Learning Assitant 
This project is implementation for Mashine Learning Masters Course on School of Electrical Engineering in Belgrade

### Data Structure

The data folder contains the data from past seasons as well as the current season. It is structured as follows:

+ season/cleaned_players.csv : The overview stats for the season
+ season/gws/gw_number.csv : GW-specific stats for the particular season
+ season/gws/merged_gws.csv : GW-by-GW stats for each player in a single file
+ season/players/player_name/gws.csv : GW-by-GW stats for that specific player
+ season/players/player_name/history.csv : Prior seasons history stats for that specific player.

### Accessing the Data Directly in Python

You can access data files within this repository programmatically using Python and the `pandas` library. Below is an example using the `data/2023-24/gws/merged_gw.csv` file. Similar methods can be applied to other data files in the repository. Note this is using the raw URL for direct file access, bypassing the GitHub UI.

```python
import pandas as pd

# URL of the CSV file (example)
url = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2023-24/gws/merged_gw.csv"

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(url)
```

### Data Dictionary

For a comprehensive explanation of all variables and columns in the dataset, see the [DATA_DICTIONARY.md](DATA_DICTIONARY.md) file.

### Player Position Data

In players_raw.csv, element_type is the field that corresponds to the position.
1 = GK
2 = DEF
3 = MID
4 = FWD

### Errata

+ GW35 expected points data is wrong (all values are 0).

### Contributing

+ If you feel like there is some data that is missing which you would like to see, then please feel free to create a PR or create an issue highlighting what is missing and what you would like to be added
+ If you have access to old data (pre-2016) then please feel free to create Pull Requests adding the data to the repo or create an issue with links to old data and I will add them myself.

### Using

If you use data from here for your website or blog posts, then I would humbly request that you please add a link back to this repo as the data source (and I would in turn add a link to your post/site as a notable usage of this repo).

## Downloading Your Team Data

You can download the data for your team by executing the following steps:

```
python teams_scraper.py <team_id>
#Eg: python teams_scraper.py 4582
```

This will create a new folder called "team_<team_id>_data18-19" with individual files of all the important data