"""Project initialization and common objects"""

from datetime import datetime
import logging
import os
from pathlib import Path
import requests
import sys


logging.basicConfig(
    stream=sys.stdout,
    format='[%(levelname)s][%(module)s] %(message)s',
    level=os.getenv('LOGLEVEL', 'info').upper()
)

logging.getLogger('nflgame').setLevel(logging.WARNING)

now = datetime.now()

home = Path(os.getenv('HOME'))
workdir = Path(os.getenv('WORKDIR', '.'))

cachedir = home / '.local/share/nflmodel'

if not cachedir.exists():
    cachedir.mkdir(parents=True)

dbfile = cachedir / 'nfldb.sqlite'


def get_current_season_week():
    """
    Returns current nfl season and week via the feeds-rs api call

    For more api information visit http://www.nfl.com/feeds-rs?_wadl

    """
    try:
        url = "http://www.nfl.com/feeds-rs/currentWeek.json"
        response = requests.get(url)
        output = response.json()
    except requests.exceptions.ConnectionError:
        sys.exit("error: nflmodel requires an internet connection.")

    # week is capped to week 17
    current_season = output["seasonId"]
    current_week = min(output["week"], 17)

    return (current_season, current_week)
