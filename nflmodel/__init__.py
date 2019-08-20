"""Project initialization and common objects"""

from datetime import datetime
import logging
import os
from pathlib import Path
import sys


logging.basicConfig(
    stream=sys.stdout,
    format='[%(levelname)s][%(module)s] %(message)s',
    level=os.getenv('LOGLEVEL', 'info').upper()
)

now = datetime.now()

home = Path.home()
workdir = Path.cwd()

cachedir = home / '.local/share/nflmodel'
cachedir.mkdir(parents=True, exist_ok=True)

dbfile = cachedir / 'nfldb.sqlite'
