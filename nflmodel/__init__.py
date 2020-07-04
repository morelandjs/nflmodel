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

home = Path(os.getenv('HOME'))
workdir = Path(os.getenv('WORKDIR', '.'))

cachedir = home / '.local/share/nflmodel'

if not cachedir.exists():
    cachedir.mkdir(parents=True)
