"""Download NFL game data and store in a SQL database"""

from datetime import datetime
import logging
import os
import requests
import sqlite3
import time

import nflgame
import pandas as pd
from sqlalchemy import create_engine

from . import dbfile, now


def initialize_database(conn):
    """
    Initialize the SQL database and create the games table.

    """
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS games(
        datetime DATETIME,
        season INTEGER,
        week INTEGER,
        team_home TEXT,
        qb_home TEXT,
        score_home INTEGER,
        team_away TEXT,
        qb_away TEXT,
        score_away INTEGER,
        UNIQUE(datetime, team_home, team_away));
    """)

    conn.commit()


def start_time(sched):
    """
    Return game's datetime.

    """
    eid = sched['eid']
    year = eid[:4]
    month = eid[4:6]
    day = eid[6:8]
    time = sched['time']

    if ('meridiem' in sched) and sched['meridiem'] in ['AM', 'PM']:
        meridiem = sched['meridiem']
    else:
        meridiem = 'PM'

    return datetime.strptime(f'{year}/{month}/{day} {time} {meridiem}',
                         '%Y/%m/%d %I:%M %p')


def update_model(cache_timestamp):
    """
    Return True is model needs updating, False otherwise.

    """
    games, last_updated = nflgame.sched._create_schedule()
    game_times = [start_time(g) for g in games.values() if start_time(g) < now]

    return True if cache_timestamp < max(game_times) else False


def starting_quarterbacks(game):
    """
    Estimate starting quarterbacks from number of pass attempts.

    """
    def quarterback(team):
        atts, qb = max([
            (d['att'], d['name'])
            for _, d in game.data[team]['stats']['passing'].items()
        ])

        return qb

    return (quarterback('home'), quarterback('away'))


def get_current_season_week():
    """
    Returns current nfl season and week via the feeds-rs api call

    For more api information visit http://www.nfl.com/feeds-rs?_wadl

    """
    url = 'http://www.nfl.com/feeds-rs/currentWeek.json'
    response = requests.get(url)
    output = response.json()

    current_season = output['seasonId']
    current_week = output['week']

    return (current_season, current_week)


def update_database(conn, rebuild=False):
    """
    Save games to the SQL database.

    """
    c = conn.cursor()
    c.execute("SELECT season, week FROM games ORDER BY season DESC, week DESC")
    last_update = c.fetchone()

    current_season, current_week = get_current_season_week()

    start_season, start_week = (
        (2009, 1) if (last_update is None) or
        (rebuild is True) else last_update
    )

    for season in range(start_season, current_season + 1):
        end_week = current_week if season == current_season else 17

        for week in range(start_week, end_week + 1):

            # print progress to stdout
            logging.info('updating season {} week {}'.format(season, week))

            games_gen = nflgame.games_gen(season, week, kind='REG', started=True)

            if games_gen is None:
                break

            # skip games which are not yet populated
            for g in games_gen:
                try:
                    qb_home, qb_away = starting_quarterbacks(g)

                    c.execute("""
                        INSERT INTO games(
                            datetime,
                            season,
                            week,
                            team_home,
                            qb_home,
                            score_home,
                            team_away,
                            qb_away,
                            score_away)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """, (start_time(g.schedule), season, week, g.home,
                          qb_home, g.score_home, g.away, qb_away, g.score_away))
                except (sqlite3.IntegrityError, KeyError):
                    continue

            conn.commit()

        start_week = 1


def load_games(update=False, rebuild=False):
    """
    Establish connection, then initialize and update database

    """
    engine = create_engine(r"sqlite:///{}".format(dbfile))
    conn = sqlite3.connect(str(dbfile))

    initialize_database(conn)

    if update or rebuild or not dbfile.exists():
        update_database(conn, rebuild=rebuild)

    return pd.read_sql_table('games', engine)
