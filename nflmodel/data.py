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
        date TEXT,
        season INTEGER,
        week INTEGER,
        team_home TEXT,
        qb_home TEXT,
        score_home INTEGER,
        team_away TEXT,
        qb_away TEXT,
        score_away INTEGER,
        UNIQUE(date, team_home, team_away));
    """)

    conn.commit()


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

            # loop over games in season and week
            for g in nflgame.games_gen(
                season, week=week, kind='REG', started=True):
                qb_home, qb_away = starting_quarterbacks(g)

                try:
                    date = '-'.join([g.eid[:4], g.eid[4:6], g.eid[6:8]])

                    c.execute("""
                        INSERT INTO games(
                            date,
                            season,
                            week,
                            team_home,
                            qb_home,
                            score_home,
                            team_away,
                            qb_away,
                            score_away)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """, (date, season, week, g.home, qb_home, g.score_home,
                          g.away, qb_away, g.score_away))
                except sqlite3.IntegrityError:
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

    if update is True:
        update_database(conn, rebuild=rebuild)

    return pd.read_sql_table('games', engine)
