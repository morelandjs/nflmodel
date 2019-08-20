"""Download NFL game data and store in a SQL database"""

import logging
import sqlite3

import nflgame

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
        score_home INTEGER,
        team_away TEXT,
        score_away INTEGER,
        UNIQUE(date, team_home, team_away));
    """)

    conn.commit()


def update_database(conn, refresh=False):
    """
    Save games to the SQL database.

    """
    c = conn.cursor()
    c.execute("SELECT season FROM games ORDER BY date DESC LIMIT 1")
    last_update = c.fetchone()

    start_season = (
        2009 if (last_update is None) or
        (refresh is True) else last_update[0]
    )

    end_season = now.year

    # loop over nfl season years 2009-present
    for season in range(start_season, end_season + 1):

        # print progress to stdout
        logging.info(f'Updating Season {season}')

        # loop over games in season and week
        for g in nflgame.games_gen(season, kind='REG'):

            date = '-'.join([g.eid[:4], g.eid[4:6], g.eid[6:8]])
            week = g.schedule['week']
            values = (date, season, week,
                      g.home, g.score_home, g.away, g.score_away)

            try:
                c.execute("""
                    INSERT INTO games(
                        date,
                        season,
                        week,
                        team_home,
                        score_home,
                        team_away,
                        score_away)
                    VALUES (?, ?, ?, ?, ?, ?, ?);
                """, values)
            except sqlite3.IntegrityError:
                continue

    conn.commit()


def run(refresh=False):
    """
    Establish connection, then initialize and update database

    """
    conn = sqlite3.connect(str(dbfile))
    initialize_database(conn)
    update_database(conn, refresh=refresh)
    conn.close()
