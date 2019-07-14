#!/usr/bin/env python2.7

from melo import Melo
import nfldb
import numpy as np
import pandas as pd
from pyDOE import lhs


# Load NFL game data
db = nfldb.connect()
query = nfldb.Query(db)
query.game(season_type='Regular', finished=True)

fields = ['start_time', 'home_team', 'home_score', 'away_team', 'away_score']

games = pd.DataFrame([
    tuple(getattr(g, f) for f in fields)
    for g in sorted(query.as_games(), key=lambda g: g.start_time)
], columns=fields)

# Add columns to track rest days
teams = np.union1d(games.home_team, games.away_team)
time = {team: games.start_time.min() for team in teams}

for index, game in games.iterrows():
    home_rest, away_rest = [
        (game.start_time - time[getattr(game, team)]).days
        for team in ('home_team', 'away_team')
    ]
    for team in ('home_team', 'away_team'):
        time[getattr(game, team)] = game.start_time

    games.at[index, 'home_rest'] = home_rest if index > 16 else 7
    games.at[index, 'away_rest'] = away_rest if index > 16 else 7


def melo_wrapper(mode, kfactor, home_bias, decay_rate, fatigue):
    """
    Melo class wrapper.

    """
    if mode == 'spreads':
        # comparison lines
        lines = np.arange(-59.5, 60.5)

        # comparison bias factors
        home_fatigue = fatigue * np.exp(-games.home_rest / 7.0)
        away_fatigue = fatigue * np.exp(-games.away_rest / 7.0)
        biases = home_bias - fatigue * (home_fatigue - away_fatigue)

        # point spreads anti-commute
        commutes = False

        # pairwise comparison statistic
        comparisons = games.home_score - games.away_score
    elif mode == 'totals':
        # comparison lines
        lines = np.arange(-0.5, 101.5)

        # comparison bias factors
        home_fatigue = fatigue * np.exp(-games.home_rest / 7.0)
        away_fatigue = fatigue * np.exp(-games.away_rest / 7.0)
        biases = home_bias - fatigue * (home_fatigue + away_fatigue)

        # point totals commute
        commutes = True

        # pairwise comparison statistic
        comparisons = games.home_score + games.away_score
    else:
        raise ValueError('no such mode')

    def regress(years):
        return 1 - np.exp(-years/(decay_rate + 1e-12))

    regress_unit = 'year'

    model = Melo(kfactor, lines=lines, sigma=1.0, regress=regress,
                 regress_unit=regress_unit, commutes=commutes)

    model.fit(
        games.start_time,
        games.home_team,
        games.away_team,
        comparisons,
        biases,
    )

    return model


nfl_spreads = melo_wrapper('spreads', .166, .182, 3.48, 0.514)
nfl_totals = melo_wrapper('totals', .130, 0.064, 0.775, 0.276)


if __name__ == "__main__":

    labels, limits = map(list, zip(*[
        ('k',       (0.0,  0.5)),
        ('bias',    (0.0,  0.5)),
        ('decay',   (0.0,  1.0)),
        ('fatigue', (0.0,  1.0)),
    ]))

    xmin, xmax = map(np.array, zip(*limits))
    X = xmin + (xmax - xmin) * lhs(len(labels), samples=1000)
    y = np.array([melo_wrapper('totals', *x).loss for x in X])

    xopt = X[y.argsort()][:20].mean(axis=0)
    for label, x in zip(labels, xopt):
        print('{}: {}'.format(label, x))
