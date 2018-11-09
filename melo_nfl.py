#!/usr/bin/env python2.7

import nfldb
from datetime import datetime, timedelta
import numpy as np
from skopt import gp_minimize

from melo import Melo
import scipy

import matplotlib.pyplot as plt

# Pull NFL game data
db = nfldb.connect()
q = nfldb.Query(db)
q.game(season_type='Regular', finished=True)

# Structure the pairwise comparison data in list format.
dates = [g.start_time for g in q.as_games()]
labels1 = [g.home_team for g in q.as_games()]
labels2 = [g.away_team for g in q.as_games()]
spreads = [g.home_score - g.away_score for g in q.as_games()]
totals = [g.home_score + g.away_score for g in q.as_games()]

def melo_wrapper(mode, k, bias, decay):
    """
    Thin wrapper to pass arguments to the Melo library.

    """
    values, lines = {
        'Fermi': (spreads, np.arange(-50.5, 51.5)),
        'Bose': (totals, np.arange(-0.5, 101.5)),
    }[mode]

    return Melo(
        dates, labels1, labels2, values, lines=lines,
        mode=mode, k=k, bias=bias,
        decay=lambda t: 1 if t < timedelta(weeks=20) else decay
    )

# calculate margin-dependent Elo ratings
nfl_spreads = melo_wrapper('Fermi', .245, .166, .597)
nfl_totals  = melo_wrapper('Bose', .133,  0.01, .698)

if __name__ == "__main__":

    # optimize point total and point spread
    for mode, bounds in [
            ('Fermi', [(0.2, 0.3), (0.1, 0.2), (.5, .6)]),
            ('Bose',  [(0.1, 0.2), (0.0, 0.01), (.6, .7)]),
    ]:

        def obj(parameters):
            """
            Evaluates the mean absolute error for a set of input
            parameters: kfactor, decay, regress.

            """
            k, bias, decay = parameters

            melo = melo_wrapper(mode, k, bias, decay)

            statistics = melo.statistics(smooth=5)
            mean, value = [statistics[k] for k in ('mean', 'value')]

            residuals = (mean - value)[256:]
            return np.abs(residuals).mean()

        res_gp = gp_minimize(obj, bounds, n_calls=100)

        # report diagnostics
        print("mode: {}".format(mode))
        print("best mean absolute error: {:.4f}".format(res_gp.fun))
        print("best parameters: {}".format(res_gp.x))
