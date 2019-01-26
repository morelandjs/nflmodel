#!/usr/bin/env python2.7

from datetime import datetime, timedelta
import nfldb
import numpy as np
from scipy.optimize import minimize

from melo import Melo


db = nfldb.connect()
q = nfldb.Query(db)
#q.game(season_type='Regular', finished=True)
q.game(finished=True)

dates = [g.start_time for g in q.as_games()]
labels1 = [g.home_team for g in q.as_games()]
labels2 = [g.away_team for g in q.as_games()]
spreads = [g.home_score - g.away_score for g in q.as_games()]
totals = [g.home_score + g.away_score for g in q.as_games()]


def melo_wrapper(mode, k, bias, decay, smooth, verbose=False):
    """
    Thin wrapper to pass arguments to the Melo library.

    """
    values, lines = {
        'Fermi': (spreads, np.arange(-59.5, 60.5)),
        'Bose': (totals, np.arange(-0.5, 105.5)),
    }[mode]

    return Melo(
        dates, labels1, labels2, values, lines=lines,
        mode=mode, k=k, bias=bias, smooth=smooth,
        decay=lambda t: 1 if t < timedelta(weeks=20) else decay
    )

nfl_spreads = melo_wrapper('Fermi', .257, .148, .627, 10)
nfl_totals = melo_wrapper('Bose', .154,  0.15, .588, 10)


if __name__ == "__main__":

    # optimize point total and point spread
    for mode in ['Fermi', 'Bose']:

        def obj(args):
            melo = melo_wrapper(mode, *args)
            residuals = melo.residuals()
            return np.abs(residuals).mean()

        x0 = (.20, .15, .60, 7.0)
        bounds = [(0.1, 0.3), (0.1, 0.2), (0.55, 0.75), (0.0, 20.0)]
        res = minimize(obj, x0=x0, bounds=bounds, tol=1e-2)

        # report diagnostics
        print("mode: {}".format(mode))
        print("best mean absolute error: {:.4f}".format(res.fun))
        print("best parameters: {}".format(res.x))
        print("last updated: {}".format(nfl_spreads.last_update))
