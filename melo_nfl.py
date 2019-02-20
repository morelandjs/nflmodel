#!/usr/bin/env python2.7

from datetime import datetime, timedelta
from pathlib import Path

import nfldb
import numpy as np
from skopt import gp_minimize

from melo import Melo


db = nfldb.connect()
q = nfldb.Query(db)
q.game(season_type='Regular', finished=True)

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
        'fermi': (spreads, np.arange(-59.5, 60.5)),
        'bose': (totals, np.arange(-0.5, 105.5)),
    }[mode]

    return Melo(
        dates, labels1, labels2, values, lines=lines,
        mode=mode, k=k, bias=bias, smooth=smooth,
        decay=lambda t: 1 if t < timedelta(weeks=20) else decay
    )


def from_cache(mode, retrain=False, **kwargs):
    """
    Load the melo args from the cache if available, otherwise
    train and cache a new instance.

    """
    cachefile = Path('cachedir', '{}.cache'.format(mode.lower()))

    if not retrain and cachefile.exists():
        args = np.loadtxt(cachefile)
        return melo_wrapper(mode, *args)

    def obj(args):
        melo = melo_wrapper(mode, *args)
        return melo.entropy()

    x0 = {
        'fermi': (0.20, 0.15, 0.60, 7.0),
        'bose': (0.20, 0.00, 0.60, 7.0),
    }[mode]

    bounds = {
        'fermi': [(0.1, 0.3), (0.1, 0.2), (0.55, 0.75), (0.0, 15.0)],
        'bose': [(0.1, 0.3), (-0.01, 0.01), (0.55, 0.75), (0.0, 15.0)],
    }[mode]

    res = gp_minimize(obj, bounds, n_calls=100, n_jobs=4, verbose=True)

    print("mode: {}".format(mode))
    print("best mean absolute error: {:.4f}".format(res.fun))
    print("best parameters: {}".format(res.x))

    if not cachefile.parent.exists():
        cachefile.parent.mkdir()

    np.savetxt(cachefile, res.x)
    return melo_wrapper(mode, *res.x)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='calibrate model parameters for point spreads and totals',
        argument_default=argparse.SUPPRESS
    )

    parser.add_argument(
        '--retrain', action='store_true', default=False,
        help='retrain even if model args are cached'
    )

    args = parser.parse_args()
    kwargs = vars(args)

    for mode in 'fermi', 'bose':
        from_cache(mode, **kwargs)
else:
    nfl_spreads = from_cache('fermi')
    nfl_totals = from_cache('bose')
