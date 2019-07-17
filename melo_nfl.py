#!/usr/bin/env python2.7

import operator
from pathlib import Path
import pickle

from hyperopt import fmin, hp, tpe
from melo import Melo
import nfldb
import numpy as np


class MeloNFL(Melo):
    """
    Generate NFL point-spread or point-total predictions
    using the Margin-dependent Elo (MELO) model.

    """
    def __init__(self, mode, kfactor, home_field, halflife, fatigue):

        self.mode = mode
        self.kfactor = kfactor
        self.home_field = home_field
        self.halflife = halflife
        self.fatigue = fatigue

        # model operation mode: 'spread' or 'total'
        if self.mode not in ['spread', 'total']:
            raise ValueError(
                "Unknown mode; valid options are 'spread' and 'total'")

        # mode-specific training hyperparameters
        self.commutes, self.compare, self.lines = {
            'total': (True, operator.add, np.arange(-0.5, 101.5)),
            'spread': (False, operator.sub, np.arange(-59.5, 60.5)),
        }[mode]

        # nfl game data
        db = nfldb.connect()
        query = nfldb.Query(db)
        query.game(season_type='Regular')
        self.games = np.rec.array(
            [g for g in self.gamedata(query)],
            names=[
                'start_time',
                'home_team',
                'away_team',
                'home_score',
                'away_score',
                'home_bias',
            ]
        )

        # instantiate the Melo base class
        Melo.__init__(self, self.kfactor, lines=self.lines, sigma=1.0,
                      regress=self.regress, regress_unit='year',
                      commutes=self.commutes)

        # train on completed games only
        self.completed_games = self.games[
            (self.games.home_score > 0) | (self.games.away_score > 0)
        ]

        # calibrate the model using the game data
        self.fit(
            self.completed_games.start_time,
            self.completed_games.home_team,
            self.completed_games.away_team,
            self.compare(
                self.completed_games.home_score,
                self.completed_games.away_score,
            ),
            self.completed_games.home_bias
        )

    def gamedata(self, query):
        """
        Generator that yields a tuple of attributes for each game.

        """
        start_time = {}

        for g in sorted(query.as_games(), key=lambda g: g.start_time):

            try:
                home_rest = (g.start_time - start_time[g.home_team]).days
                away_rest = (g.start_time - start_time[g.away_team]).days
            except KeyError:
                home_rest = 250
                away_rest = 250

            for team in [g.home_team, g.away_team]:
                start_time[team] = g.start_time

            yield (
                g.start_time,
                g.home_team,
                g.away_team,
                g.home_score,
                g.away_score,
                self.bias(home_rest, away_rest)
            )

    def regress(self, years):
        """
        Regresses future ratings to the mean.

        """
        with np.errstate(divide='ignore'):
            return 1 - .5**np.divide(years, self.halflife)

    def bias(self, home_rest_days, away_rest_days):
        """
        Computes circumstantial bias factor given each team's rest.
        Accounts for home field advantage and rest.

        """
        home_fatigue = self.fatigue * np.exp(-home_rest_days / 7.)
        away_fatigue = self.fatigue * np.exp(-away_rest_days / 7.)

        return self.home_field - self.compare(home_fatigue, away_fatigue)

    def probability(self, times, labels1, labels2, lines=0, bias=None):
        """
        Survival function probability distribution

        """
        bias = self.home_field if bias is None else bias
        return super(MeloNFL, self).probability(
            times, labels1, labels2, bias=bias, lines=lines
        )

    def percentile(self, times, labels1, labels2, p=50, bias=None):
        """
        Distribution percentiles

        """
        bias = self.home_field if bias is None else bias
        return super(MeloNFL, self).percentile(
            times, labels1, labels2, bias=bias, p=p
        )

    def quantile(self, times, labels1, labels2, q=.5, bias=None):
        """
        Distribution quantiles

        """
        bias = self.home_field if bias is None else bias
        return super(MeloNFL, self).quantile(
            times, labels1, labels2, bias=bias, q=q
        )

    def mean(self, times, labels1, labels2, bias=None):
        """
        Distribution mean

        """
        bias = self.home_field if bias is None else bias
        return super(MeloNFL, self).mean(
            times, labels1, labels2, bias=bias
        )

    def median(self, times, labels1, labels2, bias=None):
        """
        Distribution median

        """
        bias = self.home_field if bias is None else bias
        return super(MeloNFL, self).median(
            times, labels1, labels2, bias=bias
        )

    def sample(self, times, labels1, labels2, size=100, bias=None):
        """
        Sample the distribution

        """
        bias = self.home_field if bias is None else bias
        return super(MeloNFL, self).sample(
            times, labels1, labels2, bias=bias, size=size
        )


def calibrated_parameters(mode, max_evals=200, retrain=False):
    """
    Optimizes the MeloNFL model hyper parameters. Returns cached values
    if retrain is False and the parameters are cached, otherwise it
    optimizes the parameters and saves them to the cache.

    """
    cachedir = Path('/home/morelandjs/.local/share/melo_nfl')

    if not cachedir.exists():
        cachedir.mkdir()

    cachefile = cachedir / '{}.pkl'.format(mode)

    if not retrain and cachefile.exists():
        return pickle.load(cachefile.open(mode='rb'))

    def objective(args):
        return MeloNFL(mode, *args).loss

    space = (
        hp.uniform('kfactor', 0, 0.5),
        hp.uniform('home_field', 0.0, 0.5),
        hp.uniform('halflife', 0.0, 5.0),
        hp.uniform('fatigue', 0, 1.0),
    )

    parameters = fmin(objective, space, algo=tpe.suggest,
                      max_evals=max_evals)

    print(parameters)

    with cachefile.open(mode='wb') as f:
        pickle.dump(parameters, f)

    return parameters


retrain = (True if __name__ == '__main__' else False)

nfl_spreads, nfl_totals = [
    MeloNFL(mode, **calibrated_parameters(mode, retrain=retrain))
    for mode in ('spread', 'total')
]
