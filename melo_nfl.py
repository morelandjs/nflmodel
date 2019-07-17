#!/usr/bin/env python2.7

import operator
from pathlib import Path
import pickle

from hyperopt import fmin, hp, tpe
from melo import Melo
import nfldb
import numpy as np
import pandas as pd


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
        if mode not in ['spread', 'total']:
            raise ValueError(
                "Unknown mode; valid options are 'spread' and 'total'")

        # mode-specific training hyperparameters
        commutes, compare, lines = {
            'total': (True, operator.add, np.arange(-0.5, 101.5)),
            'spread': (False, operator.sub, np.arange(-59.5, 60.5)),
        }[mode]

        # nfl score data
        db = nfldb.connect()
        query = nfldb.Query(db)
        query.game(season_type='Regular', finished=True)
        games = self.dataframe(query)

        # regress future ratings to the mean
        def regress(years):
            with np.errstate(divide='ignore'):
                return 1 - .5**np.divide(years, halflife)

        # instantiate the Melo base class
        Melo.__init__(self, kfactor, lines=lines, sigma=1.0, regress=regress,
                      regress_unit='year', commutes=commutes)

        # determine bias factors from home field advantage and fatigue
        home_fatigue = fatigue * np.exp(-games.home_rest / 7.)
        away_fatigue = fatigue * np.exp(-games.away_rest / 7.)
        biases = home_field - compare(home_fatigue, away_fatigue)

        # calibrate the model using the game data
        self.fit(
            games.start_time,
            games.home_team,
            games.away_team,
            compare(games.home_score, games.away_score),
            biases,
        )

    def dataframe(self, query):
        """
        Returns pandas dataframe of NFL game scores.

        """
        fields = [
            'start_time',
            'home_team',
            'home_score',
            'away_team',
            'away_score',
        ]

        games = pd.DataFrame([
            tuple(getattr(g, f) for f in fields)
            for g in sorted(query.as_games(), key=lambda g: g.start_time)
        ], columns=fields)

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

        return games

    def probability(self, times, labels1, labels2, lines=0, bias=None):
        bias = self.home_field if bias is None else bias
        return super(MeloNFL, self).probability(
            times, labels1, labels2, bias=bias, lines=lines
        )

    def percentile(self, times, labels1, labels2, p=50, bias=None):
        bias = self.home_field if bias is None else bias
        return super(MeloNFL, self).percentile(
            times, labels1, labels2, bias=bias, p=p
        )

    def quantile(self, times, labels1, labels2, q=.5, bias=None):
        bias = self.home_field if bias is None else bias
        return super(MeloNFL, self).quantile(
            times, labels1, labels2, bias=bias, q=q
        )

    def mean(self, times, labels1, labels2, bias=None):
        bias = self.home_field if bias is None else bias
        return super(MeloNFL, self).mean(
            times, labels1, labels2, bias=bias
        )

    def median(self, times, labels1, labels2, bias=None):
        bias = self.home_field if bias is None else bias
        return super(MeloNFL, self).median(
            times, labels1, labels2, bias=bias
        )

    def sample(self, times, labels1, labels2, size=100, bias=None):
        bias = self.home_field if bias is None else bias
        return super(MeloNFL, self).sample(
            times, labels1, labels2, bias=bias, size=size
        )


def calibrated_parameters(mode, evals=100, retrain=False):
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
        hp.uniform('fatigue', 0, 0.5),
    )

    parameters = fmin(objective, space, algo=tpe.suggest, max_evals=200)

    print(parameters)

    with cachefile.open(mode='wb') as f:
        pickle.dump(parameters, f)

    return parameters


retrain = (True if __name__ == '__main__' else False)

nfl_spreads, nfl_totals = [
    MeloNFL(mode, **calibrated_parameters(mode, retrain=retrain))
    for mode in ('spread', 'total')
]
