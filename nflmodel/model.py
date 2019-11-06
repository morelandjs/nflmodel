"""Trains model and exposes predictor class objects"""

import logging
import operator

from hyperopt import fmin, hp, tpe, Trials
from joblib import dump, load
import matplotlib.pyplot as plt
from melo import Melo
import numpy as np
import pandas as pd

from .data import load_games
from . import cachedir


class MeloNFL(Melo):
    """
    Generate NFL point-spread or point-total predictions
    using the Margin-dependent Elo (MELO) model.

    """
    def __init__(self, mode, kfactor, regress, rest_bonus, exp_bonus, weight_qb):

        # hyperparameters
        self.mode = mode
        self.kfactor = kfactor
        self.regress = lambda months: regress if months > 3 else 0
        self.rest_bonus = rest_bonus
        self.exp_bonus = exp_bonus
        self.weight_qb = weight_qb

        # model operation mode: 'spread' or 'total'
        if self.mode not in ['spread', 'total']:
            raise ValueError(
                "Unknown mode; valid options are 'spread' and 'total'")

        # mode-specific training hyperparameters
        self.commutes, self.compare, self.lines = {
            'total': (True, operator.add, np.arange(-0.5, 101.5)),
            'spread': (False, operator.sub, np.arange(-59.5, 60.5)),
        }[mode]

        # pre-process training data
        self.games = self.format_gamedata(
            load_games(update=False, rebuild=False))

        # compute optimization rms_error
        self.rms_error = self.train(condition_prior=True)

    def rest(self, games):
        """
        Equal to rest_bonus if team is coming off bye week, 0 otherwise

        """
        # game dates for every team
        game_dates = pd.concat([
            games[['date', 'team_home']].rename(
                columns={'team_home': 'team'}),
            games[['date', 'team_away']].rename(
                columns={'team_away': 'team'}),
        ]).sort_values('date')

        # compute days rested
        for team in ['home', 'away']:
            games_prev = game_dates.rename(
                columns={'team': 'team_{}'.format(team)})

            games_prev['date_{}_prev'.format(team)] = games.date

            games = pd.merge_asof(
                games, games_prev,
                on='date', by='team_{}'.format(team),
                allow_exact_matches=False
            )

        # true if team is comming off bye week, false otherwise
        ten_days = pd.Timedelta('10 days')
        games['home_rested'] = (games.date - games.date_home_prev) > ten_days
        games['away_rested'] = (games.date - games.date_away_prev) > ten_days

        # return rest bias correction
        return self.rest_bonus * self.compare(
            games.home_rested.astype(int),
            games.away_rested.astype(int),
        )

    def experience(self, games):
        """
        Equal to experience_factor * (1 - 0.5**games_played)

        """
        qb_games = pd.DataFrame(
            np.vstack((games.qb_home, games.qb_away)).ravel('F'),
            columns=['qb']
        ).groupby('qb').cumcount(ascending=True)

        qb_games = pd.DataFrame(
            qb_games.values.reshape(-1, 2),
            columns=['home_played', 'away_played'],
        )

        home_exp = 1 - np.exp(-qb_games.home_played / 16.)
        away_exp = 1 - np.exp(-qb_games.away_played / 16.)

        return self.exp_bonus * self.compare(home_exp, away_exp)

    def format_gamedata(self, games):
        """
        Preprocesses raw game data, returning a model input table.

        """
        # convert dates to datetime type
        games['date'] = pd.to_datetime(games.date)

        # sort games by date
        games = games.sort_values('date')

        # give jacksonville jaguars a single name
        games.replace('JAC', 'JAX', inplace=True)

        # give teams which haved moved cities their current name
        games.replace('SD', 'LAC', inplace=True)
        games.replace('STL', 'LA', inplace=True)

        # game dates for every team
        game_dates = pd.concat([
            games[['date', 'team_home']].rename(
                columns={'team_home': 'team'}),
            games[['date', 'team_away']].rename(
                columns={'team_away': 'team'}),
        ]).sort_values('date')

        # create home and away label columns
        games['home'] = games['team_home'] + '-' + games['qb_home']
        games['away'] = games['team_away'] + '-' + games['qb_away']

        # compute circumstantial bias factors
        games['bias'] = self.rest(games) + self.experience(games)

        # reorder columns and drop uncecessary fields
        games = games[[
            'date',
            'home',
            'away',
            'team_home',
            'team_away',
            'qb_home',
            'qb_away',
            'score_home',
            'score_away',
            'bias',
        ]]

        return games

    def train(self, condition_prior=False):
        """
        Retrain the model on the most current game data.

        """
        # rating is weighted average of team and qb ratings
        def combine(team_rating, qb_rating):
            return (1 - self.weight_qb)*team_rating + self.weight_qb*qb_rating

        # instantiate the Melo base class
        super(MeloNFL, self).__init__(
            self.kfactor, lines=self.lines, sigma=1.0,
            regress=self.regress, regress_unit='month',
            commutes=self.commutes, combine=combine)

        # train the model
        self.fit(
            self.games.date,
            self.games.home,
            self.games.away,
            self.compare(
                self.games.score_home,
                self.games.score_away,
            ),
            self.games.bias
        )

        # compute mean absolute error for calibration
        burnin = 256
        residuals = self.residuals()[burnin:]

        return np.sqrt(np.square(residuals).mean())

    @classmethod
    def from_cache(cls, mode, steps=100, retrain=False):
        """
        Optimizes the MeloNFL model hyper parameters. Returns cached values
        if retrain is False and the parameters are cached, otherwise it
        optimizes the parameters and saves them to the cache.

        """
        cachefile = cachedir / '{}.pkl'.format(mode)

        if not retrain and cachefile.exists():
            logging.debug('loading {} model from cache', mode)
            return load(cachefile)

        def objective(params):
            return cls(mode, *params).rms_error

        space = (
            hp.uniform('kfactor', 0.1, 0.4),
            hp.uniform('regress', 0.0, 1.0),
            hp.uniform('rest_bonus', -0.5, 0.5),
            hp.uniform('exp_bonus', -0.3, 0.3),
            hp.uniform('weight_qb', 0.0, 1.0),
        )

        trials = Trials()

        logging.info('optimizing {} hyperparameters'.format(mode))

        parameters = fmin(objective, space, algo=tpe.suggest,
                          max_evals=steps, trials=trials,
                          show_progressbar=False)

        plotdir = cachedir / 'plots'

        if not plotdir.exists():
            plotdir.mkdir()

        fig, axes = plt.subplots(
            ncols=5, figsize=(12, 3), sharey=True)

        losses = trials.losses()

        for ax, (label, vals) in zip(axes.flat, trials.vals.items()):
            c = plt.cm.coolwarm(np.linspace(0, 1, len(vals)))
            ax.scatter(vals, losses, c=c)
            ax.axvline(parameters[label], color='k')
            ax.set_xlabel(label)
            if ax.is_first_col():
                ax.set_ylabel('Mean absolute error')

        plotfile = plotdir / '{}_params.pdf'.format(mode)
        plt.tight_layout()
        plt.savefig(str(plotfile))

        model = cls(mode, **parameters)

        logging.info('writing cache file %s', cachefile)

        if not cachefile.parent.exists():
            cachefile.parent.mkdir()

        dump(model, cachefile, protocol=2)

        return model
