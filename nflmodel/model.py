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
    def __init__(self, mode, kfactor, regress, rest_bonus):

        # hyperparameters
        self.mode = mode
        self.kfactor = kfactor
        self.regress = lambda months: regress if months > 3 else 0
        self.rest_bonus = rest_bonus

        # model operation mode: 'spread' or 'total'
        if self.mode not in ['spread', 'total']:
            raise ValueError(
                "Unknown mode; valid options are 'spread' and 'total'")

        # mode-specific training hyperparameters
        self.commutes, self.compare, self.lines = {
            'total': (True, operator.add, np.arange(-0.5, 101.5)),
            'spread': (False, operator.sub, np.arange(-59.5, 60.5)),
        }[mode]

        self.games = self.format_gamedata(
            load_games(update=False, rebuild=False))

        # compute optimization loss
        self.loss = self.train(condition_prior=True)

    def train(self, condition_prior=False):
        """
        Retrain the model on the most current game data.

        """
        # instantiate the Melo base class
        super(MeloNFL, self).__init__(
            self.kfactor, lines=self.lines, sigma=1.0,
            regress=self.regress, regress_unit='month',
            commutes=self.commutes)

        # condition the prior using output of the model
        if condition_prior is True:
            self.fit(
                self.games.date,
                self.games.home,
                self.games.away,
                self.compare(
                    self.games.score_home,
                    self.games.score_away,
                ),
                self.games.rest_bonus
            )

            self.prior_rating.update({
                label: self.record[label].rating.mean(axis=0)
                for label in self.labels
            })

        # train the model
        self.fit(
            self.games.date,
            self.games.home,
            self.games.away,
            self.compare(
                self.games.score_home,
                self.games.score_away,
            ),
            self.games.rest_bonus
        )

        # compute mean absolute error for calibration
        burnin = 256
        residuals = self.residuals()[burnin:]

        return np.sqrt(np.square(residuals).mean())

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

        # calculate rest days
        for team in ['home', 'away']:
            games_prev = game_dates.rename(
                columns={'team': 'team_{}'.format(team)})

            games_prev['date_{}_prev'.format(team)] = games.date

            games = pd.merge_asof(
                games, games_prev,
                on='date', by='team_{}'.format(team),
                allow_exact_matches=False
            )

        # add rest days columns
        ten_days = pd.Timedelta('10 days')
        games['home_rested'] = (games.date - games.date_home_prev) > ten_days
        games['away_rested'] = (games.date - games.date_away_prev) > ten_days

        # add bias factors
        games['rest_bonus'] = self.rest_bonus * self.compare(
            games.home_rested.astype(int), games.away_rested.astype(int))

        # create home and away label columns
        games['home'] = games['team_home'] + '|' + games['qb_home']
        games['away'] = games['team_away'] + '|' + games['qb_away']

        # reorder columns and drop uncecessary fields
        games = games[[
            'date',
            'home',
            'away',
            'score_home',
            'score_away',
            'rest_bonus',
        ]]

        return games

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
            return cls(mode, *params).loss

        space = (
            hp.uniform('kfactor', 0.0, 0.5),
            hp.uniform('regress', 0.0, 1.0),
            hp.uniform('rest_bonus', -0.5, 0.5),
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
            ncols=4, figsize=(12, 3), sharey=True)

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
