"""Trains model and exposes predictor class objects"""

from datetime import datetime
import logging
import operator
import os
import pickle

from hyperopt import fmin, hp, tpe, Trials
import matplotlib.pyplot as plt
from melo import Melo
import numpy as np
import pandas as pd

from .data import load_games, update_model
from . import cachedir


class MeloNFL(Melo):
    """
    Generate NFL point-spread or point-total predictions
    using the Margin-dependent Elo (MELO) model.

    """
    def __init__(self, mode, kfactor, regress_coeff,
                 rest_bonus, exp_bonus, weight_qb, burnin=512):

        # hyperparameters
        self.mode = mode
        self.kfactor = kfactor
        self.regress_coeff = regress_coeff
        self.rest_bonus = rest_bonus
        self.exp_bonus = exp_bonus
        self.weight_qb = weight_qb
        self.burnin = burnin

        # model operation mode: "spread" or "total"
        if self.mode not in ["spread", "total"]:
            raise ValueError(
                "Unknown mode; valid options are 'spread' and 'total'")

        # mode-specific training hyperparameters
        self.commutes, self.compare, self.lines = {
            "total": (True, operator.add, np.arange(-0.5, 111.5)),
            "spread": (False, operator.sub, np.arange(-59.5, 60.5)),
        }[mode]

        # pre-process training data
        self.games = self.format_gamedata(load_games(update=False))
        self.teams = np.union1d(self.games.team_home, self.games.team_away)
        self.qbs = np.union1d(self.games.qb_home, self.games.qb_away)

        self.games = self.games[:-256]

        # train the model
        self.train()

        # compute performance metrics
        self.residuals_ = self.residuals(standardize=False)
        self.mean_abs_error = np.mean(np.abs(self.residuals_[burnin:]))
        self.rms_error = np.sqrt(np.mean(self.residuals_[burnin:]**2))

    def regress(self, months):
        """
        Regress ratings to the mean as a function of elapsed time.

        Regression fraction equals self.regress_coeff if months > 3, else 0.

        """
        return self.regress_coeff if months > 3 else 0

    def bias(self, games):
        """
        Circumstantial bias correction factor for each game.

        The bias factor includes two terms: a rest factor which compares the
        rest of each team, and an experience factor which compares the
        experience of each quarterback.

        """
        rest_level_away = 1 - np.exp(-games.rest_days_away / 5.)
        rest_level_home = 1 - np.exp(-games.rest_days_home / 5.)

        rest_bias = self.rest_bonus * self.compare(
            rest_level_away,
            rest_level_home,
        )

        exp_level_away = 1 - np.exp(-games.exp_away / 7.)
        exp_level_home = 1 - np.exp(-games.exp_home / 7.)

        exp_bias = self.exp_bonus * self.compare(
            exp_level_away,
            exp_level_home,
        )

        return rest_bias + exp_bias

    def combine(self, team_rating, qb_rating):
        """
        Combines team and quarterback ratings to form a single rating

        """
        return (1 - self.weight_qb)*team_rating + self.weight_qb*qb_rating

    def format_gamedata(self, games):
        """
        Preprocesses raw game data, returning a model input table.

        This function calculates some new columns and adds them to the
        games table:

                column  description
                  home  home team name joined to home quarterback name
                  away  away team name joined to away quarterback name
        rest_days_home  home team days rested
        rest_days_away  away team days rested
              exp_home  games played by the home quarterback
              exp_away  games played by the away quarterback

        """
        # sort games by date
        games = games.sort_values(by=["datetime", "team_home"])

        # give jacksonville jaguars a single name
        games.replace("JAC", "JAX", inplace=True)

        # give teams which haved moved cities their current name
        games.replace("SD", "LAC", inplace=True)
        games.replace("STL", "LA", inplace=True)

        # game dates for every team
        game_dates = pd.concat([
            games[["datetime", "team_home"]].rename(
                columns={"team_home": "team"}),
            games[["datetime", "team_away"]].rename(
                columns={"team_away": "team"}),
        ]).sort_values("datetime")

        # create home and away label columns
        games["home"] = games["team_home"] + '-' + games["qb_home"]
        games["away"] = games["team_away"] + '-' + games["qb_away"]

        # game dates for every team
        game_dates = pd.concat([
            games[["datetime", "team_home"]].rename(
                columns={"team_home": "team"}),
            games[["datetime", "team_away"]].rename(
                columns={"team_away": "team"}),
        ]).sort_values("datetime")

        # compute days rested
        for team in ["home", "away"]:
            games_prev = game_dates.rename(
                columns={"team": "team_{}".format(team)})

            games_prev["date_{}_prev".format(team)] = games.datetime

            games = pd.merge_asof(
                games, games_prev,
                on="datetime", by="team_{}".format(team),
                allow_exact_matches=False
            )

        # days rested since last game
        one_day = pd.Timedelta("1 days")
        games["rest_days_home"] = \
            (games.datetime - games.date_home_prev) / one_day
        games["rest_days_away"] = \
            (games.datetime - games.date_away_prev) / one_day

        # set days rested to 7 at beginning of the season (default)
        games["rest_days_home"] = games["rest_days_home"].where(
            games.week > 1, other=7)
        games["rest_days_away"] = games["rest_days_away"].where(
            games.week > 1, other=7)

        # games played by each qb
        qb_home = games[["datetime", "qb_home"]].rename(
            columns={"qb_home": "qb"})
        qb_away = games[["datetime", "qb_away"]].rename(
            columns={"qb_away": "qb"})

        qb_exp = pd.concat([qb_home, qb_away]).sort_values("datetime")
        qb_exp["exp"] = qb_exp.groupby("qb").cumcount()

        for team in ["home", "away"]:
            games = games.merge(
                qb_exp.rename(columns={
                    "qb": "qb_{}".format(team),
                    "exp": "exp_{}".format(team)
                }), on=["datetime", "qb_{}".format(team)],
            )

        return games

    def train(self):
        """
        Trains the Margin Elo (MELO) model on the historical game data.

        """
        # instantiate the Melo base class
        super(MeloNFL, self).__init__(
            self.kfactor, lines=self.lines, sigma=1.0,
            regress=self.regress, regress_unit="month",
            commutes=self.commutes, combine=self.combine)

        # train the model
        self.fit(
            self.games.datetime,
            self.games.away,
            self.games.home,
            self.compare(
                self.games.score_away,
                self.games.score_home,
            ),
            self.bias(self.games)
        )

    def visualize_hyperopt(mode, trials, parameters):
        """
        Visualize hyperopt loss minimization.

        """
        plotdir = cachedir / "plots"

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
            ax.set_xlim(min(vals), max(vals))

            if ax.is_first_col():
                ax.set_ylabel("Mean absolute error")

        plotfile = plotdir / "{}_params.pdf".format(mode)
        plt.tight_layout()
        plt.savefig(str(plotfile))

    def rank(self, time, statistic="mean"):
        """
        Modify melo ranking function to only consider teams (ignore qbs).

        """
        return super(MeloNFL, self).rank(
            time, labels=self.teams, statistic=statistic)

    @classmethod
    def from_cache(cls, mode, steps=100, calibrate=False):
        """
        Optimizes the MeloNFL model hyper parameters. Returns cached values
        if calibrate is False and the parameters are cached, otherwise it
        optimizes the parameters and saves them to the cache.

        """
        cachefile = cachedir / "{}.pkl".format(mode)

        if not calibrate and cachefile.exists():
            cache_timestamp = datetime.fromtimestamp(
                os.path.getmtime(cachefile))

            model = pickle.load(cachefile.open(mode="rb"))

            if update_model(cache_timestamp):
                logging.info("updating model")
                load_games(update=True)
                model.train()

                logging.info("caching {} model to {}".format(mode, cachefile))
                pickle.dump(model, cachefile.open(mode="wb"),
                            protocol=pickle.HIGHEST_PROTOCOL)

            return model

        def objective(params):
            return cls(mode, *params).mean_abs_error

        limits = {
            "spread": [
                ("kfactor",       0.1,  0.4),
                ("regress_coeff", 0.1,  0.5),
                ("rest_bonus",    0.0,  0.5),
                ("exp_bonus",     0.0,  0.5),
                ("weight_qb",     0.0,  1.0),
            ],
            "total": [
                ("kfactor",       0.0, 0.3),
                ("regress_coeff", 0.0, 0.5),
                ("rest_bonus",   -0.5, 0.0),
                ("exp_bonus",     0.1, 0.4),
                ("weight_qb",     0.0, 1.0),
            ]
        }

        space = [hp.uniform(*lim) for lim in limits[mode]]

        trials = Trials()

        logging.info("calibrating {} hyperparameters".format(mode))

        load_games(update=True)

        parameters = fmin(objective, space, algo=tpe.suggest, max_evals=steps,
                          trials=trials, show_progressbar=False)

        model = cls(mode, **parameters)

        cls.visualize_hyperopt(mode, trials, parameters)

        cachefile.parent.mkdir(exist_ok=True)

        with cachefile.open(mode="wb") as f:
            logging.info("caching {} model to {}".format(mode, cachefile))
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

        return model
