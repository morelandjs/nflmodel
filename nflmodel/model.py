"""Trains model and exposes predictor class objects"""
from functools import partial
import logging
import operator
import pickle

from armchair_analysis.game_data import game_data
from elora import Elora
from hyperopt import fmin, hp, tpe, Trials
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import cachedir


class EloraNFL(Elora):
    """
    Generate NFL point-spread or point-total predictions
    using the Elo regressor algorithm (elora)

    """
    def __init__(self, mode, kfactor, regress_frac, scale=13, burnin=512):

        # hyperparameters
        self.mode = mode
        self.kfactor = kfactor
        self.regress_frac = regress_frac
        self.scale = scale
        self.burnin = burnin

        # model operation mode: "spread" or "total"
        if self.mode not in ["spread", "total"]:
            raise ValueError(
                "Unknown mode; valid options are 'spread' and 'total'")

        # mode-specific training hyperparameters
        self.commutes, self.compare = {
            "total": (True, operator.add),
            "spread": (False, operator.sub),
        }[mode]

        # pre-process training data
        self.games = self.format_gamedata(game_data.dataframe)
        self.teams = np.union1d(self.games.team_home, self.games.team_away)

        # train the model
        self.train()

        # compute performance metrics
        self.residuals_ = self.residuals(standardize=False)
        self.mean_abs_error = np.mean(np.abs(self.residuals_[burnin:]))
        self.rms_error = np.sqrt(np.mean(self.residuals_[burnin:]**2))

        # components for binary cross entropy loss
        #y = self.examples.value > self.mean_value
        #tiny = 1e-5
        #yp = np.clip(
        #    self.sf(
        #        self.mean_value,
        #        self.examples.time,
        #        self.examples.label1,
        #        self.examples.label2,
        #        self.examples.bias),
        #    tiny, 1 - tiny)

        # binary cross entropy loss
        #self.log_loss = -np.mean(y*np.log(yp) + (1 - y)*np.log(1 - yp))

    def regression_coeff(self, elapsed_time):
        """
        Regress ratings to the mean as a function of elapsed time.

        Regression fraction equals:

            self.regress_frac if elapsed_days > 90, else 1

        """
        elapsed_months = elapsed_time / np.timedelta64(1, 'D')

        return self.regress_frac if elapsed_months > 90 else 1

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
        games = games.sort_values(by=["date", "team_home"])

        # give jacksonville jaguars a single name
        games.replace("JAC", "JAX", inplace=True)

        # give teams which haved moved cities their current name
        games.replace("SD", "LAC", inplace=True)
        games.replace("STL", "LA", inplace=True)

        # game dates for every team
        game_dates = pd.concat([
            games[["date", "team_home"]].rename(
                columns={"team_home": "team"}),
            games[["date", "team_away"]].rename(
                columns={"team_away": "team"}),
        ]).sort_values("date")

        # game dates for every team
        game_dates = pd.concat([
            games[["date", "team_home"]].rename(
                columns={"team_home": "team"}),
            games[["date", "team_away"]].rename(
                columns={"team_away": "team"}),
        ]).sort_values("date")

        # compute days rested
        for team in ["home", "away"]:
            games_prev = game_dates.rename(
                columns={"team": "team_{}".format(team)})

            games_prev["date_{}_prev".format(team)] = games.date

            games = pd.merge_asof(
                games, games_prev,
                on="date", by="team_{}".format(team),
                allow_exact_matches=False
            )

        # days rested since last game
        one_day = pd.Timedelta("1 days")
        games["rest_days_home"] = np.clip(
            (games.date - games.date_home_prev) / one_day, 3, 16).fillna(7)
        games["rest_days_away"] = np.clip(
            (games.date - games.date_away_prev) / one_day, 3, 16).fillna(7)

        return games

    def train(self):
        """
        Trains the Margin Elo (MELO) model on the historical game data.

        """
        super(EloraNFL, self).__init__(
            self.kfactor,
            scale=self.scale,
            commutes=self.commutes)

        self.fit(
            self.games.date,
            self.games.team_away,
            self.games.team_home,
            self.compare(
                self.games.tm_pts_away,
                self.games.tm_pts_home))

    def visualize_hyperopt(mode, trials, parameters):
        """
        Visualize hyperopt loss minimization.

        """
        plotdir = cachedir / "plots"

        if not plotdir.exists():
            plotdir.mkdir()

        fig, axes = plt.subplots(
            ncols=2, figsize=(12, 3), sharey=True)

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

    def rank(self, time, order_by='mean', reverse=False):
        """
        Rank labels at specified 'time' according to 'order_by'
        comparison value.

        Args:
            time (np.datetime64): time to compute the ranking
            order_by (string, optional): options are 'mean' and 'win_prob'
                (default is 'mean')
            reverse (bool, optional): reverses the ranking order if true
                (default is False)

        Returns:
            ranked_labels (list of tuple): list of (label, value) pairs

        """
        value = {
            "win prob": partial(self.sf, 0),
            "mean": self.mean,
         }.get(order_by, None)

        if value is None:
            raise ValueError("no such comparison function")

        ranked_list = [
            (label, value(time, label, None, biases=-self.commutator))
            for label in self.labels]

        return sorted(ranked_list, key=lambda v: v[1], reverse=reverse)

    @classmethod
    def from_cache(cls, mode, steps=100, calibrate=False):
        """
        Optimizes the EloraNFL model hyper parameters. Returns cached values
        if calibrate is False and the parameters are cached, otherwise it
        optimizes the parameters and saves them to the cache.

        """
        cachefile = cachedir / "{}.pkl".format(mode)

        if not calibrate and cachefile.exists():
            return pickle.load(cachefile.open(mode="rb"))

        def evaluation_function(params):
            return cls(mode, *params).mean_abs_error

        limits = {
            "spread": [
                ("kfactor",     0.02, 0.12),
                ("regress_frac", 0.0,  1.0),
            ],
            "total": [
                ("kfactor",      0.01, 0.07),
                ("regress_frac",  0.0, 1.0)]}

        space = [hp.uniform(*lim) for lim in limits[mode]]

        trials = Trials()

        logging.info("calibrating {} hyperparameters".format(mode))

        parameters = fmin(evaluation_function, space, algo=tpe.suggest,
                          max_evals=steps, trials=trials,
                          show_progressbar=False)

        model = cls(mode, **parameters)

        cls.visualize_hyperopt(mode, trials, parameters)

        cachefile.parent.mkdir(exist_ok=True)

        with cachefile.open(mode="wb") as f:
            logging.info("caching {} model to {}".format(mode, cachefile))
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

        return model


if __name__ == '__main__':

    spreads = EloraNFL.from_cache('spread', calibrate=False)
    games = spreads.games

    spread_pred = spreads.mean(games.date, games.team_away, games.team_home)
    spread_vegas = -games.spread_vegas

    plt.plot(spread_pred, spread_vegas, 'o')
    plt.show()
