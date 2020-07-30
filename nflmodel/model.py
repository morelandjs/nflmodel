"""Trains team model and exposes predictor class objects"""
from functools import partial
import logging
import operator
import pickle

from armchair_analysis.game_data import game_data
from elora import Elora
from hyperopt import fmin, hp, tpe, Trials
import matplotlib.pyplot as plt
import numpy as np
from qbmodel.model import EloraQB

from . import cachedir


class EloraTeam(Elora):
    """
    Generate NFL team point-spread or point-total predictions
    using the Elo regressor algorithm (elora)

    """
    def __init__(self, mode, kfactor, regress_frac, rest_coeff,
                 scale=1, burnin=512):

        # hyperparameters
        self.mode = mode
        self.kfactor = kfactor
        self.regress_frac = regress_frac
        self.rest_coeff = rest_coeff
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
        self.games = game_data.dataframe.sort_values(
            by=['date', 'team_away', 'team_home'])
        self.teams = np.union1d(self.games.team_home, self.games.team_away)

        # train the model
        self.train()

        # compute performance metrics
        self.residuals_ = self.residuals(standardize=False)
        self.mean_abs_error = np.mean(np.abs(self.residuals_[burnin:]))
        self.rms_error = np.sqrt(np.mean(self.residuals_[burnin:]**2))

        # components for binary cross entropy loss
        tiny = 1e-5
        yp = np.clip(
            self.pdf(
                self.examples.value,
                self.examples.time,
                self.examples.label1,
                self.examples.label2,
                self.examples.bias),
            tiny, 1 - tiny)

        # binary cross entropy loss
        self.log_loss = -np.log(yp).mean()

    def regression_coeff(self, elapsed_time):
        """
        Regress ratings to the mean as a function of elapsed time.

        Regression fraction equals:

            self.regress_frac if elapsed_days > 90, else 1

        """
        elapsed_days = elapsed_time / np.timedelta64(1, 'D')

        tiny= 1e-6
        arg = np.clip(self.regress_frac, tiny, 1 - tiny)
        factor = np.log(self.regress_frac)/365.

        return np.exp(factor * elapsed_days)

    def bias(self, games):
        """
        Circumstantial bias factors which apply to a single game.

        """
        rest_adv = self.rest_coeff * self.compare(
            games.tm_rest_days_away, games.tm_rest_days_home)

        # TODO finish this!

        #qb_model = EloraQB.from_cache(self.mode)

        #qb_pts = qb_model.mean(
        #    games.date, games.qb_away, games.qb_home)

        #qb_prev_pts = qb_model.mean(
        #    games.date, games.qb_prev_away, games.qb_prev_home)

        #qb_chg_bias = self.qb_chg_adj * (qb_pts - qb_prev_pts)

        return rest_adv #+ qb_chg_bias

    def train(self):
        """
        Trains the Margin Elo (MELO) model on the historical game data.

        """
        super(EloraTeam, self).__init__(
            self.kfactor,
            scale=self.scale,
            commutes=self.commutes)

        values = self.compare(
                self.games.tm_pts_away,
                self.games.tm_pts_home)
        #plt.hist(values, bins=np.arange(-40.5, 41.5))
        #plt.show()
        #quit()

        self.fit(
            self.games.date,
            self.games.team_away,
            self.games.team_home,
            self.compare(
                self.games.tm_pts_away,
                self.games.tm_pts_home),
            biases=self.bias(self.games)
        )

    def visualize_hyperopt(mode, trials, parameters, measure, filename):
        """
        Visualize hyperopt loss minimization.

        """
        plotdir = cachedir / "plots"
        nparam = len(parameters)

        if not plotdir.exists():
            plotdir.mkdir()

        fig, axes = plt.subplots(
            ncols=nparam, figsize=(4*nparam, 3), sharey=True)

        axes = np.array(axes, ndmin=1)

        losses = trials.losses()

        for ax, (label, vals) in zip(axes.flat, trials.vals.items()):
            c = plt.cm.coolwarm(np.linspace(0, 1, len(vals)))
            ax.scatter(vals, losses, c=c)
            ax.axvline(parameters[label], color='k')
            ax.set_xlabel(label)
            ax.set_xlim(min(vals), max(vals))

            if ax.is_first_col():
                ax.set_ylabel(measure)

        plotfile = plotdir / filename
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
        Optimizes the EloraTeam model hyper parameters. Returns cached values
        if calibrate is False and the parameters are cached, otherwise it
        optimizes the parameters and saves them to the cache.

        """
        cachefile = cachedir / "{}.pkl".format(mode)

        if not calibrate and cachefile.exists():
            return pickle.load(cachefile.open(mode="rb"))

        limits = {
            "spread": [
                ("kfactor",     0.02, 0.12),
                ("regress_frac", 0.0,  1.0),
                ("rest_coeff", -0.50, 0.75)],
            "total": [
                ("kfactor",      0.01, 0.07),
                ("regress_frac",  0.0,  1.0),
                ("rest_coeff",   -0.5,  0.5)]}

        space = [hp.uniform(*lim) for lim in limits[mode]]

        trials = Trials()

        logging.info("calibrating {} kfactor and regress_frac "
                     "hyperparameters".format(mode))

        def calibrate_loc_params(params):
            return cls(mode, *params).mean_abs_error

        parameters = fmin(
            calibrate_loc_params, space, algo=tpe.suggest,
            max_evals=steps, trials=trials, show_progressbar=False)

        kfactor = parameters['kfactor']
        regress_frac = parameters['regress_frac']
        rest_coeff = parameters['rest_coeff']

        cls.visualize_hyperopt(
            mode, trials, parameters,
            'Mean absolute error', f'{mode}_loc_params.pdf')

        limits = {
            "spread": [
                ("scale", 5., 25.)],
            "total": [
                ("scale", 5., 25.)]}

        space = [hp.uniform(*lim) for lim in limits[mode]]

        trials = Trials()

        logging.info("calibrating {} scale hyperparameter".format(mode))

        def calibrate_scale_param(params):
            return cls(
                mode, kfactor, regress_frac, rest_coeff,
                scale=params[0]).log_loss

        parameters = fmin(
            calibrate_scale_param, space, algo=tpe.suggest,
            max_evals=steps, trials=trials, show_progressbar=False)

        scale = parameters['scale']

        cls.visualize_hyperopt(
            mode, trials, parameters,
            'Log loss', f'{mode}_scale_params.pdf')

        model = cls(
            mode, kfactor, regress_frac, rest_coeff,
            scale=scale)

        cachefile.parent.mkdir(exist_ok=True)

        with cachefile.open(mode="wb") as f:
            logging.info("caching {} model to {}".format(mode, cachefile))
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

        return model


if __name__ == '__main__':

    spreads = EloraTeam.from_cache('spread', calibrate=False)
    games = spreads.games

    spread_pred = spreads.mean(games.date, games.team_away, games.team_home)
    spread_vegas = -games.spread_vegas

    plt.plot(spread_pred, spread_vegas, 'o')
    plt.show()
