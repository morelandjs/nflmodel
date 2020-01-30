#!/usr/bin/env python3

import argparse
import logging
import sys

from nflmodel import model
import numpy as np
import pandas as pd


def vegas_lines(filepath):
    """
    Vegas game lines and outcomes.

    """
    games = pd.read_csv(filepath)

    alias = {'ARZ': 'ARI', 'LAR': 'LA', 'JAC': 'JAX'}
    games.replace(alias, inplace=True)

    return games


def simulate_betslip(threshold, burnin=512):
    """
    Simulate betting on games using a certain decision threshold.

    """
    # load model predictions for each game
    spread_model = model.MeloNFL.from_cache('spread')
    games = spread_model.games

    # create date column
    games.loc[:]["date"] = games["datetime"].dt.date.astype(str)

    # add model residuals
    games["model_residual"] = spread_model.residuals_

    # add column of vegas lines for each game
    games = games.merge(
        vegas_lines("nfl_lines_2009-2019.csv"),
        how='inner',
    )[burnin:]

    # observed home point spread
    spread_home_obs = games.score_away - games.score_home

    # add vegas residuals
    games["vegas_residual"] = games.spread_home_vegas - spread_home_obs

    # predict home and away team win probabilities
    games["prob_away_model"] = spread_model.probability(
        games.date, games.away, games.home,
        lines=games.spread_home_vegas[:, np.newaxis]
    )

    games['prob_home_model'] = 1 - games.prob_away_model

    # bet on teams if they exceed the threshold cover probability
    games["bet_away"] = games.prob_away_model > threshold
    games["bet_home"] = games.prob_home_model > threshold

    # select games which fulfill betting criteria
    games = games[games.bet_home | games.bet_away]
    games["bet"] = np.where(games.bet_home, "home", "away")

    # record bet results
    games["spread_home"] = games.score_away - games.score_home
    games["cover_home"] = games.spread_home < games.spread_home_vegas
    games["cover_away"] = ~games.cover_home

    # bets won
    games["won"] = (
        (games.bet_home & games.cover_home) |
        (games.bet_away & games.cover_away)
    )

    # prune columns
    games = games[[
        "date",
        "team_home",
        "team_away",
        "spread_home_vegas",
        "bet",
        "won",
        "vegas_residual",
        "model_residual",
    ]]

    # cleanup output
    games.rename(
        columns={
            "team_home": "home",
            "team_away": "away",
            "spread_home_vegas": "home_line",
        }, inplace=True
    )

    return games


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="\n".join([
            "Simulate NFL betting strategy using a pre-determined decision threshold",
        ]), formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "threshold",
        type=float,
        help="cover likelihood used to set decision threshold",
    )

    args = parser.parse_args()
    kwargs = vars(args)

    betslip = simulate_betslip(**kwargs)

    if betslip.empty:
        sys.exit("no games meet the specified betting threshold")

    bets_won = np.sum(betslip.won)
    bets_lost = np.sum(~betslip.won)
    bets_made = len(betslip)
    perc = bets_won / float(bets_made)

    bets_won_rand = np.random.binomial(bets_made, 0.5, size=10000)
    perc_rand_iqr = np.quantile(bets_won_rand / bets_made, q=[.1, .9])

    vegas_mae = np.abs(betslip.vegas_residual).mean()
    model_mae = np.abs(betslip.model_residual).mean()

    logging.info("{} won, {} lost".format(bets_won, bets_lost))
    logging.info("{:.2f}% correct model; {:.2f}-{:.2f}% random chance"
                 .format(perc, *perc_rand_iqr))
    logging.info("Vegas mean abs error: {:.2f} pts".format(vegas_mae))
    logging.info("Model mean abs error: {:.2f} pts".format(model_mae))
