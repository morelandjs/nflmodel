"""Simulate the profitability of the model as a bettor"""

import logging

from armchair_analysis.game_data import game_data
import matplotlib.pyplot as plt

from nflmodel import model


def bet_spread(mode, burnin=512, threshold=0.55, odds=110):
    """
    Predict game outcomes, place bets, and keep a running total of
    profit.

    """
    nfl_model = model.EloraTeam.from_cache(mode, calibrate=False)

    games = nfl_model.games.iloc[burnin:]

    away_cover_prob = nfl_model.sf(
        -games.spread_vegas,
        games.date,
        games.team_away,
        games.team_home)

    home_cover_prob = 1 - away_cover_prob

    bet_home = home_cover_prob > threshold
    bet_away = away_cover_prob > threshold

    home_spread = games.tm_pts_home - games.tm_pts_away
    cover_home = home_spread - games.spread_vegas > 0
    cover_away = home_spread - games.spread_vegas < 0

    home_correct = cover_home[bet_home].sum()
    home_incorrect = cover_away[bet_home].sum()

    away_correct = cover_away[bet_away].sum()
    away_incorrect = cover_home[bet_away].sum()

    correct = home_correct + away_correct
    incorrect = home_incorrect + away_incorrect

    roi = (100 * correct - odds * incorrect) / (odds * (correct + incorrect))
    roi_report = '{:.3f}'.format(100*roi)

    logging.info(f'{correct} correct, {incorrect} incorrect')
    logging.info(f'{roi_report}% return per bet')


if __name__ == '__main__':
    simulate_bets('spread')
