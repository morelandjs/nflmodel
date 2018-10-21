melo-nfl
========

*NFL ratings and predictions*

This module trains the margin-dependent Elo (melo) model on NFL game data. It creates two trained melo class objects, nfl_spreads and nfl_totals, which may be used to predict NFL point spreads and point totals. It also provides an interface to optimize the hyperparameters of the model.

Usage
-----
```
from datetime import datetime
from melo_nfl import nfl_spreads

ranked_teams = nfl_spreads.rank(datetime.today(), statistic='median')
```