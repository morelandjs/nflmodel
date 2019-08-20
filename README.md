NFL Model (under construction)
==============================

*NFL ratings and predictions*

This module trains the margin-dependent Elo (melo) model on NFL game data. It creates two trained melo class objects, nfl_spreads and nfl_totals, which may be used to predict NFL point spreads and point totals. It also provides an interface to optimize the hyperparameters of the model.

Installation
------------

```
git clone git@github.com:morelandjs/melo-nfl.git && cd melo-nfl
pip install .
```

Quick Start
-----------
First, populate the database
```
nflmodel update --refresh
```
Then train the model on the dataset
```
nflmodel train --steps 200
```
Finally, import the trained model as a Python package to generate predictions
```
from nflmodel.model import nfl_spreads
from datetime import datetime

ranked_teams = nfl_spreads.rank(datetime.today(), statistic='median')
```
