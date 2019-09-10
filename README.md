NFL Model
=========

*NFL ratings and predictions*

This package trains the margin-dependent Elo model (MELO) on NFL game data.

Installation
------------

```
git clone git@github.com:morelandjs/nfl-model.git && cd nfl-model
pip install -r requirements.txt
```

Quick Start
-----------
First, populate the database
```
nflmodel update
```
Then train the model on the dataset
```
nflmodel train --steps 100
```
Finally, compute point spread and point total predictions
```
nflmodel predict 2018-12-30 TB LAR
nflmodel predict 2019-09-05 CLE TEN
```
The model also ranks teams by their mean expected point spread (and point total) against a league average opponent.
```
nflmodel rank
```
Additionally, you can validate the model predictions by calling
```
nflmodel validate
```
which generates two figures, `validate_spread.pdf` and `validate_total.pdf`, visualizing the distribution of prediction residuals and quantiles.
