NFL Model
=========

*NFL ratings and predictions*

This module trains the margin-dependent Elo (melo) model on NFL game data.

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
nflmodel update
```
Then train the model on the dataset
```
nflmodel train --steps 200
```
Finally, compute point spread and point total predictions
```
nflmodel predict 2018-12-30 TB LAR
nflmodel predict 2019-09-05 CLE TEN
```
