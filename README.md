NFL Model
=========

*NFL ratings and predictions*

This package trains the margin-dependent Elo model (MELO) on NFL game data.

Installation
------------

```
git clone git@github.com:morelandjs/nfl-model.git && cd nfl-model
pip install .
```

Quick Start
-----------
First, populate the database
```
> nflmodel update
```
Then train the model on the dataset (this will take a few minutes)
```
> nflmodel calibrate --steps 100
```
Once trained, the model can forecast point spread and point total statistics for the upcoming week
```
> nflmodel forecast

           favorite underdog  win prob  spread  total
date                                                 
2019-11-24     @CLE      MIA      0.76   -13.1   43.2
2019-11-24      @NE      DAL      0.78   -10.2   44.4
2019-11-24      @NO      CAR      0.74    -8.7   49.7
2019-11-24      @SF       GB      0.66    -7.5   49.2
2019-11-24     @CHI      NYG      0.77    -6.9   46.1
2019-11-24     @ATL       TB      0.71    -5.7   50.7
2019-11-25      BAL      @LA      0.61    -5.2   48.1
2019-11-24      PIT     @CIN      0.67    -4.3   41.8
2019-11-21     @HOU      IND      0.67    -3.2   48.3
2019-11-24     @BUF      DEN      0.63    -3.0   41.6
2019-11-24      OAK     @NYJ      0.60    -2.8   46.4
2019-11-24      SEA     @PHI      0.64    -2.4   49.6
2019-11-24     @TEN      JAX      0.54    -2.1   45.0
2019-11-24      DET     @WAS      0.51    -0.5   46.4

```
The model can also rank teams by their expected performance against a league average opponent
```
> nflmodel rank

[INFO][nflmodel] Expected performance against a league average opponent
   win rank  win prob spread rank  spread total rank  total
1        NE      0.81          NE    11.3         TB   50.8
2       BAL      0.81         BAL     9.8         KC   49.7
3       SEA      0.77          SF     7.0        NYG   48.7
4        SF      0.76          NO     6.1        BAL   48.6
5        NO      0.75          LA     5.9        SEA   48.6
6       MIN      0.71         MIN     5.7        ARI   48.2
7       HOU      0.70         DAL     5.1        CAR   47.0
8        GB      0.68          KC     4.1        DET   46.8
9        LA      0.66         HOU     3.8         SF   46.3
10       KC      0.65         SEA     3.8        OAK   46.2
11      DAL      0.61          GB     3.1        DAL   46.2
12      PHI      0.58         PHI     2.9         GB   46.2
13      PIT      0.58         BUF     2.4        MIA   45.9
14      CAR      0.58         LAC     2.1        PHI   45.9
15      OAK      0.57         CHI     2.1        NYJ   45.7
16      JAX      0.57         PIT     2.1        ATL   45.4
17      BUF      0.56         ATL     1.4        HOU   45.3
18      CLE      0.54         IND     1.2         NO   45.2
19      IND      0.54         CAR     1.1        MIN   45.0
20      TEN      0.54         JAX     1.1        CIN   44.6
21      CHI      0.53         DEN     1.0         LA   44.6
22      ATL      0.52         CLE     0.6        JAX   44.5
23      DEN      0.50         DET    -1.5        TEN   44.4
24      LAC      0.47         TEN    -1.6        CLE   44.2
25      NYJ      0.40         ARI    -2.4        IND   44.0
26      ARI      0.40          TB    -2.5        BUF   43.7
27      DET      0.38         OAK    -2.6        PIT   43.5
28       TB      0.38         NYJ    -3.4        WAS   43.5
29      CIN      0.35         NYG    -3.8        DEN   43.4
30      MIA      0.33         CIN    -3.9        LAC   42.9
31      NYG      0.31         WAS    -5.6         NE   42.6
32      WAS      0.30         MIA    -6.9        CHI   40.5 
```
Additionally, you can validate the model predictions by calling
```
nflmodel validate
```
which generates two figures, `validate_spread.pdf` and `validate_total.pdf`, visualizing the distribution of prediction residuals and quantiles.
