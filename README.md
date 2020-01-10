NFL Model
=========

*NFL ratings and predictions*

This package trains the margin-dependent Elo model (MELO) on NFL game data.

Installation
------------

```
git clone https://github.com/morelandjs/nfl-model.git && cd nfl-model
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

[INFO][nflmodel] Forecast for season 2019 week 17

           favorite underdog  win prob  spread  total
date                                                 
2019-12-29      @NE      MIA      0.90   -17.1   43.3
2019-12-29      @LA      ARI      0.70   -13.1   46.8
2019-12-29     @DEN      OAK      0.73   -11.0   42.4
2019-12-29     @BUF      NYJ      0.72   -10.8   40.4
2019-12-29     @DAL      WAS      0.75    -9.3   46.9
2019-12-29     @MIN      CHI      0.69    -9.1   34.5
2019-12-29     @BAL      PIT      0.88    -9.1   42.9
2019-12-29       NO     @CAR      0.78    -8.2   53.9
2019-12-29       GB     @DET      0.83    -7.7   44.2
2019-12-29      @KC      LAC      0.89    -7.3   43.9
2019-12-29      PHI     @NYG      0.66    -5.9   49.1
2019-12-29      CLE     @CIN      0.65    -4.0   45.2
2019-12-29     @HOU      TEN      0.74    -3.4   47.7
2019-12-29     @JAX      IND      0.54    -2.0   43.7
2019-12-29      @TB      ATL      0.52    -1.4   51.4
2019-12-29     @SEA       SF      0.51    -0.6   50.8 

*win probability and spread are for the favored team

```
The model can also rank teams by their expected performance against a league average opponent
```
> nflmodel rank

[INFO][nflmodel] Rankings as of 2020-01-09T21:09:54

       win prob        spread         total
rank                                       
1      SF  0.78  │   NO  -8.1  │   TB  50.4
2      NO  0.78  │   KC  -8.0  │  MIA  48.6
3      KC  0.77  │  BAL  -7.4  │  NYG  48.3
4      GB  0.75  │   NE  -7.1  │  CAR  48.3
5     BAL  0.75  │   SF  -6.3  │   KC  48.2
6      NE  0.68  │  DAL  -4.9  │   NO  47.9
7     SEA  0.65  │   LA  -4.5  │   SF  47.4
8      LA  0.63  │   GB  -3.6  │  BAL  47.3
9     TEN  0.62  │  TEN  -3.0  │  ARI  47.2
10    ATL  0.62  │  PHI  -2.9  │  SEA  47.0
11    HOU  0.61  │  MIN  -2.9  │   LA  46.6
12    PHI  0.61  │  ATL  -2.6  │  ATL  46.4
13    DEN  0.61  │  SEA  -1.7  │  DET  46.2
14    CHI  0.57  │  HOU  -1.3  │  DAL  46.2
15    MIN  0.55  │  DEN  -1.1  │  HOU  46.2
16    PIT  0.55  │  PIT  -0.9  │  CLE  46.1
17    NYJ  0.52  │  CHI  -0.6  │  IND  45.9
18    DAL  0.52  │   TB  -0.4  │  TEN  45.8
19    BUF  0.48  │  BUF  -0.2  │  PHI  45.6
20    JAX  0.47  │  LAC   0.5  │  CIN  45.3
21     TB  0.47  │  IND   1.5  │  OAK  45.1
22    ARI  0.44  │  ARI   1.6  │  WAS  45.0
23    MIA  0.42  │  JAX   2.1  │  MIN  44.6
24    OAK  0.41  │  NYJ   2.1  │  LAC  44.4
25    IND  0.40  │  CLE   2.6  │   GB  44.4
26    CLE  0.39  │  DET   3.0  │  JAX  44.1
27    CAR  0.38  │  CAR   3.9  │  NYJ  43.5
28    LAC  0.34  │  CIN   4.0  │   NE  43.2
29    NYG  0.33  │  NYG   4.4  │  DEN  41.9
30    DET  0.30  │  MIA   5.0  │  PIT  41.8
31    CIN  0.30  │  OAK   5.5  │  BUF  41.1
32    WAS  0.28  │  WAS   5.5  │  CHI  40.4 

*expected performance against league average
opponent on a neutral field
```
And it can generate point spread and point total predictions for arbitrary matchups in the future...
```
> nflmodel predict 2019-12-08 CLE BAL --spread -110 -115 -12 --total -110 -110 45                 

[INFO][nflmodel] 2019-12-08T00:00:00 CLE at BAL

               away   home
team            CLE    BAL
win prob        12%    88%
spread         13.7  -13.7
total          47.4   47.4
score            17     31
spread cover    45%    55%
spread return  -15%     3%
                          
               over  under
total cover     56%    44%
total return     9%   -19% 

*actual return rate lower than predicted

```

Additionally, you can validate the model predictions by calling
```
nflmodel validate
```
which generates two figures, `validate_spread.pdf` and `validate_total.pdf`, visualizing the distribution of prediction residuals and quantiles.
