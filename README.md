
About
=====
This is a "reference" implementation of the proposed refactor of GPU split evaluation kernel in XGBoost: https://github.com/trivialfis/xgboost/tree/rework-evaluation. The reference implementation is simple to understand and debug since it uses a single CPU thread.

Expected output
===============
For the toy example included in `main.cc`, the first line of output is correct but the second line is wrong:
```
findex = 0, fvalue = 1, loss_chg = 4
findex = -1, fvalue = 0, loss_chg = -3.40282e+38
```
