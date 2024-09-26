# 9/23
Spent about 45 minutes on this.

## Key Commands

```
python mymain.py [folder-name] 
python mymain.py evaluate 
```

## Experiments
Tried CV Search, didn't really help...
In the one-hot encoder, handle_unknown='ignore' caused issues with NAN's in prediction.
Increasing the number of random forest predictors from 100 -> 500 did help performance

Increase lasso iter's by a magnitude.

Hyper-Paramter Tuning for Lasso
Alpha - 0.00054 