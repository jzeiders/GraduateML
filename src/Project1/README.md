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

Hyper-Paramter Tuning for Lasso, selected using cross-validation.
Alpha - 0.00054 

Hyper Parameter Lasso
Best parameters for Random Forest: {'model__n_estimators': 200, 'model__min_samples_split': 5, 'model__min_samples_leaf': 1, 'model__max_features': 'sqrt', 'model__max_depth': 20}