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


Best parameters for ElasticNet: {'model__alpha': np.float64(0.00042813323987193956), 'model__l1_ratio': np.float64(1.0)}

Best parameters for ElasticNet: {'model__alpha': np.float64(0.0006951927961775605), 'model__l1_ratio': np.float64(0.55)}

Target encoding dit much worse.

# 10/12
Going to try upgrading from random forest to XGBoost.


Implementing initial XGBoost was limited success. 

 model=XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        ),

Changing the parameters greatly increased performance

       model=XGBRegressor(
            n_estimators=5000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        ),
TODO: Train on these to get even better.


Tried Elastic CV -> Did better but not great. With a CV of 5
Trying CV 50 -> Din't do any better. That's becasuse I was wrong with what that does.

Trying to do even better feature eng using automated script for dropping. Didn't really help

Trying to use a much larger cv search -> same outcome. Our solution must be fairly decent.

Standard Scaler is very important.

Feature engineering does help but not as much I would think...