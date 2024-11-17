https://liangfgithub.github.io/Proj/F24_Proj3.html

# GOAL
AUC Target of 0.986


## Basic Experiments    
This got us very, very close on split1
```
model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        n_estimators=1000
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
```

Boosting the number of classifiers to a 1000 got us there on Split 1. Didn't on split 4. Or Split 3. Literally only worked on 1 split....