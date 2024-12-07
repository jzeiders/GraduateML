https://liangfgithub.github.io/Proj/F24_Proj3.html
https://campuswire.com/c/GB46E5679/feed/278

# GOAL
AUC Target of 0.986

# Train Format
id: int
sentiment 0,1
review: text
embedding_[1,1536]

# Test Format
id: int
review: text
embedding_[1,1536]

# Test Label Format
id: int
sentiment 0,1

## Part 1 Basic Experiments    
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

Running a really big hyper-parameter search. Hopefully it gets us much closer.


Grid Search Findings
Best parameters found:
alpha: 0.2694123337985215
colsample_bytree: 0.6976502088991097
gamma: 0.08414552108646528
lambda: 0.21876421957307024
learning_rate: 0.17743060060520235
max_depth: 6
min_child_weight: 1
n_estimators: 161
scale_pos_weight: 1.1887128330883843
subsample: 0.9849789179768444
Best CV score: 0.983416

Wasn't enough (SAD). Trying to just blow up the estimator count to 1k on that and see if it works. Still didn't work for any of them.


Trying Logistic Regresion to see if that works. Options:
        'C': [0.01, 0.1, 1, 10, 100]  # C is the inverse of alpha in Logistic Regression

Fold 4 -> C: 10
Fold 3 -> C: 10
Fold 2 -> C: 10


WOOT IT WORKED. Just had to optimize the alpha on logistic regression.

## Interperability
Trying to do it with BERT by dropping 1 word at a time. Doesn't seem to have worked....

Ok so things seems a little better with sentence level analysis. Getting a lot more BERT embeddings for testing. I want to make sure the weight transition matrix is decent, using 10k samples for that.

Then I am setting it up to use word, trigram, & sentence level analysis to produce the relative importance of each word. 

BERT is wrong because it always giving zero. Replacing the direct matrix transformation with predictions got things corrected. Not sure what I was doing wrong? I should have been able to directly multiply by the weights? Maybe I wasn't actually saving the weights correctly.

Basics of analysis are going well, saving the results in the results folder.

TODO:
1. Clean up the visualization
2. Plot the trigrams? 
3. Check the visualizations with and without the sentence level check
4. Figure out how to uplaod things.


For the visualization:
- trying the pre-built lime library# GraduateML
