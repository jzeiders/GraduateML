import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from category_encoders import OneHotEncoder
from xgboost import XGBRegressor
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.filterwarnings("ignore")

# Configuration
# I had created a configuration based experiment runner for testing & tuning the models.
# I've just in-lined the best outcome here. If you're wondering why I use this weird format..
config = {
    'feature_engineering': {
        'drop_columns': ['Latitude', 'Longitude'],
        'highly_correlated': ['TotRms_AbvGrd', 'Garage_Yr_Blt', 'Garage_Area', 'Latitude'],
        'potential_non_linear': ['Lot_Frontage'],
        'sparse_categories': ['Street'],
        'numeric_as_categorical': ['MS_SubClass', 'Overall_Qual', 'Overall_Cond', 'Mo_Sold', 'Bsmt_Full_Bath', 'Bsmt_Half_Bath', 'Full_Bath', 'Half_Bath', 'Bedroom_AbvGr', 'Kitchen_AbvGr', 'Garage_Cars']
    },
    'models': {
        'ElasticNetCV': {
            'type': 'ElasticNetCV',
            'params': {
                'cv': 5,
                'l1_ratio': [0.1,0.2,0.3,0.4,0.5,0.7,0.8,0.9,0.95,0.97,0.99,1],
                'random_state': 42,
                'n_jobs': -1,
            },
        },
        'XGBoost': {
            'type': 'XGBRegressor',
            'params': {
                'n_estimators': 5000,
                'learning_rate': 0.05,
                'max_depth': 6,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_jobs': -1,
            },
        },
    },
    'encoding': 'onehot',
}

# Outlier capper class
class OutlierCapper(BaseEstimator, TransformerMixin):
    def __init__(self, lower_quantile=0.05, upper_quantile=0.95):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.lower_bound = None
        self.upper_bound = None
    
    def fit(self, X, y=None):
        self.lower_bound = np.quantile(X, self.lower_quantile, axis=0)
        self.upper_bound = np.quantile(X, self.upper_quantile, axis=0)
        return self
    
    def transform(self, X):
        X_capped = np.clip(X, self.lower_bound, self.upper_bound)
        return X_capped

# Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Feature engineering
drop_cols = config['feature_engineering']['drop_columns'] + \
            config['feature_engineering']['highly_correlated'] + \
            config['feature_engineering']['potential_non_linear'] + \
            config['feature_engineering']['sparse_categories']

train = train.drop(columns=drop_cols, errors='ignore')
test = test.drop(columns=drop_cols, errors='ignore')

# Separate features and target
y = train['Sale_Price']
X = train.drop(['Sale_Price', 'PID'], axis=1)

# Identify numeric and categorical columns
numeric_features = list(set(X.select_dtypes(include=['int64', 'float64']).columns) - 
                        set(config['feature_engineering']['numeric_as_categorical']))
categorical_features = X.select_dtypes(include=['object']).columns.tolist() + \
                       config['feature_engineering']['numeric_as_categorical']

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('outlier', OutlierCapper())
        ]), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
            ('onehot', OneHotEncoder(use_cat_names=True))
        ]), categorical_features)
    ])

# Create models
model1 = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', ElasticNetCV(**config['models']['ElasticNetCV']['params']))
])

model2 = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(**config['models']['XGBoost']['params']))
])

# Fit models
model1.fit(X, np.log(y))
model2.fit(X, np.log(y))

# Prepare test data
X_test = test.drop('PID', axis=1)

# Make predictions
preds1 = np.exp(model1.predict(X_test))
preds2 = np.exp(model2.predict(X_test))

# Create submission files
submission1 = pd.DataFrame({'PID': test['PID'], 'Sale_Price': preds1})
submission2 = pd.DataFrame({'PID': test['PID'], 'Sale_Price': preds2})

submission1.to_csv('mysubmission1.txt', index=False)
submission2.to_csv('mysubmission2.txt', index=False)