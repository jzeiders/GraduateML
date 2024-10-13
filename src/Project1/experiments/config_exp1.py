# config_exp1.py

config = {
    'experiment_name': 'exp1',
    'data_dir': 'data/fold1',
    'results_dir': 'results/exp1',

    'feature_engineering': {
        'drop_columns': ['Latitude', 'Longitude'],
        'highly_correlated': ['TotRms_AbvGrd', 'Garage_Yr_Blt', 'Garage_Area', 'Latitude'],
        'potential_non_linear': ['Lot_Frontage'],
        'sparse_categories': ['Street'],
    },

    'models': {
        'ElasticNetCV': {
            'type': 'ElasticNetCV',
            'params': {
                'cv': 5,
                'random_state': 42,
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

    'encoding': 'onehot',  # Options: 'onehot' or 'target'
}
