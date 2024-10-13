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
                'l1_ratio': [0.1,0.2,0.3,0.4, 0.5, 0.7, 0.8, 0.9, 0.95,0.97, 0.99, 1],
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
    'encoding': 'onehot',  # Options: 'onehot' or 'target'
}