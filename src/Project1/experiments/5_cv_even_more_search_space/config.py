config = {
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
                'l1_ratio': [0.1,0.2,0.3,0.4, 0.5, 0.7, 0.8, 0.9, 0.95,0.97, 0.99, 1],
                'random_state': 42,
                'n_jobs': -1,
            },
        }
    },

    'encoding': 'onehot',  # Options: 'onehot' or 'target'
}
