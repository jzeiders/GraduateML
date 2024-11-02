config = {
    'model': {
        'type': 'ElasticNet',
        'params': {
            'alpha': 0.1,
            'l1_ratio': 0.5
        }
    },
    'feature_engineering': {
        'date_features': {
            'include': True,
            'features': ['month', 'dayofweek']
        },
        'lag_features': {
            'include': False
        },
        'rolling_stats': {
            'include': False
        },
        'holiday_proximity': {
            'include': False
        },
        'interaction_features': {
            'include': False
        }
    }
}
