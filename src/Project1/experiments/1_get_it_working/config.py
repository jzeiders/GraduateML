config = {
    'feature_engineering': {
        'drop_columns': ['Latitude', 'Longitude'],
        'highly_correlated': ['TotRms_AbvGrd', 'Garage_Yr_Blt', 'Garage_Area', 'Latitude'],
        'potential_non_linear': ['Lot_Frontage'],
        'sparse_categories': ['Street'],
    },

    'models': {
        'ElasticNet': {
            'type': 'ElasticNet',
            'params': {},
        }
     
    },

    'encoding': 'onehot',  # Options: 'onehot' or 'target'
}
