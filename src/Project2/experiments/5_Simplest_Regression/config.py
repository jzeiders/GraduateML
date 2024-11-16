config = {
    "data_dir": "data",  

    "feature_engineering": {
        "date_features": {
            "include": True,
            "features": ['Year', 'WeekOfYear']  # Reduced to only year and week of year
        },
        "holiday_proximity": {
            "include": False
        },
        "interaction_features": {
            "include": False  # Disabled interaction features
        },
        "polynomial_features": ['Year']
    },

    "preprocessing": {
        "numerical_features": [
            'Year',
        ],
        "categorical_features": [
            'WeekOfYear',
            "IsHoliday",
            "Store",
            "Dept"
        ]  # No categorical features needed
    },

    "model": {
        "type": "LinearRegression",  # Changed to ElasticNet for linear regression
        "params": {
        }
    }
}