config = {
    "data_dir": "data",  

    "feature_engineering": {
        "date_features": {
            "include": True,
            "features": ['Year', 'WeekOfYear']  # Reduced to only year and week of year
        },
        "holiday_proximity": {
            "include": True
        },
        "interaction_features": {
            "include": True  # Disabled interaction features
        },
        "polynomial_features": ['Year']
    },

    "preprocessing": {
        "numerical_features": [
            'Year2',
        ],
        "categorical_features": [
            'WeekOfYear',
            "IsHoliday",
            "Store",
            "Dept",
            "Store_Dept",
            "NearToHoliday_Dept",
            'NearToHoliday',
        ]  # No categorical features needed
    },

    "model": {
        "type": "LinearRegression",  # Changed to ElasticNet for linear regression
        "params": {
        }
    }
}