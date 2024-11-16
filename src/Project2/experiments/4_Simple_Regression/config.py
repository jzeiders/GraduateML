config = {
    "data_dir": "data",  # Directory where your data folds and test labels are stored

    "feature_engineering": {
        "date_features": {
            "include": True,
            "features": ["month", "year", 'dayofyear']  # Features to extract from the Date column
        },
        "holiday_proximity": {
            "include": False  # Whether to include holiday proximity features
        },
        "interaction_features": {
            "include": True  # Whether to include interaction features like Store_Dept
        }
    },

    "preprocessing": {
        "numerical_features": [],
        "categorical_features": [
            "Store",
            "Dept",
            "Store_Dept",
            "IsHoliday", 
            "Month",
            'DayOfYear', 
            "Year",
        ]
    },

    "model": {
        "type": "XGBRegressor",  # Model type to use
        "params": {}
    }
}
