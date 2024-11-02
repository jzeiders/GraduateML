config = {
    "data_dir": "data",  # Directory where your data folds and test labels are stored

    "feature_engineering": {
        "date_features": {
            "include": True,
            "features": ["month", "weekofyear", "dayofweek"]  # Features to extract from the Date column
        },
        "holiday_proximity": {
            "include": True  # Whether to include holiday proximity features
        },
        "interaction_features": {
            "include": True  # Whether to include interaction features like Store_Dept
        }
    },

    "preprocessing": {
        "numerical_features": [
            "Store", 
            "Dept",
            "Month", 
            "WeekOfYear", 
            "DayOfWeek", 
            "DaysUntilHoliday", 
            "DaysSinceHoliday"
        ],
        "categorical_features": [
            "IsHoliday", 
            "Store_Dept"
        ]
    },

    "model": {
        "type": "ElasticNetCV",  # Model type to use
        "params": {
            "l1_ratio": 0.5,        # Balance between L1 and L2 regularization
            "alphas": [0.1, 1.0, 10.0],  # List of alphas to try in cross-validation
            "cv": 5,                 # Number of cross-validation folds
            "random_state": 42       # Seed for reproducibility
        }
    }
}
