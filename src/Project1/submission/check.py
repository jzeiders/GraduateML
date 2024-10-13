import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

# Load the actual values
test_y = pd.read_csv('test_y.csv')

# Load the predictions
submission1 = pd.read_csv('mysubmission1.txt')
submission2 = pd.read_csv('mysubmission2.txt')

# Ensure that the order of PIDs matches
test_y = test_y.sort_values('PID')
submission1 = submission1.sort_values('PID')
submission2 = submission2.sort_values('PID')

# Calculate RMSE for both submissions
rmse1 = np.sqrt(mean_squared_error(np.log(test_y['Sale_Price']), np.log(submission1['Sale_Price'])))
rmse2 = np.sqrt(mean_squared_error(np.log(test_y['Sale_Price']), np.log(submission2['Sale_Price'])))

print(f"RMSE for submission 1 (ElasticNetCV): {rmse1:.5f}")
print(f"RMSE for submission 2 (XGBoost): {rmse2:.5f}")