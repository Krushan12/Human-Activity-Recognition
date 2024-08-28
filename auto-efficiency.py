import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

from sklearn.tree import DecisionTreeRegressor
from tqdm import tqdm  # Import tqdm for progress tracking during loops

np.random.seed(42)

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
data = pd.read_csv(url, delim_whitespace=True, header=None,
                 names=["mpg", "cylinders", "displacement", "horsepower", "weight",
                        "acceleration", "model year", "origin", "car name"])

# Clean the data by removing rows with missing or junk values
data = data[data['horsepower']!='?'].reset_index(drop=True)
data.dropna(inplace=True)
data = data.drop('car name', axis = 1)

# Separate features (X) and target variable (y)
y = data['mpg']
X = data.drop('mpg', axis = 1)

# Convert data types to float64
X = X.astype('float64')

# Split the data into training and final testing sets
X_frame = X[:350].reset_index(drop=True)
y_series = y[:350].reset_index(drop=True)

X_final_test = X[350:].reset_index(drop=True)
y_final_test = y[350:].reset_index(drop=True)

# Set the number of folds for cross-validation
k = 9
fold_size = int(len(X_frame)/k)

# Calculate the mean of the target variable for later accuracy calculation
output_mean = y_final_test.mean()

# Define hyperparameters to be tuned
depths = [1,2,3,4,5,6,7,8,9,10]
criterion = "information_gain"

# Initialize variables to store optimal hyperparameters and corresponding performance metrics
optimal_depth = None
min_avg_rmse = np.inf

# Perform hyperparameter tuning using cross-validation
for depth in tqdm(depths):
    curr_avg_rmse = 0
    
    for i in range(k):
        # Split the data into training and validation sets for each fold
        X_train = pd.concat((X_frame[0:i*fold_size], X_frame[(i+1)*fold_size:]), axis=0).reset_index(drop=True)
        y_train = pd.concat((y_series[0:i*fold_size], y_series[(i+1)*fold_size:]), axis=0).reset_index(drop=True)
        
        X_validation = X_frame[i*fold_size:(i+1)*fold_size].reset_index(drop=True)
        y_validation = y_series[i*fold_size:(i+1)*fold_size].reset_index(drop=True)
        
        # Train the custom DecisionTree model
        tree = DecisionTree(criterion=criterion, max_depth=depth)
        tree.fit(X_train, y_train)
        
        # Make predictions on the validation set
        y_hat = tree.predict(X_validation)
        
        # Calculate and accumulate the RMSE for the current fold
        current_rmse = rmse(y_hat, y_validation)
        curr_avg_rmse += current_rmse
    
    # Calculate the average RMSE for the current depth
    curr_avg_rmse = curr_avg_rmse / k

    # Update optimal hyperparameters if the current depth yields lower average RMSE
    if curr_avg_rmse < min_avg_rmse:
        optimal_depth = depth
        min_avg_rmse = curr_avg_rmse

# Using Own Implementation
# Initialize and train the custom DecisionTree model with optimal hyperparameters
custom_model = DecisionTree(criterion=criterion, max_depth=optimal_depth)
custom_model.fit(X_frame, y_series)

# Make predictions on the final testing set
y_hat_custom = custom_model.predict(X_final_test)

# Calculate performance metrics for the custom model
custom_rmse_value = rmse(y_hat_custom, y_final_test)
custom_mae_value = mae(y_hat_custom, y_final_test)
custom_accuracy_value = (1 - (custom_rmse_value / output_mean)) * 100

# Using sklearn
# Initialize and train the DecisionTreeRegressor model from scikit-learn with optimal hyperparameters
sklearn_model = DecisionTreeRegressor(max_depth=optimal_depth)
sklearn_model.fit(X_frame, y_series)

# Make predictions on the final testing set using scikit-learn model
y_hat_sklearn = sklearn_model.predict(X_final_test)

# Calculate performance metrics for the scikit-learn model
sklearn_rmse_value = rmse(y_hat_sklearn, y_final_test)
sklearn_mae_value = mae(y_hat_sklearn, y_final_test)
sklearn_accuracy_value = (1 - (sklearn_rmse_value / output_mean)) * 100

# Print the results
print("Optimal Depth: {}, Optimal Criterion: {}".format(optimal_depth, criterion))
print("Our Model RMSE: {}, Our Model MAE: {}, Our Model Accuracy: {}".format(custom_rmse_value, custom_mae_value, custom_accuracy_value))
print("Sklearn Model RMSE: {}, Sklearn Model MAE: {}, Sklearn Model Accuracy: {}".format(sklearn_rmse_value, sklearn_mae_value, sklearn_accuracy_value))
