import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    BaggingRegressor
)
from sklearn.tree import DecisionTreeRegressor
import lightgbm as lgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Load the data from the provided Excel file path
# Replace 'file_path' with the actual path to your Excel file
df = pd.read_excel(file_path)

# Define feature matrix (X) and target variable (y)
# 'target' should be replaced with the name of the column you want to predict
X = df.drop(columns=[target])  # Drop the target column from the feature matrix
y = df[target]  # The target column

# Split the data into training and testing sets (default is 75% training, 25% testing)
# Replace 'target' with the actual name of the target column
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Initialize models to evaluate
models = {
    'DecisionTree': DecisionTreeRegressor(),
    'Bagging(Tree)': BaggingRegressor(estimator=DecisionTreeRegressor()),
    'RandomForest': RandomForestRegressor(),
    'AdaBoost': AdaBoostRegressor(),
    'GradientBoosting': GradientBoostingRegressor(),
    'LightGBM': lgb.LGBMRegressor(),
    'ExtraTrees': ExtraTreesRegressor()
}

# Function to evaluate model performance on training and test data
def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    # Fit the model on the training data
    model.fit(X_train, y_train)
    
    # Make predictions for both training and testing data
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Small epsilon to avoid division by zero in MRE calculation
    eps = 1e-8
    
    # Calculate performance metrics for the training set
    train_r2 = r2_score(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))  # Root Mean Squared Error
    train_mse = mean_squared_error(y_train, y_train_pred)  # Mean Squared Error
    train_mae = mean_absolute_error(y_train, y_train_pred)  # Mean Absolute Error
    train_mre = np.mean(np.abs((y_train - y_train_pred) / (np.abs(y_train) + eps)))  # Mean Relative Error
    
    # Calculate performance metrics for the test set
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mre = np.mean(np.abs((y_test - y_test_pred) / (np.abs(y_test) + eps)))
    
    # Print evaluation results
    print(f"Model: {name}")
    print(f"Train R²: {train_r2:.4f}, RMSE: {train_rmse:.4f}, MSE: {train_mse:.4f}, MAE: {train_mae:.4f}, MRE: {train_mre:.4f}")
    print(f"Test  R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}, MSE: {test_mse:.4f}, MAE: {test_mae:.4f}, MRE: {test_mre:.4f}")
    print("-" * 50)

# Loop through each model in the dictionary and evaluate its performance
for name, model in models.items():
    evaluate_model(name, model, X_train, X_test, y_train, y_test)
