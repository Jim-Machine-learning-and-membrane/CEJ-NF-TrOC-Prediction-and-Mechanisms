import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

# Load data from the provided Excel file path
# Replace 'file_path' with the actual path to your Excel file
df = pd.read_excel(file_path)

# Define feature matrix (X) and target variable (y)
# 'target' should be replaced with the name of the column you want to predict
X = df.drop([target], axis=1)
y = df[target]

# Split the data into training and testing sets (test_size specifies the proportion of data used for testing)
# Set 'random_state' for reproducibility of the split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

# Initialize and train the XGBoost regression model with specified parameters
# Replace 'params' with the dictionary of model hyperparameters
model = xgb.XGBRegressor(**params, objective='reg:squarederror', random_state=seed, n_jobs=-1)
model.fit(X_train, y_train)

# Set up KFold cross-validation with specified number of splits (n_splits)
kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

# Perform cross-validation and compute R² scores for the model
cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2')

# Make predictions on both training and testing sets
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate performance metrics for the training and testing data
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))  # Root Mean Squared Error
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_mse = mean_squared_error(y_train, y_train_pred)  # Mean Squared Error
test_mse = mean_squared_error(y_test, y_test_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)  # Mean Absolute Error
test_mae = mean_absolute_error(y_test, y_test_pred)
train_mre = np.mean(np.abs((y_train - y_train_pred) / y_train))  # Mean Relative Error
test_mre = np.mean(np.abs((y_test - y_test_pred) / y_test))

# Print the cross-validation scores and the performance metrics for both training and testing data
print(cv_scores.mean())  # Mean R² score from cross-validation
print(train_r2, train_rmse, train_mse, train_mae, train_mre)  # Training set metrics
print(test_r2, test_rmse, test_mse, test_mae, test_mre)  # Test set metrics

# Plot the actual vs predicted values for both training and testing data
plt.figure()
plt.scatter(y_train, y_train_pred, label='Train', alpha=0.6)  # Scatter plot for training data
plt.scatter(y_test, y_test_pred, label='Test', alpha=0.6)  # Scatter plot for test data
plt.plot([y.min(), y.max()], [y.min(), y.max()], linestyle='--', label='Ideal')  # Ideal line (y=x)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.legend()
plt.tight_layout()
plt.show()
