import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

# Load the dataset from the provided Excel file path
# Replace 'file_path' with the actual path to your Excel file
df = pd.read_excel(file_path)

# Define feature matrix (X) and target variable (y)
# 'target' should be replaced with the name of the column you want to predict
X = df.drop([target], axis=1)  # Drop the target column from the feature matrix
y = df[target]  # Target variable

# Split the data into training and testing sets (test_size specifies the test data ratio)
# 'random_state' ensures the split is reproducible
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

# Initialize and train the XGBoost regressor model
# Replace 'params' with the dictionary of hyperparameters for the XGBoost model
model = xgb.XGBRegressor(**params, objective='reg:squarederror', random_state=seed, n_jobs=-1)
model.fit(X_train, y_train)

# Set up K-Fold cross-validation with specified number of splits (n_splits)
kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

# Perform cross-validation and compute R² scores for the model
cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2')

# Make predictions for both training and testing data
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate performance metrics for the training and testing data
train_r2 = r2_score(y_train, y_train_pred)  # R² score for training data
test_r2 = r2_score(y_test, y_test_pred)  # R² score for testing data
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))  # Root Mean Squared Error for training data
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))  # Root Mean Squared Error for testing data
train_mse = mean_squared_error(y_train, y_train_pred)  # Mean Squared Error for training data
test_mse = mean_squared_error(y_test, y_test_pred)  # Mean Squared Error for testing data
train_mae = mean_absolute_error(y_train, y_train_pred)  # Mean Absolute Error for training data
test_mae = mean_absolute_error(y_test, y_test_pred)  # Mean Absolute Error for testing data
train_mre = np.mean(np.abs((y_train - y_train_pred) / y_train))  # Mean Relative Error for training data
test_mre = np.mean(np.abs((y_test - y_test_pred) / y_test))  # Mean Relative Error for testing data

# Print the cross-validation R² score, as well as training and testing metrics
print(cv_scores.mean())  # Average R² score from cross-validation
print(train_r2, train_rmse, train_mse, train_mae, train_mre)  # Training set metrics
print(test_r2, test_rmse, test_mse, test_mae, test_mre)  # Test set metrics

# Create a scatter plot to visualize the actual vs predicted values for both training and testing data
plt.figure()
plt.scatter(y_train, y_train_pred, label='Train', alpha=0.6)  # Scatter plot for training data
plt.scatter(y_test, y_test_pred, label='Test', alpha=0.6)  # Scatter plot for test data

# Plot the ideal line (y=x) for reference
plt.plot([y.min(), y.max()], [y.min(), y.max()], linestyle='--', label='Ideal')  # Ideal line (y=x)

# Set plot labels and display the legend
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.legend()

# Adjust the layout and display the plot
plt.tight_layout()
plt.show()
