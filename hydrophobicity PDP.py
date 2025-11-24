import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.inspection import PartialDependenceDisplay

# Load the data from the provided Excel file path
# Replace 'file_path' with the actual path to your Excel file
df = pd.read_excel(file_path)

# Define feature matrix (X) and target variable (y)
# 'target' should be replaced with the actual name of the target column
X = df.drop(target, axis=1)
y = df[target]

# Split the data into training and testing sets (default is 75% training, 25% testing)
# Replace 'target' with the actual target column
X_train, X_test, y_train, y_test = train_test_split(X, y)

# List of features for which Partial Dependence Plots will be created
features = ["Molecular charge", "TrOC concentration", "Log D", "Zeta potential"]

# Initialize and train the XGBoost regressor model
model = xgb.XGBRegressor()
model.fit(X_train, y_train)

# Determine the number of features and calculate the required number of rows and columns for subplots
n_features = len(features)
n_cols = 2  # Define the number of columns in the subplot grid
n_rows = (n_features + n_cols - 1) // n_cols  # Calculate the number of rows

# Create subplots
fig, axes = plt.subplots(n_rows, n_cols)
axes = axes.flatten()  # Flatten the axes array for easier indexing

# Loop through each feature and plot its partial dependence
for i, feature in enumerate(features):
    # Create the partial dependence plot for the current feature
    PartialDependenceDisplay.from_estimator(model, X_train, [feature], ax=axes[i])
    
    # Set the x-axis range to match the feature's minimum and maximum values
    feature_range = X_train[feature].describe()[['min', 'max']].values.flatten()
    axes[i].set_xlim(feature_range)  # Set the limits of the x-axis to feature's min and max
    
    # Set axis labels
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Partial Dependence')

# Hide any unused subplots if the number of features is not a perfect multiple of the columns
for j in range(n_features, n_rows * n_cols):
    axes[j].axis('off')

# Adjust the layout to avoid overlapping elements
plt.tight_layout()

# Show the plot
plt.show()
