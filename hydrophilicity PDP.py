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
# 'target' should be replaced with the name of the column you want to predict
X = df.drop(target, axis=1)
y = df[target]

# Split the data into training and testing sets (default 75% training, 25% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Define the features for which to plot partial dependence plots
features = ["Molecular charge", "TrOC concentration", "Log D", "Zeta potential"]

# Initialize and train the XGBoost regressor model
model = xgb.XGBRegressor()
model.fit(X_train, y_train)

# Number of features to plot and define grid dimensions (2 columns)
n_features = len(features)
n_cols = 2
n_rows = (n_features + n_cols - 1) // n_cols  # Calculate required number of rows

# Create subplots for each feature's partial dependence plot
fig, axes = plt.subplots(n_rows, n_cols)
axes = axes.flatten()  # Flatten axes array for easy indexing

# Loop through each feature and plot its partial dependence
for i, feature in enumerate(features):
    # Create partial dependence plot for the current feature
    PartialDependenceDisplay.from_estimator(model, X_train, [feature], ax=axes[i])
    
    # Set x-axis range based on the feature's minimum and maximum values
    feature_range = X_train[feature].describe()[['min', 'max']].values.flatten()
    axes[i].set_xlim(feature_range)
    
    # Set labels for each subplot
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Partial Dependence')

# Turn off axes for any empty subplots (if n_features is not a perfect multiple of n_cols)
for j in range(n_features, n_rows * n_cols):
    axes[j].axis('off')

# Adjust the layout and display the plot
plt.tight_layout()
plt.show()
