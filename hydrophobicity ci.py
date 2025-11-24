import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import make_scorer, r2_score
from sklearn.base import clone

# Load the dataset from the provided Excel file
# Replace 'file_path' with the actual path to your Excel file
df = pd.read_excel(file_path)

# Split the data into feature matrix (X) and target variable (y)
# Replace 'target' with the actual name of the target column
X = df.drop(columns=[target])  # Feature matrix (excluding the target column)
y = df[target]  # Target variable

# Split the data into training and testing sets with the specified test size and random seed
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

# Define the models to be evaluated
models = {
    "XGB": xgb.XGBRegressor(**params_xgb, random_state=seed, n_jobs=-1),
    "RF": RandomForestRegressor(**params_rf, random_state=seed, n_jobs=-1),
    "GBDT": GradientBoostingRegressor(**params_gbdt, random_state=seed),
}

# Define the scoring metric (R²)
scorer = make_scorer(r2_score)

# Perform cross-validation and store the average R² score for each model
scores = {k: cross_val_score(m, X_train, y_train, cv=cv_splits, scoring=scorer).mean() for k, m in models.items()}

# Select the preferred model and fit it to the training data
model = models[preferred_model_key]
model.fit(X_train, y_train)

# Create a SHAP explainer to interpret the model
explainer = shap.Explainer(model)
S_test = explainer(X_test)  # Get SHAP values for the test set

# Take the absolute value of SHAP values for feature importance
S = np.abs(S_test.values)

# Get feature names from the test set
feat_names = list(X_test.columns)

# Function to calculate bootstrap confidence intervals for the SHAP values
def bootstrap_ci(mat, B, agg=np.mean, q=(2.5, 97.5), random_state=None):
    rng = np.random.default_rng(random_state)
    n = mat.shape[0]
    buf = np.empty((B, mat.shape[1]))
    
    # Perform bootstrap resampling
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        buf[b] = agg(mat[idx], axis=0)
    
    # Calculate point estimate and confidence intervals
    point = agg(mat, axis=0)
    lo, hi = np.percentile(buf, q[0], axis=0), np.percentile(buf, q[1], axis=0)
    pct = 100 * point / np.nansum(point)  # Calculate percentage contribution
    
    # Create a DataFrame with the results
    out = pd.DataFrame({"Mean|SHAP|": point, "CI_low": lo, "CI_high": hi, "Pct": pct}, index=feat_names)
    return out.sort_values("Mean|SHAP|", ascending=False)  # Sort by mean SHAP value

# Function to create masks based on a threshold for a specific column
def mask_by_rule(Xref, col, thresh):
    m1 = Xref[col] < thresh  # Hydrophilic (below threshold)
    m2 = ~m1  # Hydrophobic (above threshold)
    return m1.values, m2.values

# Apply the mask based on the threshold values for the given column
m_h, m_o = mask_by_rule(X_test, logd_col, logd_thresh)

# Calculate the bootstrap confidence intervals for SHAP values for all data, hydrophilic, and hydrophobic subsets
tab_all = bootstrap_ci(S, B=n_boot, random_state=seed_all)
tab_h = bootstrap_ci(S[m_h], B=n_boot, random_state=seed_h) if m_h.any() else pd.DataFrame(columns=["Mean|SHAP|","CI_low","CI_high","Pct"])
tab_o = bootstrap_ci(S[m_o], B=n_boot, random_state=seed_o) if m_o.any() else pd.DataFrame(columns=["Mean|SHAP|","CI_low","CI_high","Pct"])

# Assign group labels for all, hydrophilic, and hydrophobic data
tab_all["Group"] = "All"
tab_h["Group"] = "Hydrophilic"
tab_o["Group"] = "Hydrophobic"

# Concatenate the results into one DataFrame
table4 = pd.concat([tab_h, tab_o], axis=0, ignore_index=False)

# Normalize the 'Pct' column within each group to ensure it sums to 100%
table4["Pct"] = table4.groupby("Group")["Pct"].transform(lambda v: 100 * v / v.sum())

# Reset the index and rename the columns for clarity
table4 = table4.reset_index().rename(columns={"index": "Feature"})

# Select relevant columns and reorder them
table4 = table4[["Group", "Feature", "Mean|SHAP|", "CI_low", "CI_high", "Pct"]]

# Save the final table to an Excel file
table4.to_excel(output_path, index=False)
