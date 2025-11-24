import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import make_scorer, r2_score
from sklearn.base import clone

# Load data from the provided Excel file path
# Replace 'file_path' with the actual file path of the Excel file
df = pd.read_excel(file_path)

# Define feature matrix (X) and target variable (y)
# 'target' should be replaced with the name of the column you want to predict
X = df.drop(columns=[target])
y = df[target]

# Split the data into training and testing sets (test_size specifies the test data ratio)
# Set 'random_state' for reproducibility of the split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

# Define a dictionary of models to evaluate, using specific parameters
models = {
    "XGB": xgb.XGBRegressor(**params_xgb, random_state=seed, n_jobs=-1),
    "RF": RandomForestRegressor(**params_rf, random_state=seed, n_jobs=-1),
    "GBDT": GradientBoostingRegressor(**params_gbdt, random_state=seed),
}

# Define the scoring metric to evaluate models using R²
scorer = make_scorer(r2_score)

# Evaluate models using cross-validation and compute the mean R² score
scores = {k: cross_val_score(m, X_train, y_train, cv=cv_splits, scoring=scorer).mean() for k, m in models.items()}

# Select the preferred model based on the previous evaluations and fit it to the training data
model = models[preferred_model_key]
model.fit(X_train, y_train)

# Create a SHAP explainer for the selected model
explainer = shap.Explainer(model)

# Compute SHAP values for the test set
S_test = explainer(X_test)

# Take the absolute value of SHAP values for feature importance
S = np.abs(S_test.values)

# List of feature names for the test set
feat_names = list(X_test.columns)

# Function to calculate bootstrap confidence intervals (CI) for SHAP values
def bootstrap_ci(mat, B, agg=np.mean, q=(2.5, 97.5), random_state=None):
    rng = np.random.default_rng(random_state)
    n = mat.shape[0]
    buf = np.empty((B, mat.shape[1]))
    
    # Bootstrap resampling
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        buf[b] = agg(mat[idx], axis=0)
    
    # Calculate point estimates and CI intervals
    point = agg(mat, axis=0)
    lo, hi = np.percentile(buf, q[0], axis=0), np.percentile(buf, q[1], axis=0)
    pct = 100 * point / np.nansum(point)
    
    # Return the results as a DataFrame sorted by mean absolute SHAP value
    out = pd.DataFrame({"Mean|SHAP|": point, "CI_low": lo, "CI_high": hi, "Pct": pct}, index=feat_names)
    return out.sort_values("Mean|SHAP|", ascending=False)

# Function to create masks based on a threshold for a specific column in the reference DataFrame
def mask_by_rule(Xref, col, thresh):
    m1 = Xref[col] < thresh
    m2 = ~m1
    return m1.values, m2.values

# Apply the mask to the test set based on a threshold value for a specific column
m_h, m_o = mask_by_rule(X_test, logd_col, logd_thresh)

# Calculate bootstrap CI for the entire set of SHAP values
tab_all = bootstrap_ci(S, B=n_boot, random_state=seed_all)

# Calculate bootstrap CI for hydrophilic subset (m_h) if any such samples exist
tab_h = bootstrap_ci(S[m_h], B=n_boot, random_state=seed_h) if m_h.any() else pd.DataFrame(columns=["Mean|SHAP|","CI_low","CI_high","Pct"])

# Calculate bootstrap CI for hydrophobic subset (m_o) if any such samples exist
tab_o = bootstrap_ci(S[m_o], B=n_boot, random_state=seed_o) if m_o.any() else pd.DataFrame(columns=["Mean|SHAP|","CI_low","CI_high","Pct"])

# Add group labels to the results
tab_all["Group"] = "All"
tab_h["Group"] = "Hydrophilic"
tab_o["Group"] = "Hydrophobic"

# Concatenate the results into a single table
table4 = pd.concat([tab_h, tab_o], axis=0, ignore_index=False)

# Normalize the 'Pct' values by group to ensure they sum to 100% within each group
table4["Pct"] = table4.groupby("Group")["Pct"].transform(lambda v: 100 * v / v.sum())

# Reset the index and rename columns for clarity
table4 = table4.reset_index().rename(columns={"index": "Feature"})

# Select and order columns to output
table4 = table4[["Group", "Feature", "Mean|SHAP|", "CI_low", "CI_high", "Pct"]]

# Save the resulting table to an Excel file at the specified output path
table4.to_excel(output_path, index=False)
