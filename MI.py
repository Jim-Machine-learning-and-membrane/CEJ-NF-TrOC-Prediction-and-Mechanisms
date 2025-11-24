import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler

# Set the font style and size for the plots
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 28

# Load the data from the provided Excel file path
# Replace 'file_path' with the actual path to your Excel file
df = pd.read_excel(file_path)

# Select only numeric columns, convert them to numeric type, drop rows with missing values, and reset the index
df_selected = (
    df.select_dtypes(include=[np.number])  # Select numeric columns
    .apply(pd.to_numeric, errors='coerce')  # Convert all values to numeric, replacing errors with NaN
    .dropna()  # Drop rows with missing values
    .reset_index(drop=True)  # Reset index after dropping rows
)

# Define the target column and feature columns
target_col = target  # Replace 'target' with the actual name of the target column
feature_cols = [c for c in df_selected.columns if c != target_col]

# Define color maps for visualization
blue_cmap = LinearSegmentedColormap.from_list('blue_grad', ['#d8e8f5', '#004c97'])
red_cmap  = LinearSegmentedColormap.from_list('red_grad',  ['#fde0dd', '#a50f15'])

# Calculate Spearman correlation between each feature and the target column
spearman_with_target = (
    df_selected[feature_cols + [target_col]]  # Select features and target column
    .corr(method='spearman')[target_col]  # Compute Spearman correlation
    .drop(target_col)  # Drop the target column itself
)

# Sort the correlation values in ascending order
sp_sorted = spearman_with_target.sort_values(ascending=True)

# Normalize the Spearman values for coloring the bars
norm_sp = plt.Normalize(vmin=0, vmax=abs(sp_sorted).max())  # Normalize by the maximum absolute value
colors_sp = [blue_cmap(norm_sp(abs(v))) for v in sp_sorted.values]  # Map Spearman values to colors

# Create a bar plot for Spearman correlations
plt.figure(figsize=(8, 7))
plt.barh(sp_sorted.index, sp_sorted.values, color=colors_sp, edgecolor='none')  # Plot bars with color
plt.axvline(0, color='0.75', lw=1)  # Add a vertical line at x=0
plt.grid(axis='x', linestyle='--', alpha=0.3)  # Add grid lines on the x-axis
plt.tight_layout()  # Adjust layout to prevent overlap
plt.savefig(output_path_spearman, dpi=1200)  # Save the plot to the specified path
plt.show()  # Display the plot

# Save the Spearman correlations to a CSV file
sp_sorted.rename("Spearman_rho").to_csv(spearman_csv_path)

# Prepare data for Mutual Information (MI) calculation
X = StandardScaler().fit_transform(df_selected[feature_cols])  # Standardize the feature columns
y = df_selected[target_col].values  # Target values

# Compute Mutual Information for each feature with the target
mi = mutual_info_regression(X, y)

# Create a series with the MI values and sort them
mi_series = pd.Series(mi, index=feature_cols)
mi_sorted = mi_series.sort_values(ascending=True)

# Normalize the MI values for coloring the bars
norm_mi = plt.Normalize(vmin=0, vmax=mi_sorted.max())  # Normalize by the maximum MI value
colors_mi = [red_cmap(norm_mi(v)) for v in mi_sorted.values]  # Map MI values to colors

# Create a bar plot for Mutual Information values
plt.figure(figsize=(8, 7))
plt.barh(mi_sorted.index, mi_sorted.values, color=colors_mi, edgecolor='none')  # Plot bars with color
plt.grid(axis='x', linestyle='--', alpha=0.3)  # Add grid lines on the x-axis
plt.tight_layout()  # Adjust layout to prevent overlap
plt.savefig(output_path_mi, dpi=1200)  # Save the plot to the specified path
plt.show()  # Display the plot

# Save the Mutual Information values to a CSV file
mi_sorted.rename("Mutual_Information").to_csv(mi_csv_path)
