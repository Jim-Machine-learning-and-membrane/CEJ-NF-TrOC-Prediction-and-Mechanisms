import pandas as pd
from scipy.stats import levene

# Load data from the provided Excel file path
# Replace 'file_path' with the actual file path of the Excel file
data = pd.read_excel(file_path)

# Split data into features (X) and target variable (y)
# 'TrOC rejection' is the target variable we are interested in
X = data.drop(['TrOC rejection'], axis=1)
y = data['TrOC rejection']

# Calculate basic statistics for each feature (X)
stats = pd.DataFrame({
    'min': X.min(),    # Minimum value for each feature
    'max': X.max(),    # Maximum value for each feature
    'mean': X.mean(),  # Mean value for each feature
    'std': X.std()     # Standard deviation for each feature
})

# Initialize a dictionary to store p-values from Levene's test
variance_pvals = {}

# Perform Levene's test for homogeneity of variance for each feature
# The data is split into two groups based on whether the target variable y is below or above its median
for col in X.columns:
    g1 = X[y <= y.median()][col]  # Group 1: Data where target y is below or equal to the median
    g2 = X[y > y.median()][col]   # Group 2: Data where target y is above the median
    _, p = levene(g1, g2)         # Levene's test for equal variances between the two groups
    variance_pvals[col] = p        # Store p-value for the feature

# Add the Levene's test p-values as a new column to the stats DataFrame
stats['Levene_p_value'] = stats.index.map(variance_pvals)

# Set the display format to show 4 decimal places for floating-point numbers
pd.set_option('display.float_format', '{:.4f}'.format)

# Print the statistics table to the console
print(stats)

# Save the statistics table to an Excel file at the specified output path
stats.to_excel(output_path, index=True)
