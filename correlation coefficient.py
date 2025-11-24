import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Load data from the provided Excel file path
# Note: 'file_path' should be replaced with the actual file path of the Excel file
df = pd.read_excel(file_path)

# Compute the correlation matrix for numeric columns in the DataFrame
# This will calculate the pairwise correlations between all numeric columns
correlation_matrix = df.corr(numeric_only=True)

# Create a custom colormap using a gradient of two colors
# The colormap starts from a green color and transitions to an orange color
cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#66c2a5', '#fc8d62'])

# Create a new figure for the heatmap
plt.figure()

# Plot the heatmap using seaborn with the custom colormap
# 'annot' can be set to True if you want to display the correlation values on the heatmap
sns.heatmap(correlation_matrix, cmap=cmap, annot=False)

# Adjust the layout to prevent overlapping elements
plt.tight_layout()

# Display the plot
plt.show()
