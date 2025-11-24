import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

# Load the data from the provided Excel file path
# Replace 'file_path' with the actual path to your Excel file
df = pd.read_excel(file_path)

# Select numerical columns in the DataFrame (columns with numeric data types)
num_cols = df.select_dtypes(include=[np.number]).columns

# Initialize the KNN imputer with specified parameters
# Replace 'k', 'weighting', and 'metric' with your desired values
imputer = KNNImputer(n_neighbors=k, weights=weighting, metric=metric)

# Create a copy of the original DataFrame to store the imputed values
df_imputed = df.copy()

# Apply the KNN imputer to the numerical columns and replace the original values with imputed ones
df_imputed[num_cols] = imputer.fit_transform(df[num_cols])

# Save the imputed DataFrame to an Excel file at the specified output path
df_imputed.to_excel(output_path, index=False)
