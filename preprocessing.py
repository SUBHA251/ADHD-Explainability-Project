# Import required libraries
# Pandas is used for data manipulation and analysis
# NumPy is used for numerical operations and handling missing values
import pandas as pd
import numpy as np

# Load the ADHD dataset from CSV file into a DataFrame
df = pd.read_csv("data/adhd.csv")

# Display first few rows to understand dataset structure
print("Dataset Loaded Successfully")
print(df.head())

# Check the number of rows and columns in the dataset
print("\nShape:")
print(df.shape)

# Display all column names
print("\nColumns:")
print(df.columns)

# Check for missing values in each column
print("\nMissing Values:")
print(df.isnull().sum())

# Display data types of all columns
print("\nData Types:")
print(df.dtypes)

# Explore the most frequent values in the HEIGHT column
# Helps identify unusual or invalid values
print("\nTop HEIGHT values:")
print(df["HEIGHT"].value_counts().head(10))

# Identify all ADHD-related columns in the dataset
print("\nADHD columns:")
for col in df.columns:
    if "adhd" in col.lower():
        print(col)

# Examine distribution of ADHD labels
print(df["ADHD_2324"].value_counts())

# Display statistical summary of ADHD_2324 column
print(df["ADHD_2324"].describe())

# Check distribution of ADHD severity categories
print(df["ADHDSev_2324"].value_counts())

# Replace invalid height codes (9990, 9999) with NaN
# These values represent missing/unknown data
df["HEIGHT"] = df["HEIGHT"].replace([9990, 9999], np.nan)
df["HEIGHT"] = df["HEIGHT"].fillna(df["HEIGHT"].median())
print("\nTop HEIGHT values:")
print(df["HEIGHT"].value_counts().head(10))

# Remove rows where ADHD_2324 contains invalid codes
# 95 and 99 typically represent missing or unknown responses
df = df[~df["ADHD_2324"].isin([95, 99])]

# Verify remaining ADHD labels
print(df["ADHD_2324"].value_counts())

# Keep only valid ADHD categories (1, 2, 3)
# Remove any other unexpected values
df = df[df["ADHD_2324"].isin([1, 2, 3])]

# Convert multiclass labels into binary labels
# 1 -> 0 (No ADHD)
# 3 -> 0 (Non-ADHD/Subclinical)
# 2 -> 1 (ADHD)
df["ADHD_2324"] = df["ADHD_2324"].replace({
    1: 0,
    3: 0,
    2: 1
})

# Ensure invalid height values are replaced again
# (This step is redundant but harmless)
df["HEIGHT"] = df["HEIGHT"].replace([9990, 9999], np.nan)

# Check final class distribution after preprocessing
print(df["ADHD_2324"].value_counts())

# Import train-test split function
from sklearn.model_selection import train_test_split

# Separate features (X) and target variable (y)
# X contains all predictor variables
# y contains the ADHD labels to be predicted
X = df.drop("ADHD_2324", axis=1)
y = df["ADHD_2324"]

# Split dataset into training and testing sets
# 80% data used for training
# 20% data reserved for testing
# random_state=42 ensures reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
