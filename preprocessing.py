import pandas as pd
import numpy as np

df = pd.read_csv("data/adhd.csv")

print("Dataset Loaded Successfully")
print(df.head())

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nData Types:")
print(df.dtypes)

# print(df["HEIGHT"].value_counts().head(10))

# for col in df.columns:
#     if "adhd" in col.lower():
#         print(col)

# print(df.describe())

print("\nTop HEIGHT values:")
print(df["HEIGHT"].value_counts().head(10))

print("\nADHD columns:")
for col in df.columns:
    if "adhd" in col.lower():
        print(col)

print(df["ADHD_2324"].value_counts())
print(df["ADHD_2324"].describe())
print(df["ADHDSev_2324"].value_counts())
df["HEIGHT"] = df["HEIGHT"].replace([9990, 9999], np.nan)

df = df[~df["ADHD_2324"].isin([95, 99])]
print(df["ADHD_2324"].value_counts())

df = df[df["ADHD_2324"].isin([1,2,3])]

df["ADHD_2324"] = df["ADHD_2324"].replace({
    1: 0,
    3: 0,
    2: 1
})

df["HEIGHT"] = df["HEIGHT"].replace([9990, 9999], np.nan)
print(df["ADHD_2324"].value_counts())
from sklearn.model_selection import train_test_split

X = df.drop("ADHD_2324", axis=1)
y = df["ADHD_2324"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)