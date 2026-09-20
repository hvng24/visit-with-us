import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import HfApi

DATASET_PATH = "hf://datasets/hvng24/tourism-dataset/tourism.csv"
REPO_ID = "hvng24/tourism-dataset"

api = HfApi(token=os.getenv("HF_TOKEN"))

df = pd.read_csv(DATASET_PATH)

unnamed_cols = [c for c in df.columns if c.startswith("Unnamed")]
df = df.drop(columns=unnamed_cols)
df = df.drop(columns=["CustomerID"])

df["Gender"] = df["Gender"].str.strip().replace({"Fe Male": "Female", "Fe male": "Female"})

numerical_cols = df.select_dtypes(include=[np.number]).columns
for col in numerical_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

categorical_cols = df.select_dtypes(include=["object"]).columns
for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])

categorical_features = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "ProductPitched",
    "MaritalStatus",
    "Designation",
]
for col in categorical_features:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

target_col = "ProdTaken"
X = df.drop(columns=[target_col])
y = df[target_col]

print(f"Features shape: {X.shape}, Target shape: {y.shape}")
print(y.value_counts())

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train size: {Xtrain.shape[0]}, Test size: {Xtest.shape[0]}")

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

for file_name in ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]:
    api.upload_file(
        path_or_fileobj=file_name,
        path_in_repo=file_name,
        repo_id=REPO_ID,
        repo_type="dataset",
    )
    print(f"Uploaded {file_name} to {REPO_ID}")

print("Data preparation completed successfully.")
