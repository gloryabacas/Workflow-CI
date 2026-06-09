
import pandas as pd
import numpy as np
import kagglehub
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

def preprocess():

    # Download dataset
    path = kagglehub.dataset_download(
        "redwankarimsony/heart-disease-data"
    )

    # Load dataset
    df = pd.read_csv(
        os.path.join(path, "heart_disease_uci.csv")
    )

    # Hapus kolom ID
    df.drop(columns=["id"], inplace=True)

    # Membuat target biner
    df["target"] = (df["num"] > 0).astype(int)

    # Hapus target lama
    df.drop(columns=["num"], inplace=True)

    # Missing value numerik
    numeric_cols = df.select_dtypes(
        include=["int64", "float64"]
    ).columns

    for col in numeric_cols:
        df[col] = df[col].fillna(
            df[col].median()
        )

    # Missing value kategorikal
    categorical_cols = df.select_dtypes(
        include=["object"]
    ).columns

    for col in categorical_cols:
        df[col] = df[col].fillna(
            df[col].mode()[0]
        )

    # Hapus duplikat
    df.drop_duplicates(inplace=True)

    # Encoding
    le = LabelEncoder()

    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    # Outlier Handling (IQR)
    numeric_cols = [
        col for col in df.columns
        if col != "target"
    ]

    for col in numeric_cols:

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - (1.5 * IQR)
        upper = Q3 + (1.5 * IQR)

        df = df[
            (df[col] >= lower)
            &
            (df[col] <= upper)
        ]

    # Scaling
    scaler = StandardScaler()

    feature_cols = [
        col for col in df.columns
        if col != "target"
    ]

    df[feature_cols] = scaler.fit_transform(
        df[feature_cols]
    )

    # Save
    df.to_csv(
        "dataset_preprocessing.csv",
        index=False
    )

    print("Preprocessing selesai.")
    print(df.shape)

if __name__ == "__main__":
    preprocess()
