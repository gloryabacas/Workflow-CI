import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import mlflow
import mlflow.sklearn

# =====================================
# KONEKSI KE DAGSHUB VIA TOKEN
# =====================================
import dagshub
dagshub.init(
    repo_owner='gloryabacas',
    repo_name='Eksperimen_SML_Glorya',
    mlflow=True
)

# =====================================
# LOAD DATA
# =====================================
df = pd.read_csv("dataset_preprocessing.csv")

X = df.drop("target", axis=1)
y = df["target"]

# =====================================
# SPLIT DATA
# =====================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================
# BUAT FOLDER ARTIFACTS
# =====================================
os.makedirs("artifacts", exist_ok=True)

# =====================================
# TRAINING & LOGGING
# =====================================
mlflow.set_experiment("heart_disease_experiment")

with mlflow.start_run(run_name="LogReg_CI_run"):

    # Training
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluasi
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall    = recall_score(y_test, y_pred, average="weighted")
    f1        = f1_score(y_test, y_pred, average="weighted")

    # Log metrics
    mlflow.log_metric("accuracy",  accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall",    recall)
    mlflow.log_metric("f1_score",  f1)

    # Log model
    model_path = "artifacts/baseline_model.pkl"
    joblib.dump(model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    mlflow.sklearn.log_model(model, artifact_path="model")

    # Log confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix - Heart Disease")
    plt.tight_layout()
    plot_path = "artifacts/confusion_matrix.png"
    plt.savefig(plot_path)
    plt.close()
    mlflow.log_artifact(plot_path, artifact_path="plots")

    print("\n===== MODEL EVALUATION =====")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

print("\nTraining selesai. Artifacts tersimpan.")
