import dagshub
dagshub.init(
    repo_owner='gloryabacas',
    repo_name='Eksperimen_SML_Glorya',
    mlflow=True
)

import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

import mlflow
import mlflow.sklearn

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
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =====================================
# BUAT FOLDER ARTIFACTS
# =====================================
os.makedirs("artifacts", exist_ok=True)

# =====================================
# EXPERIMENT 1: BASELINE
# =====================================
mlflow.set_experiment("heart_disease_experiment")

with mlflow.start_run(run_name="LogReg_baseline"):

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall    = recall_score(y_test, y_pred, average="weighted")
    f1        = f1_score(y_test, y_pred, average="weighted")

    mlflow.log_metric("accuracy",  accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall",    recall)
    mlflow.log_metric("f1_score",  f1)

    # LOG MODEL
    model_path = "artifacts/baseline_model.pkl"
    joblib.dump(model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    mlflow.sklearn.log_model(model, artifact_path="model")

    # LOG PLOT CONFUSION MATRIX
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix - LogReg Baseline")
    plt.tight_layout()
    plot_path = "artifacts/confusion_matrix_baseline.png"
    plt.savefig(plot_path)
    plt.close()
    mlflow.log_artifact(plot_path, artifact_path="plots")

    print("\n===== BASELINE EVALUATION =====")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

# =====================================
# EXPERIMENT 2: TUNING (GridSearchCV)
# =====================================
mlflow.set_experiment("heart_disease_tuning")

param_grid = {
    "C": [0.01, 0.1, 1, 10, 100],
    "solver": ["liblinear", "lbfgs"]
}

grid = GridSearchCV(
    LogisticRegression(max_iter=1000),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy"
)

with mlflow.start_run(run_name="LogReg_GridSearch_v1"):

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall    = recall_score(y_test, y_pred, average="weighted")
    f1        = f1_score(y_test, y_pred, average="weighted")

    mlflow.log_param("best_C",         grid.best_params_["C"])
    mlflow.log_param("best_solver",    grid.best_params_["solver"])
    mlflow.log_metric("cv_best_score", grid.best_score_)
    mlflow.log_metric("accuracy",      accuracy)
    mlflow.log_metric("precision",     precision)
    mlflow.log_metric("recall",        recall)
    mlflow.log_metric("f1_score",      f1)

    # LOG MODEL
    best_model_path = "artifacts/best_model.pkl"
    joblib.dump(best_model, best_model_path)
    mlflow.log_artifact(best_model_path, artifact_path="best_model")
    mlflow.sklearn.log_model(best_model, artifact_path="best_model")

    # LOG PLOT CONFUSION MATRIX
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix - LogReg Tuned")
    plt.tight_layout()
    plot_path = "artifacts/confusion_matrix_tuning.png"
    plt.savefig(plot_path)
    plt.close()
    mlflow.log_artifact(plot_path, artifact_path="plots")

    print("\n===== BEST PARAMETER =====")
    print(grid.best_params_)

    print("\n===== TUNING EVALUATION =====")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

print("\nTraining selesai.")
print("Artifacts tersimpan di DagsHub!")