import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import mlflow
import mlflow.sklearn
import dagshub

# 1. Mengaktifkan autolog sebelum melatih model sesuai ketentuan Dicoding
mlflow.autolog()

# 2. Inisialisasi DagsHub Tracking
dagshub.init(
    repo_owner='gloryabacas',
    repo_name='Workflow-CI',
    mlflow=True
)

# 3. Membaca dataset preprocessing
df = pd.read_csv("dataset_preprocessing.csv")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

os.makedirs("artifacts", exist_ok=True)

# 4. Memberikan nama eksperimen kriteria basic
mlflow.set_experiment("heart_disease_experiment")

# 5. Eksekusi run tracking MLflow
with mlflow.start_run(run_name="LogReg_CI_run"):

    # Inisialisasi model dasar tanpa hyperparameter tuning
    model = LogisticRegression(max_iter=1000, random_state=42)
    
    # Proses training model (autolog otomatis mencatat parameter, model, dan metrik)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Menyimpan file .pkl secara lokal ke dalam folder artifacts untuk workflow
    model_path = "artifacts/baseline_model.pkl"
    joblib.dump(model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    
    # Pembuatan dan pencatatan plot Confusion Matrix tambahan
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix - Heart Disease")
    plt.tight_layout()
    
    plot_path = "artifacts/confusion_matrix.png"
    plt.savefig(plot_path)
    plt.close()
    mlflow.log_artifact(plot_path, artifact_path="plots")

print("\nTraining selesai dengan mlflow.autolog(). Artifacts tersimpan.")
