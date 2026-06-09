import dagshub
dagshub.init(
    repo_owner='gloryabacas',
    repo_name='Eksperimen_SML_Glorya',
    mlflow=True
)

import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

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
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================
# GRID SEARCH
# =====================================

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

# =====================================
# MLFLOW
# =====================================

mlflow.set_experiment(
    "heart_disease_tuning"
)

with mlflow.start_run():

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(
        y_test,
        y_pred,
        average="weighted"
    )
    rec = recall_score(
        y_test,
        y_pred,
        average="weighted"
    )
    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted"
    )

    mlflow.log_param(
        "best_C",
        grid.best_params_["C"]
    )

    mlflow.log_param(
        "best_solver",
        grid.best_params_["solver"]
    )

    mlflow.log_metric(
        "accuracy",
        acc
    )

    mlflow.log_metric(
        "precision",
        prec
    )

    mlflow.log_metric(
        "recall",
        rec
    )

    mlflow.log_metric(
        "f1_score",
        f1
    )

    mlflow.sklearn.log_model(
        best_model,
        "best_model"
    )
    # Run kedua dengan model berbeda
from sklearn.ensemble import RandomForestClassifier

with mlflow.start_run(run_name="RandomForest_baseline"):
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    # log metrics...
    

    print("\n===== BEST PARAMETER =====")
    print(grid.best_params_)

    print("\n===== EVALUATION =====")
    print("Accuracy :", acc)
    print("Precision:", prec)
    print("Recall   :", rec)
    print("F1 Score :", f1)
