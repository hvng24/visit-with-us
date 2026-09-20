import os
import pandas as pd
import joblib
import mlflow
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

DATASET_REPO = "hvng24/tourism-dataset"
MODEL_REPO = "hvng24/tourism-model"

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tourism-package-prediction")

api = HfApi(token=os.getenv("HF_TOKEN"))

Xtrain = pd.read_csv(f"hf://datasets/{DATASET_REPO}/Xtrain.csv")
Xtest = pd.read_csv(f"hf://datasets/{DATASET_REPO}/Xtest.csv")
ytrain = pd.read_csv(f"hf://datasets/{DATASET_REPO}/ytrain.csv").values.ravel()
ytest = pd.read_csv(f"hf://datasets/{DATASET_REPO}/ytest.csv").values.ravel()

numeric_features = Xtrain.columns.tolist()
preprocessor = make_column_transformer((StandardScaler(), numeric_features))

xgb_model = xgb.XGBClassifier(random_state=42, n_jobs=-1, eval_metric="logloss")
model_pipeline = make_pipeline(preprocessor, xgb_model)

param_grid = {
    "xgbclassifier__n_estimators": [100, 200],
    "xgbclassifier__max_depth": [3, 5],
    "xgbclassifier__learning_rate": [0.05, 0.1],
    "xgbclassifier__scale_pos_weight": [1, 3],
}

with mlflow.start_run():
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, n_jobs=-1, scoring="roc_auc")
    grid_search.fit(Xtrain, ytrain)

    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        with mlflow.start_run(nested=True):
            mlflow.log_params(results["params"][i])
            mlflow.log_metric("mean_roc_auc", results["mean_test_score"][i])

    print(f"Best parameters: {grid_search.best_params_}")
    mlflow.log_params(grid_search.best_params_)
    best_model = grid_search.best_estimator_

    y_pred_train = best_model.predict(Xtrain)
    y_pred_test = best_model.predict(Xtest)
    y_proba_train = best_model.predict_proba(Xtrain)[:, 1]
    y_proba_test = best_model.predict_proba(Xtest)[:, 1]

    metrics = {
        "train_accuracy": accuracy_score(ytrain, y_pred_train),
        "test_accuracy": accuracy_score(ytest, y_pred_test),
        "train_precision": precision_score(ytrain, y_pred_train, zero_division=0),
        "test_precision": precision_score(ytest, y_pred_test, zero_division=0),
        "train_recall": recall_score(ytrain, y_pred_train, zero_division=0),
        "test_recall": recall_score(ytest, y_pred_test, zero_division=0),
        "train_f1_score": f1_score(ytrain, y_pred_train, zero_division=0),
        "test_f1_score": f1_score(ytest, y_pred_test, zero_division=0),
        "train_roc_auc": roc_auc_score(ytrain, y_proba_train),
        "test_roc_auc": roc_auc_score(ytest, y_proba_test),
    }
    mlflow.log_metrics(metrics)

    print(classification_report(ytest, y_pred_test, target_names=["No Purchase", "Purchase"]))
    print(confusion_matrix(ytest, y_pred_test))
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    model_path = "best_tourism_model_v1.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")

    try:
        api.repo_info(repo_id=MODEL_REPO, repo_type="model")
        print(f"Repository '{MODEL_REPO}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Repository '{MODEL_REPO}' not found. Creating it...")
        create_repo(repo_id=MODEL_REPO, repo_type="model", private=False, token=os.getenv("HF_TOKEN"))

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=model_path,
        repo_id=MODEL_REPO,
        repo_type="model",
    )
    print(f"Model uploaded to Hugging Face Hub: {MODEL_REPO}")

print("Model training and registration completed successfully.")
