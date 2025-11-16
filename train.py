# train.py
import os
import argparse
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from poison import feature_poison
from eval_utils import plot_confusion_matrix, evaluate_and_log

def load_data():
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target
    target_names = list(iris.target_names)
    return X, y, target_names

def build_model(model_type="logreg", random_state=42):
    if model_type == "logreg":
        clf = LogisticRegression(max_iter=500, random_state=random_state)
    else:
        clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clf)
    ])
    return pipe

def run_experiment(poison_fraction, model_type, run_name, seed, mlflow_experiment=None, out_dir="outputs"):
    X, y, target_names = load_data()
    # split once (use a clean test set)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    # Poison only training set
    X_train_poisoned = feature_poison(X_train, fraction=poison_fraction, random_state=seed)
    model = build_model(model_type=model_type, random_state=seed)
    # Start MLflow run
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("poison_fraction", poison_fraction)
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("seed", seed)
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))

        model.fit(X_train_poisoned, y_train)
        preds = model.predict(X_test)
        metrics = evaluate_and_log(y_test, preds)
        mlflow.log_metrics(metrics)
        # Save confusion matrix
        os.makedirs(out_dir, exist_ok=True)
        cm_path = os.path.join(out_dir, f"confusion_{int(poison_fraction*100)}_{model_type}_{seed}.png")
        plot_confusion_matrix(y_test, preds, labels=list(range(len(target_names))), out_path=cm_path)
        mlflow.log_artifact(cm_path, artifact_path="confusion_matrices")

        # Save classification report as text artifact
        report = classification_report(y_test, preds, target_names=target_names)
        with open(os.path.join(out_dir, "classification_report.txt"), "w") as f:
            f.write(report)
        mlflow.log_artifact(os.path.join(out_dir, "classification_report.txt"))

        # log model
        mlflow.sklearn.log_model(model, artifact_path="model")
        print(f"Run finished: poison={poison_fraction}, acc={metrics['accuracy']:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="logreg", choices=["logreg", "rf"])
    parser.add_argument("--seeds", default="0,1,2", help="comma-separated seeds for repeated runs")
    parser.add_argument("--poison_levels", default="0.0,0.05,0.1,0.5")
    parser.add_argument("--experiment", default="iris_poisoning_test")
    args = parser.parse_args()

    mlflow.set_experiment(args.experiment)

    seeds = [int(s) for s in args.seeds.split(",")]
    levels = [float(x) for x in args.poison_levels.split(",")]

    for seed in seeds:
        for level in levels:
            run_name = f"poison_{int(level*100)}_seed_{seed}_{args.model}"
            run_experiment(poison_fraction=level, model_type=args.model, run_name=run_name, seed=seed)

if __name__ == "__main__":
    main()
