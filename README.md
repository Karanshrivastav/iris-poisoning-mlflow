# IRIS Dataset Data Poisoning Experiment with MLflow

## 📌 Problem Statement

This project demonstrates how **data poisoning** affects machine-learning model performance using the **IRIS dataset**. Data poisoning refers to intentionally corrupting training data to mislead or degrade the performance of ML models.

We introduce poisoning levels of **0%**, **5%**, **10%**, and **50%** using randomly generated noise and observe model performance across multiple random seeds. All experiments are tracked using **MLflow** for reproducibility.

---

## 🎯 Objectives

* Integrate synthetic poisoning into the IRIS dataset.
* Train Logistic Regression models over multiple seeds.
* Track metrics, parameters, and artifacts using MLflow.
* Understand how poisoning levels impact validation accuracy.
* Discuss mitigation strategies for poisoned or low-quality datasets.

---

## 🧠 How I Approached the Problem

1. **Start with a clean baseline** using the original IRIS dataset.
2. **Simulate multiple poisoning levels** by injecting random noise into a certain percentage of samples.
3. **Ensure reproducibility** with multiple random seeds.
4. **Log everything** (params, metrics, models) using MLflow for comparison.
5. **Evaluate and compare results** across poisoning intensities.
6. **Document findings** and propose mitigation strategies.

---

## 📁 Project Structure

```
.
├── data/
│   └── (optional raw or poisoned data dumps)
├── mlruns/                 # MLflow tracking directory
├── train.py               # Main training script
├── poisoning.py           # Data poisoning utilities
├── requirements.txt       # Environment setup
└── README.md              # Documentation
```

---

## 🧪 Important Code Snippets

### 🔹 Data Poisoning Function

```python
def poison_data(X, y, poison_level, seed):
    np.random.seed(seed)
    X_poisoned = X.copy()
    num_samples = int(len(X) * poison_level)
    indices = np.random.choice(len(X), num_samples, replace=False)
    noise = np.random.normal(0, 1, X[indices].shape)
    X_poisoned[indices] += noise
    return X_poisoned, y
```

### 🔹 MLflow Logging

```python
with mlflow.start_run():
    mlflow.log_param("model", model_name)
    mlflow.log_param("poison_level", poison_level)
    mlflow.log_param("seed", seed)
    mlflow.log_metric("accuracy", acc)

    mlflow.sklearn.log_model(model, name="model")
```

---

## 📊 Results Summary

| Poison Level | Seed | Accuracy |
| ------------ | ---- | -------- |
| 0%           | 0    | 0.9667   |
| 5%           | 0    | 1.0000   |
| 10%          | 0    | 0.8000   |
| 50%          | 0    | 1.0000   |
| ...          | ...  | ...      |

### 📌 Key Observations

* Low noise (5%) sometimes **improves performance** because logistic regression is robust.
* Mid-level poison (10%) **degrades accuracy noticeably**.
* High-level poison (50%) produces **unstable and unpredictable results**, depending on seed.
* Poisoned data **increases variance** and makes performance inconsistent across seeds.

---

## 🛡️ Mitigation Strategies

### ✔ Data-Centric

* Use **outlier/anomaly detection** before training.
* Maintain **data provenance** and data source validation.
* Use **data redundancy** (collect more raw samples).
* Apply **data sanitization** using filters or clustering.

### ✔ Model-Centric

* Train **robust models** (e.g., RANSAC, robust regression).
* Use **regularization** to reduce sensitivity to noise.
* Use **ensemble models** to stabilize performance.

### ✔ Process-Centric

* Adopt **MLflow** and versioning to track anomalies.
* Enable **DVC** to track dataset versions.
* Automate **data quality checks** in CI/CD pipelines.

---

## 📈 How Data Quantity Requirements Change With Poor Data Quality

When quality drops:

* You need **more data** to achieve the same performance.
* The model requires **greater redundancy** to counteract noisy samples.
* **Generalization suffers**, requiring more diverse training examples.
* Data collection and cleaning become more important than model tuning.

In summary: **Bad data is more expensive than good data**.

---

## ▶️ Running the Training Script

```
python train.py \
  --model logreg \
  --seeds 0,1,2 \
  --poison_levels 0.0,0.05,0.1,0.5 \
  --experiment iris_poisoning_local
```

---

## 📦 Installation

```
pip install -r requirements.txt
```

`requirements.txt` example:

```
mlflow
scikit-learn
numpy
pandas
```

---

## 🚀 Final Notes

This repository serves as a practical demonstration of **data poisoning risks** in ML pipelines and the importance of **tracking experiments**, **verifying data integrity**, and **designing robust models**.

## Author
Karan Shrivastava
**Contact:** [karanshrivastava00@gmail.com](mailto:karanshrivastava00@gmail.com)
