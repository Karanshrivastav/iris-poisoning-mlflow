# 🚀 How to Run Everything (Step-by-Step)

Follow these steps to reproduce the full experiment pipeline --- from
installing requirements to visualizing results in MLflow.

------------------------------------------------------------------------

## **1️⃣ Clone the Repository**

``` bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

------------------------------------------------------------------------

## **2️⃣ Create and Activate Virtual Environment**

``` bash
python3 -m venv venv
source venv/bin/activate        # Linux / macOS
# or
venv\Scripts\activate           # Windows
```

------------------------------------------------------------------------

## **3️⃣ Install Dependencies**

``` bash
pip install --upgrade pip
pip install -r requirements.txt
```

------------------------------------------------------------------------

## **4️⃣ Run the Training Script**

This command trains models on all poisoning levels (0%, 5%, 10%, 50%)\
across multiple seeds, logging everything into MLflow.

``` bash
python train.py \
  --model logreg \
  --seeds 0,1,2 \
  --poison_levels 0.0,0.05,0.1,0.5 \
  --experiment iris_poisoning_local
```

Expected output:

    Run finished: poison=0.0, acc=0.9667
    Run finished: poison=0.05, acc=1.0000
    Run finished: poison=0.1, acc=0.8000
    Run finished: poison=0.5, acc=1.0000
    ...

This generates **12 MLflow runs**\
(3 seeds × 4 poison levels).

------------------------------------------------------------------------

## **5️⃣ Launch MLflow Tracking UI**

### ⚠️ If running on a cloud VM (GCP / AWS / Azure):

MLflow requires allowing your external host.

Run:

``` bash
mlflow ui \
  --host 0.0.0.0 \
  --port 5000 \
  --allowed-hosts "*"
```

### Then open in your browser:

    http://<your-external-ip>:5000

Example:

    http://136.111.87.116:5000

------------------------------------------------------------------------

## **6️⃣ View Results in MLflow**

Inside MLflow UI:

1.  Open experiment **iris_poisoning_local**
2.  Select all ☐ runs
3.  Click **Compare**
4.  Go to the **Charts** tab
5.  Create **Line Chart**
    -   **X-axis:** `poison_fraction`\
    -   **Y-axis:** `metrics.accuracy`

This visualizes how accuracy changes with different poisoning levels.

------------------------------------------------------------------------

## **7️⃣ (Optional) Clean Previous MLflow Runs**

If you want to start fresh:

``` bash
rm -rf mlruns/
```

------------------------------------------------------------------------

## **8️⃣ Stop MLflow UI**

Press:

    CTRL + C

------------------------------------------------------------------------
