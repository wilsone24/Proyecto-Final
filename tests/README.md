# Local Testing Guide

## Prerequisites

### 1. Activate the virtual environment
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1
```

### 2. Install dependencies
```powershell
pip install pandas matplotlib scikit-learn xgboost mlflow pyspark delta-spark
```

### 3. Place the dataset
Copy `cardio_train.csv` (semicolon-delimited) into:
```
data/cardio_train.csv
```

---

## Running the tests

### ML training pipeline (no Spark/Java required)
```powershell
.\venv\Scripts\python.exe tests/test_training.py
```
Outputs metrics, CV ROC-AUC, and logs everything to a local SQLite MLflow DB.

### Silver ETL pipeline (requires Java + PySpark)
```powershell
.\venv\Scripts\python.exe tests/test_silver.py
```
Reads the raw CSV, runs all silver transformations, validates the schema and null counts, and writes a local Delta table to `data/silver_delta/`.

---

## MLflow UI

### Launch against the local SQLite DB
```powershell
.\venv\Scripts\mlflow.exe ui --backend-store-uri sqlite:///mlruns/mlflow.db --default-artifact-root ./mlruns/artifacts
```
Then open: http://127.0.0.1:5000

---

## Java / PySpark compatibility (Java 17+)

PySpark 4.x requires Java. The `local_spark.py` session builder already injects the
required `--add-opens` JVM flags for Java 17/21/25. No manual configuration needed.

### Verify Java is detected
```powershell
java -version
```

### Verify PySpark version
```powershell
.\venv\Scripts\python.exe -c "import pyspark; print(pyspark.__version__)"
```

---

## Inspect local Delta output

### Silver Delta table (after running test_silver.py)
```powershell
.\venv\Scripts\python.exe -c "
from tests.local_spark import spark
spark.read.format('delta').load('data/silver_delta').show(10)
"
```

---

## Dependency versions (tested)

| Package      | Version |
|--------------|---------|
| pyspark      | 4.1.1   |
| delta-spark  | 4.x     |
| Java         | 25.0.3  |
| mlflow       | 2.x     |
| xgboost      | latest  |
| scikit-learn | latest  |
