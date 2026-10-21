@echo off
echo Starting MLflow UI on http://127.0.0.1:5000 ...
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
pause
