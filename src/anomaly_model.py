# src/anomaly_model.py
import os
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

DEFAULT_MODEL_PATH = "models/anomaly_iforest.pkl"

def train_iforest_from_csv(csv_path="bill_features.csv", save_path=DEFAULT_MODEL_PATH):
    df = pd.read_csv(csv_path)
    # choose meaningful numeric + binary features
    features = [
        "items_count", "numbers_found", "sum_numbers",
        "subtotal_val", "tax_val", "total_val",
        "has_subtotal_word", "has_tax_word", "has_total_word",
        "has_thank_word", "consistent_total"
    ]
    X = df[features].fillna(0)
    model = IsolationForest(contamination=0.15, random_state=42)
    model.fit(X)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model, save_path)
    print("✅ Trained & saved model to:", save_path)
    return model

def load_anomaly_model(path=DEFAULT_MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    model = joblib.load(path)
    print("✅ Anomaly model loaded successfully!")
    return model

if __name__ == "__main__":
    # train quick if executed directly
    if not os.path.exists("bill_features.csv"):
        raise SystemExit("No bill_features.csv found. Run src/extract_features.py first.")
    train_iforest_from_csv()
