# src/detect_anomaly.py

import numpy as np

def detect_anomaly(model, features):
    """
    For supervised classifier:
    features = dict with numeric values extracted from OCR.
    Model returns: 1 = REAL, 0 = FAKE
    """
    X = np.array([[
        features["subtotal_val"],
        features["tax_val"],
        features["total_val"],
        features["has_thank_word"]
    ]])

    pred = model.predict(X)[0]  # 1 or 0
    return int(pred)
