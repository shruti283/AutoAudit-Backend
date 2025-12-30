# src/api.py
import os
import uvicorn
import joblib
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.extract_features import extract_bill_features
from src.detect_anomaly import detect_anomaly

# ---------------------------
# FastAPI init
# ---------------------------
app = FastAPI(title="AutoAudit Hybrid Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ---------------------------
# Classifier wrapper (robust)
# Accepts:
# - sklearn Pipeline / estimator (has predict)
# - tuple/list containing (vectorizer, model) or (model, vectorizer)
# ---------------------------
class ClassifierWrapper:
    def __init__(self, raw):
        self.model = None
        self.vectorizer = None

        if raw is None:
            raise ValueError("raw classifier is None")

        # If it's already an estimator/pipeline with predict()
        if hasattr(raw, "predict") and not isinstance(raw, (tuple, list)):
            self.model = raw
            return

        # If it's a tuple/list, try to find vectorizer & model
        if isinstance(raw, (tuple, list)):
            # heuristics: vectorizer has transform() and maybe vocabulary_
            # model has predict()
            vec = None
            mod = None
            for item in raw:
                if item is None:
                    continue
                if hasattr(item, "predict"):
                    mod = item
                elif hasattr(item, "transform"):
                    vec = item
            # fallback: first is vectorizer, second is model
            if mod is None and len(raw) >= 2 and hasattr(raw[1], "predict"):
                mod = raw[1]
            if vec is None and len(raw) >= 1 and hasattr(raw[0], "transform"):
                vec = raw[0]

            if mod is None:
                # maybe the tuple is (model, vectorizer) reversed; try reversing
                for item in raw[::-1]:
                    if hasattr(item, "predict"):
                        mod = item
                        break

            self.model = mod
            self.vectorizer = vec

            if self.model is None:
                raise ValueError("Could not find model inside the supplied classifier object")
            return

        # otherwise unsupported
        raise ValueError("Unsupported classifier object type")

    def predict(self, texts):
        """
        texts: list[str]
        returns: array-like predictions
        """
        if self.vectorizer is not None:
            X = self.vectorizer.transform(texts)
            return self.model.predict(X)
        else:
            # some pipelines expect raw text list; some expect 2D array-like
            return self.model.predict(texts)

    def predict_proba(self, texts):
        if hasattr(self.model, "predict_proba"):
            if self.vectorizer is not None:
                X = self.vectorizer.transform(texts)
                return self.model.predict_proba(X)
            else:
                return self.model.predict_proba(texts)
        return None


# ---------------------------
# Load models safely
# ---------------------------
ANOMALY_MODEL = None
CLASSIFIER = None

# anomaly
try:
    ANOMALY_MODEL = joblib.load("models/anomaly_model.pkl")
    print("✅ Loaded models/anomaly_model.pkl")
except Exception as e:
    ANOMALY_MODEL = None
    print("⚠️ Could not load anomaly_model.pkl:", e)

# classifier (may be pipeline or tuple)
try:
    raw_clf = joblib.load("models/receipt_classifier.pkl")
    CLASSIFIER = ClassifierWrapper(raw_clf)
    print("✅ Loaded and wrapped models/receipt_classifier.pkl")
except Exception as e:
    CLASSIFIER = None
    print("⚠️ Could not load/wrap receipt_classifier.pkl:", e)


# ---------------------------
# Root
# ---------------------------
@app.get("/")
def root():
    return {"message": "AutoAudit Hybrid Backend Running ✔"}


# ---------------------------
# Verify endpoint
# ---------------------------
@app.post("/verify")
async def verify(file: UploadFile = File(...)):
    # save uploaded file
    try:
        contents = await file.read()
        path = os.path.join(UPLOAD_DIR, file.filename)
        with open(path, "wb") as f:
            f.write(contents)
    except Exception as e:
        return JSONResponse({"error": "File saving failed", "detail": str(e)}, status_code=500)

    # extract features
    try:
        feat = extract_bill_features(path)
        if feat is None:
            raise ValueError("Feature extraction returned None")
    except Exception as e:
        # cleanup then return
        try: os.remove(path)
        except: pass
        return JSONResponse({"error": "Feature extraction failed", "detail": str(e)}, status_code=500)
    finally:
        # remove uploaded file (we already have features)
        try: os.remove(path)
        except: pass

    # --- classifier prediction (text based) ---
    classifier_pred = None
    classifier_confidence = None
    try:
        if CLASSIFIER is not None:
            raw_text = feat.get("raw_text", "")
            # ensure we pass a list
            preds = CLASSIFIER.predict([raw_text])
            classifier_pred = int(preds[0])
            probs = CLASSIFIER.predict_proba([raw_text])
            if probs is not None:
                classifier_confidence = float(np.max(probs[0]))
    except Exception as e:
        print("❌ Classifier prediction error:", e)
        classifier_pred = None
        classifier_confidence = None

    # --- anomaly detection (expects FEATURE_COLUMNS) ---
    anomaly_score = None
    anomaly_label = None
    try:
        if ANOMALY_MODEL is not None:
            # build dict with only FEATURE_COLUMNS (ordered)
            model_input = {k: feat.get(k, 0) for k in FEATURE_COLUMNS}
            anomaly_score, anomaly_label = detect_anomaly(ANOMALY_MODEL, model_input)
            if anomaly_score is not None:
                anomaly_score = float(anomaly_score)
            if anomaly_label is not None:
                anomaly_label = int(anomaly_label)
    except Exception as e:
        print("❌ Anomaly detection failed:", e)
        anomaly_score = None
        anomaly_label = None

    # --- hybrid decision logic ---
    final_verdict = "UNKNOWN"
    # classifier: treat 0 or -1 as fake; 1 as real (handle both conventions)
    if classifier_pred is not None:
        if classifier_pred in (0, -1):
            final_verdict = "FAKE ❌"
        else:
            # classifier says real -> confirm anomaly
            if anomaly_label == -1:
                final_verdict = "FAKE ❌"
            else:
                final_verdict = "REAL ✅"
    else:
        # no classifier -> rely on anomaly only
        if anomaly_label == -1:
            final_verdict = "FAKE ❌"
        elif anomaly_label == 1:
            final_verdict = "REAL ✅"
        else:
            final_verdict = "UNKNOWN"

    # rule_check (consistency)
    rule_check = bool(feat.get("consistent_total", False))

    # return lots of debugging info (frontend can hide if not needed)
    return {
        "filename": feat.get("filename"),
        "bill_text": feat.get("raw_text", ""),
        "rule_check": rule_check,

        "classifier_pred": classifier_pred,
        "classifier_confidence": classifier_confidence,

        "anomaly_score": anomaly_score,
        "anomaly_label": anomaly_label,

        "final_verdict": final_verdict,
        "features": feat
    }


# ---------------------------
# Run server if executed directly
# ---------------------------
if __name__ == "__main__":
    uvicorn.run("src.api:app", host="127.0.0.1", port=8000, reload=True)
