import json
import io
import numpy as np
import pandas as pd
import pickle
import os
from .database import get_db, STORAGE_PATH
from .versioning import get_version_df

_model_cache = {}

def get_best_eval(project_id: int, target_column: str) -> dict:
    conn = get_db()
    row = conn.execute("""
        SELECT vbc.*, me.task_type
        FROM version_best_cache vbc
        JOIN ml_evaluations me ON vbc.evaluation_id = me.id
        WHERE vbc.project_id=? AND vbc.target_column=?
        ORDER BY vbc.best_score DESC
        LIMIT 1
    """, (project_id, target_column)).fetchone()
    conn.close()
    return dict(row) if row else None

def train_predictor(project_id: int, version_id: int, target_column: str) -> dict:
    df = get_version_df(version_id)
    if df is None or target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found"}

    df = df.dropna(subset=[target_column])
    y = df[target_column]
    X = df.drop(columns=[target_column])

    col_meta = {}
    encoders = {}
    for col in X.columns:
        if X[col].dtype == object or X[col].dtype.name == "category":
            cats = X[col].dropna().unique().tolist()
            encoders[col] = cats
            X[col] = pd.factorize(X[col])[0]
            col_meta[col] = {"type": "categorical", "categories": [str(c) for c in cats]}
        elif pd.api.types.is_numeric_dtype(X[col]):
            col_meta[col] = {"type": "numeric",
                             "min": float(X[col].min()),
                             "max": float(X[col].max()),
                             "mean": float(X[col].mean())}

    X = X.fillna(X.median(numeric_only=True))
    X = X.select_dtypes(include=[np.number])
    feature_names = list(X.columns)

    n_unique = y.nunique()
    if y.dtype == object or n_unique <= 20:
        task_type = "classification"
        from sklearn.preprocessing import LabelEncoder
        from sklearn.ensemble import RandomForestClassifier
        le = LabelEncoder()
        y_enc = le.fit_transform(y.astype(str))
        label_classes = le.classes_.tolist()
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X, y_enc)
        store = {"model": model, "le": le, "task_type": "classification",
                 "feature_names": feature_names, "col_meta": col_meta,
                 "encoders": encoders, "label_classes": label_classes,
                 "target_column": target_column}
    else:
        task_type = "regression"
        from sklearn.ensemble import RandomForestRegressor
        y_num = y.astype(float)
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X, y_num)
        store = {"model": model, "task_type": "regression",
                 "feature_names": feature_names, "col_meta": col_meta,
                 "encoders": encoders, "target_column": target_column}

    cache_key = f"{project_id}_{version_id}_{target_column}"
    _model_cache[cache_key] = store

    return {
        "status": "trained",
        "task_type": task_type,
        "feature_names": feature_names,
        "col_meta": col_meta,
        "target_column": target_column
    }

def run_prediction(project_id: int, version_id: int, target_column: str, input_data: dict) -> dict:
    cache_key = f"{project_id}_{version_id}_{target_column}"
    if cache_key not in _model_cache:
        result = train_predictor(project_id, version_id, target_column)
        if "error" in result:
            return result

    store = _model_cache[cache_key]
    model = store["model"]
    feature_names = store["feature_names"]
    col_meta = store["col_meta"]
    encoders = store["encoders"]
    task_type = store["task_type"]

    row = {}
    for feat in feature_names:
        val = input_data.get(feat, 0)
        if feat in encoders:
            cats = encoders[feat]
            row[feat] = cats.index(str(val)) if str(val) in cats else 0
        else:
            try:
                row[feat] = float(val)
            except:
                row[feat] = 0

    X_input = pd.DataFrame([row])[feature_names]

    if task_type == "classification":
        pred = model.predict(X_input)[0]
        proba = model.predict_proba(X_input)[0]
        le = store["le"]
        label = le.inverse_transform([pred])[0]
        confidence = float(max(proba))
        verdict = get_verdict(store["target_column"], label, confidence)
        result = {
            "prediction": str(label),
            "confidence": round(confidence * 100, 1),
            "probabilities": {str(c): round(float(p) * 100, 1)
                              for c, p in zip(store["label_classes"], proba)},
            "verdict": verdict
        }
    else:
        pred = float(model.predict(X_input)[0])
        result = {
            "prediction": round(pred, 4),
            "verdict": f"Predicted value: {round(pred, 4)}"
        }

    conn = get_db()
    conn.execute("""
        INSERT INTO predictions (project_id, version_id, model_name, input_json, output_json)
        VALUES (?,?,?,?,?)
    """, (project_id, version_id, "random_forest", json.dumps(input_data), json.dumps(result)))
    conn.commit()
    conn.close()

    return result

def run_batch_prediction(project_id: int, version_id: int, target_column: str,
                         file_bytes: bytes) -> bytes:
    cache_key = f"{project_id}_{version_id}_{target_column}"
    if cache_key not in _model_cache:
        result = train_predictor(project_id, version_id, target_column)
        if "error" in result:
            return None

    store = _model_cache[cache_key]
    model = store["model"]
    feature_names = store["feature_names"]
    encoders = store["encoders"]
    task_type = store["task_type"]

    df = pd.read_csv(io.BytesIO(file_bytes))
    X = df.copy()

    for feat in feature_names:
        if feat not in X.columns:
            X[feat] = 0
        elif feat in encoders:
            cats = encoders[feat]
            X[feat] = X[feat].apply(lambda v: cats.index(str(v)) if str(v) in cats else 0)

    X = X[feature_names].fillna(0)

    if task_type == "classification":
        preds = model.predict(X)
        le = store["le"]
        labels = le.inverse_transform(preds)
        probas = model.predict_proba(X)
        df["prediction"] = labels
        df["confidence"] = [round(float(max(p)) * 100, 1) for p in probas]
    else:
        preds = model.predict(X)
        df["prediction"] = [round(float(p), 4) for p in preds]

    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()

def get_verdict(target_col: str, label: str, confidence: float) -> str:
    col = target_col.lower()
    label_str = str(label).lower()
    conf_pct = round(confidence * 100, 1)

    if "churn" in col:
        if label_str in ["1", "yes", "true", "churn"]:
            return f"⚠️ High churn risk ({conf_pct}% confidence) — Consider retention action"
        else:
            return f"✅ Low churn risk ({conf_pct}% confidence) — Customer likely to stay"
    elif "survived" in col or "survival" in col:
        if label_str in ["1", "yes", "true"]:
            return f"✅ Predicted to survive ({conf_pct}% confidence)"
        else:
            return f"❌ Predicted not to survive ({conf_pct}% confidence)"
    elif "default" in col or "fraud" in col:
        if label_str in ["1", "yes", "true"]:
            return f"🚨 High risk ({conf_pct}% confidence) — Flag for review"
        else:
            return f"✅ Low risk ({conf_pct}% confidence) — Appears safe"
    else:
        return f"Prediction: {label} ({conf_pct}% confidence)"
