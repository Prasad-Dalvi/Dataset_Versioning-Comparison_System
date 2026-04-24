import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
from .database import get_db
from .versioning import get_version_df

CLASSIFICATION_MODELS = {}
REGRESSION_MODELS = {}

def _load_models(task_type: str):
    from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                                   ExtraTreesClassifier, RandomForestRegressor,
                                   GradientBoostingRegressor, ExtraTreesRegressor)
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.svm import SVC, SVR
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.neural_network import MLPClassifier, MLPRegressor

    if task_type == "classification":
        models = {
            "random_forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            "gradient_boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
            "logistic_regression": LogisticRegression(max_iter=500, random_state=42),
            "svm": SVC(probability=True, random_state=42),
            "knn": KNeighborsClassifier(n_neighbors=5),
            "naive_bayes": GaussianNB(),
            "decision_tree": DecisionTreeClassifier(random_state=42),
            "extra_trees": ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            "mlp": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42),
        }
    else:
        models = {
            "random_forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            "gradient_boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
            "logistic_regression": LinearRegression(),
            "svm": SVR(),
            "knn": KNeighborsRegressor(n_neighbors=5),
            "naive_bayes": LinearRegression(),  # placeholder
            "decision_tree": DecisionTreeRegressor(random_state=42),
            "extra_trees": ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            "mlp": MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42),
        }

    try:
        import xgboost as xgb
        if task_type == "classification":
            models["xgboost"] = xgb.XGBClassifier(n_estimators=100, random_state=42,
                                                    eval_metric="logloss", use_label_encoder=False)
        else:
            models["xgboost"] = xgb.XGBRegressor(n_estimators=100, random_state=42)
    except ImportError:
        pass

    return models

def run_evaluation(eval_id: int):
    conn = get_db()
    ev = conn.execute("SELECT * FROM ml_evaluations WHERE id=?", (eval_id,)).fetchone()
    conn.close()
    if not ev:
        return

    ev = dict(ev)
    version_id = ev["version_id"]
    target_col = ev["target_column"]

    def update_status(status, progress=None, current_model=None, error=None):
        conn2 = get_db()
        updates = ["status=?"]
        vals = [status]
        if progress is not None:
            updates.append("progress=?")
            vals.append(progress)
        if current_model is not None:
            updates.append("current_model=?")
            vals.append(current_model)
        if error is not None:
            updates.append("error_message=?")
            vals.append(error)
        if status == "running" and not ev.get("started_at"):
            updates.append("started_at=datetime('now')")
        if status in ["completed", "failed"]:
            updates.append("completed_at=datetime('now')")
        vals.append(eval_id)
        conn2.execute(f"UPDATE ml_evaluations SET {', '.join(updates)} WHERE id=?", vals)
        conn2.commit()
        conn2.close()

    try:
        update_status("running", 0, "Loading data")
        df = get_version_df(version_id)
        if df is None or target_col not in df.columns:
            update_status("failed", error=f"Target column '{target_col}' not found")
            return

        df = df.dropna(subset=[target_col])
        y = df[target_col]
        X = df.drop(columns=[target_col])

        # Encode categoricals
        for col in X.select_dtypes(include=["object", "category"]).columns:
            X[col] = pd.factorize(X[col])[0]
        X = X.fillna(X.median(numeric_only=True))
        X = X.select_dtypes(include=[np.number])

        # Determine task type
        n_unique = y.nunique()
        if y.dtype == object or n_unique <= 20:
            task_type = "classification"
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(y.astype(str))
        else:
            task_type = "regression"
            y = y.astype(float)

        conn2 = get_db()
        conn2.execute("UPDATE ml_evaluations SET task_type=? WHERE id=?", (task_type, eval_id))
        conn2.commit()
        conn2.close()

        from sklearn.model_selection import cross_val_score, train_test_split
        from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                                      r2_score, mean_squared_error, confusion_matrix)
        from sklearn.preprocessing import label_binarize
        import warnings
        warnings.filterwarnings("ignore")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        models = _load_models(task_type)
        feature_names = list(X.columns)
        all_metrics = []
        total = len(models)

        for i, (name, model) in enumerate(models.items()):
            update_status("running", int((i / total) * 90), f"Training {name}")
            conn3 = get_db()
            conn3.execute("""
                INSERT OR IGNORE INTO ml_model_results (evaluation_id, model_name, status)
                VALUES (?,?,?)
            """, (eval_id, name, "training"))
            conn3.commit()
            conn3.close()

            try:
                t0 = time.time()
                model.fit(X_train, y_train)
                train_time = time.time() - t0

                y_pred = model.predict(X_test)
                metrics = {"training_time": round(train_time, 3)}

                if task_type == "classification":
                    metrics["accuracy"] = round(float(accuracy_score(y_test, y_pred)), 4)
                    metrics["f1"] = round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4)
                    try:
                        if hasattr(model, "predict_proba"):
                            y_prob = model.predict_proba(X_test)
                            if y_prob.shape[1] == 2:
                                metrics["auc_roc"] = round(float(roc_auc_score(y_test, y_prob[:, 1])), 4)
                            else:
                                metrics["auc_roc"] = round(float(roc_auc_score(
                                    label_binarize(y_test, classes=np.unique(y_train)),
                                    y_prob, multi_class="ovr", average="weighted"
                                )), 4)
                        else:
                            metrics["auc_roc"] = metrics["accuracy"]
                    except:
                        metrics["auc_roc"] = metrics["accuracy"]
                    cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
                    metrics["cv_score"] = round(float(cv_scores.mean()), 4)
                    metrics["cv_std"] = round(float(cv_scores.std()), 4)
                    cm = confusion_matrix(y_test, y_pred)
                    cm_data = cm.tolist()
                    metrics["primary_score"] = metrics["accuracy"]
                else:
                    metrics["r2"] = round(float(r2_score(y_test, y_pred)), 4)
                    metrics["mse"] = round(float(mean_squared_error(y_test, y_pred)), 4)
                    metrics["rmse"] = round(float(np.sqrt(metrics["mse"])), 4)
                    metrics["f1"] = metrics["r2"]
                    metrics["accuracy"] = max(0, metrics["r2"])
                    metrics["auc_roc"] = max(0, metrics["r2"])
                    cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")
                    metrics["cv_score"] = round(float(cv_scores.mean()), 4)
                    metrics["cv_std"] = round(float(cv_scores.std()), 4)
                    cm_data = []
                    metrics["primary_score"] = metrics["r2"]

                fi = {}
                if hasattr(model, "feature_importances_"):
                    fi = dict(zip(feature_names, [round(float(x), 4) for x in model.feature_importances_]))
                elif hasattr(model, "coef_"):
                    coef = model.coef_
                    if len(coef.shape) > 1:
                        coef = np.abs(coef).mean(axis=0)
                    fi = dict(zip(feature_names, [round(float(x), 4) for x in np.abs(coef)]))

                all_metrics.append({"name": name, **metrics, "feature_importances": fi})

                conn4 = get_db()
                conn4.execute("""
                    UPDATE ml_model_results
                    SET metrics_json=?, feature_importances_json=?, confusion_matrix_json=?,
                        training_time=?, status=?
                    WHERE evaluation_id=? AND model_name=?
                """, (json.dumps(metrics), json.dumps(fi), json.dumps(cm_data),
                      train_time, "completed", eval_id, name))
                conn4.commit()
                conn4.close()
            except Exception as e:
                conn4 = get_db()
                conn4.execute("""
                    UPDATE ml_model_results SET status=?, metrics_json=?
                    WHERE evaluation_id=? AND model_name=?
                """, ("failed", json.dumps({"error": str(e)}), eval_id, name))
                conn4.commit()
                conn4.close()

        # Ensemble
        update_status("running", 92, "Computing ensemble")
        completed = [m for m in all_metrics if "error" not in str(m)]
        if completed:
            # Simple average
            avg_metrics = {}
            for key in ["accuracy", "f1", "auc_roc", "cv_score", "r2"]:
                vals = [m[key] for m in completed if key in m and isinstance(m[key], (int, float))]
                if vals:
                    avg_metrics[key] = round(float(np.mean(vals)), 4)

            # Weighted average by cv_score
            total_weight = sum(max(m.get("cv_score", 0), 0) for m in completed)
            w_metrics = {}
            for key in ["accuracy", "f1", "auc_roc", "r2"]:
                vals = []
                for m in completed:
                    if key in m and isinstance(m[key], (int, float)):
                        w = max(m.get("cv_score", 0), 0) / max(total_weight, 1e-6)
                        vals.append(m[key] * w)
                if vals:
                    w_metrics[key] = round(float(sum(vals)), 4)

            combined = {
                "simple_avg": avg_metrics,
                "weighted_avg": w_metrics,
                "model_count": len(completed),
                "primary_score": w_metrics.get("accuracy") or w_metrics.get("r2") or 0
            }

            # Combined feature importances
            fi_combined = {}
            for m in completed:
                for feat, imp in m.get("feature_importances", {}).items():
                    fi_combined[feat] = fi_combined.get(feat, 0) + imp
            n = max(len(completed), 1)
            fi_combined = {k: round(v / n, 4) for k, v in fi_combined.items()}

            conn5 = get_db()
            conn5.execute("""
                INSERT INTO ml_ensemble_results (evaluation_id, combined_metrics_json, combined_feature_importances_json)
                VALUES (?,?,?)
            """, (eval_id, json.dumps(combined), json.dumps(fi_combined)))

            # Cache best version
            best = max(completed, key=lambda m: m.get("cv_score", 0))
            ev_row = conn5.execute("SELECT project_id, version_id FROM ml_evaluations WHERE id=?", (eval_id,)).fetchone()
            if ev_row:
                ensemble_score = combined.get("weighted_avg", {}).get("accuracy") or \
                                  combined.get("weighted_avg", {}).get("r2") or 0
                conn5.execute("""
                    INSERT OR REPLACE INTO version_best_cache
                    (project_id, version_id, evaluation_id, best_model, best_score, ensemble_score, task_type, target_column)
                    VALUES (?,?,?,?,?,?,?,?)
                """, (ev_row["project_id"], ev_row["version_id"], eval_id,
                      best["name"], best.get("cv_score", 0), ensemble_score,
                      task_type, ev["target_column"]))
            conn5.commit()
            conn5.close()

        update_status("completed", 100, "Done")

    except Exception as e:
        update_status("failed", error=str(e))
        raise
