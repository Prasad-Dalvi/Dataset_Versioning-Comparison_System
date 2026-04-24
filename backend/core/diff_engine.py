import json
import numpy as np
import pandas as pd
from scipy import stats
from .database import get_db
from .versioning import get_version_df

def compute_diff(version_a_id: int, version_b_id: int, project_id: int) -> dict:
    conn = get_db()
    cached = conn.execute(
        "SELECT * FROM diff_results WHERE version_a_id=? AND version_b_id=?",
        (version_a_id, version_b_id)
    ).fetchone()
    conn.close()
    if cached:
        r = dict(cached)
        r["result"] = json.loads(r.get("result_json") or "{}")
        return r

    df_a = get_version_df(version_a_id)
    df_b = get_version_df(version_b_id)
    if df_a is None or df_b is None:
        return {"error": "Version not found"}

    result = {}

    # Schema diff
    cols_a = set(df_a.columns)
    cols_b = set(df_b.columns)
    added = list(cols_b - cols_a)
    removed = list(cols_a - cols_b)
    type_changed = {}
    for col in cols_a & cols_b:
        if str(df_a[col].dtype) != str(df_b[col].dtype):
            type_changed[col] = {"from": str(df_a[col].dtype), "to": str(df_b[col].dtype)}
    result["schema_diff"] = {
        "added_columns": added,
        "removed_columns": removed,
        "type_changed": type_changed
    }

    # Row diff
    row_diff_pct = abs(len(df_b) - len(df_a)) / max(len(df_a), 1) * 100
    dups_a = int(df_a.duplicated().sum())
    dups_b = int(df_b.duplicated().sum())
    result["row_diff"] = {
        "rows_a": len(df_a),
        "rows_b": len(df_b),
        "row_diff": len(df_b) - len(df_a),
        "row_diff_pct": round(row_diff_pct, 2),
        "duplicates_a": dups_a,
        "duplicates_b": dups_b
    }

    # Statistical drift
    drift = {}
    common_cols = list(cols_a & cols_b)
    for col in common_cols:
        col_drift = {}
        if pd.api.types.is_numeric_dtype(df_a[col]) and pd.api.types.is_numeric_dtype(df_b[col]):
            a_vals = df_a[col].dropna().values
            b_vals = df_b[col].dropna().values
            if len(a_vals) > 1 and len(b_vals) > 1:
                ks_stat, ks_pval = stats.ks_2samp(a_vals, b_vals)
                col_drift["ks_statistic"] = round(float(ks_stat), 4)
                col_drift["ks_pvalue"] = round(float(ks_pval), 4)
                col_drift["mean_a"] = round(float(np.mean(a_vals)), 4)
                col_drift["mean_b"] = round(float(np.mean(b_vals)), 4)
                col_drift["std_a"] = round(float(np.std(a_vals)), 4)
                col_drift["std_b"] = round(float(np.std(b_vals)), 4)
                # PSI
                psi = compute_psi(a_vals, b_vals)
                col_drift["psi"] = round(float(psi), 4)
                if ks_stat > 0.3 or psi > 0.2:
                    col_drift["drift_level"] = "HIGH"
                elif ks_stat > 0.15 or psi > 0.1:
                    col_drift["drift_level"] = "MEDIUM"
                elif ks_stat > 0.05 or psi > 0.05:
                    col_drift["drift_level"] = "LOW"
                else:
                    col_drift["drift_level"] = "NONE"
        else:
            vc_a = df_a[col].value_counts(normalize=True)
            vc_b = df_b[col].value_counts(normalize=True)
            all_cats = set(vc_a.index) | set(vc_b.index)
            obs_a = np.array([vc_a.get(c, 0) for c in all_cats]) + 1e-6
            obs_b = np.array([vc_b.get(c, 0) for c in all_cats]) + 1e-6
            if len(obs_a) > 1:
                chi2, pval = stats.chisquare(obs_b / obs_b.sum(), obs_a / obs_a.sum())
                col_drift["chi2_stat"] = round(float(chi2), 4)
                col_drift["chi2_pvalue"] = round(float(pval), 4)
                if chi2 > 20:
                    col_drift["drift_level"] = "HIGH"
                elif chi2 > 10:
                    col_drift["drift_level"] = "MEDIUM"
                elif chi2 > 5:
                    col_drift["drift_level"] = "LOW"
                else:
                    col_drift["drift_level"] = "NONE"
        if col_drift:
            drift[col] = col_drift
    result["drift"] = drift

    # Overall severity
    high_count = sum(1 for c in drift.values() if c.get("drift_level") == "HIGH")
    med_count = sum(1 for c in drift.values() if c.get("drift_level") == "MEDIUM")
    schema_issues = len(added) + len(removed) + len(type_changed)

    if high_count >= 3 or schema_issues >= 3:
        severity = "CRITICAL"
    elif high_count >= 1 or schema_issues >= 1:
        severity = "HIGH"
    elif med_count >= 3:
        severity = "MEDIUM"
    elif med_count >= 1:
        severity = "LOW"
    else:
        severity = "NONE"

    result["severity"] = severity
    result["summary"] = {
        "high_drift_cols": high_count,
        "medium_drift_cols": med_count,
        "schema_issues": schema_issues
    }

    conn = get_db()
    c = conn.cursor()
    c.execute("""
        INSERT INTO diff_results (project_id, version_a_id, version_b_id, result_json, severity)
        VALUES (?,?,?,?,?)
    """, (project_id, version_a_id, version_b_id, json.dumps(result), severity))
    diff_id = c.lastrowid
    conn.commit()
    conn.close()

    return {"id": diff_id, "result": result, "severity": severity}

def compute_psi(expected, actual, buckets=10):
    try:
        breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
        breakpoints = np.unique(breakpoints)
        if len(breakpoints) < 2:
            return 0.0
        exp_counts = np.histogram(expected, bins=breakpoints)[0]
        act_counts = np.histogram(actual, bins=breakpoints)[0]
        exp_pct = exp_counts / max(exp_counts.sum(), 1) + 1e-6
        act_pct = act_counts / max(act_counts.sum(), 1) + 1e-6
        psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
        return float(psi)
    except:
        return 0.0

def get_diff_result(diff_id: int) -> dict:
    conn = get_db()
    row = conn.execute("SELECT * FROM diff_results WHERE id=?", (diff_id,)).fetchone()
    conn.close()
    if not row:
        return None
    r = dict(row)
    r["result"] = json.loads(r.get("result_json") or "{}")
    return r
