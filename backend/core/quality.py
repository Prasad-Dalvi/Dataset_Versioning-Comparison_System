import pandas as pd
import numpy as np
import json
from .database import get_db

def compute_quality(df: pd.DataFrame, version_id: int) -> dict:
    total_cells = df.shape[0] * df.shape[1]
    null_cells = df.isnull().sum().sum()
    completeness = float(1 - (null_cells / total_cells)) if total_cells > 0 else 1.0

    dup_rows = df.duplicated().sum()
    uniqueness = float(1 - (dup_rows / len(df))) if len(df) > 0 else 1.0

    outlier_penalties = []
    for col in df.select_dtypes(include=[np.number]).columns:
        series = df[col].dropna()
        if len(series) < 4:
            continue
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        if IQR == 0:
            continue
        outliers = ((series < Q1 - 1.5 * IQR) | (series > Q3 + 1.5 * IQR)).sum()
        pct = outliers / len(series)
        if pct > 0.3:
            outlier_penalties.append(pct - 0.3)
    consistency = float(1.0 - min(sum(outlier_penalties) / max(len(df.select_dtypes(include=[np.number]).columns), 1), 0.5))

    validity_issues = 0
    total_checks = 0
    for col in df.select_dtypes(include=[np.number]).columns:
        series = df[col].dropna()
        if len(series) == 0:
            continue
        total_checks += 1
        col_lower = col.lower()
        if any(k in col_lower for k in ["age", "year"]):
            if (series < 0).any() or (series > 200).any():
                validity_issues += 1
        elif any(k in col_lower for k in ["pct", "percent", "rate", "ratio", "prob"]):
            if (series < 0).any() or (series > 100).any():
                validity_issues += 1
    validity = float(1.0 - (validity_issues / total_checks)) if total_checks > 0 else 1.0

    overall = (completeness * 0.35 + uniqueness * 0.25 + consistency * 0.25 + validity * 0.15) * 100

    if overall >= 90:
        grade = "A"
    elif overall >= 75:
        grade = "B"
    elif overall >= 60:
        grade = "C"
    elif overall >= 45:
        grade = "D"
    else:
        grade = "F"

    details = {
        "completeness": round(completeness * 100, 2),
        "uniqueness": round(uniqueness * 100, 2),
        "consistency": round(consistency * 100, 2),
        "validity": round(validity * 100, 2),
        "null_count": int(null_cells),
        "duplicate_rows": int(dup_rows),
        "total_rows": len(df),
        "total_cols": len(df.columns)
    }

    conn = get_db()
    conn.execute("""
        INSERT OR REPLACE INTO quality_scores
        (version_id, completeness, uniqueness, consistency, validity, overall, grade, details_json)
        VALUES (?,?,?,?,?,?,?,?)
    """, (version_id, round(completeness*100,2), round(uniqueness*100,2),
          round(consistency*100,2), round(validity*100,2),
          round(overall,2), grade, json.dumps(details)))
    conn.commit()
    conn.close()

    return {
        "completeness": round(completeness*100, 2),
        "uniqueness": round(uniqueness*100, 2),
        "consistency": round(consistency*100, 2),
        "validity": round(validity*100, 2),
        "overall": round(overall, 2),
        "grade": grade,
        "details": details
    }

def get_quality(version_id: int) -> dict:
    conn = get_db()
    row = conn.execute("SELECT * FROM quality_scores WHERE version_id=?", (version_id,)).fetchone()
    conn.close()
    if not row:
        return None
    r = dict(row)
    r["details"] = json.loads(r.get("details_json") or "{}")
    return r
