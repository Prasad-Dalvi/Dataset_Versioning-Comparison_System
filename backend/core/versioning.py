import os
import json
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime
from .database import get_db, STORAGE_PATH, log_activity
from .quality import compute_quality

def get_or_create_branch(project_id: int, branch_name: str, user_id: int, base_version_id: int = None) -> int:
    conn = get_db()
    row = conn.execute("SELECT id FROM branches WHERE project_id=? AND name=?",
                       (project_id, branch_name)).fetchone()
    if row:
        conn.close()
        return row["id"]
    c = conn.cursor()
    c.execute("INSERT INTO branches (project_id,name,created_by,base_version_id) VALUES (?,?,?,?)",
              (project_id, branch_name, user_id, base_version_id))
    bid = c.lastrowid
    conn.commit()
    conn.close()
    return bid

def commit_version(project_id: int, file_bytes: bytes, original_filename: str,
                   commit_message: str, author_id: int, branch_name: str = "main",
                   parent_id: int = None) -> dict:
    import io
    df = pd.read_csv(io.BytesIO(file_bytes))

    conn = get_db()
    last = conn.execute(
        "SELECT MAX(version_number) as mx FROM dataset_versions WHERE project_id=?",
        (project_id,)
    ).fetchone()
    version_number = (last["mx"] or 0) + 1

    branch_id = get_or_create_branch(project_id, branch_name, author_id, parent_id)

    if parent_id is None:
        parent_row = conn.execute(
            "SELECT id FROM dataset_versions WHERE project_id=? AND branch_id=? ORDER BY version_number DESC LIMIT 1",
            (project_id, branch_id)
        ).fetchone()
        if parent_row:
            parent_id = parent_row["id"]

    schema = {}
    for col in df.columns:
        schema[col] = str(df[col].dtype)

    stats = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        s = df[col].dropna()
        stats[col] = {
            "mean": float(s.mean()) if len(s) else None,
            "std": float(s.std()) if len(s) else None,
            "min": float(s.min()) if len(s) else None,
            "max": float(s.max()) if len(s) else None,
            "median": float(s.median()) if len(s) else None,
            "null_count": int(df[col].isnull().sum())
        }
    for col in df.select_dtypes(include=["object", "category"]).columns:
        vc = df[col].value_counts()
        stats[col] = {
            "unique_count": int(df[col].nunique()),
            "top_values": vc.head(5).to_dict(),
            "null_count": int(df[col].isnull().sum())
        }

    proj_dir = os.path.join(STORAGE_PATH, f"project_{project_id}")
    os.makedirs(proj_dir, exist_ok=True)
    file_path = os.path.join(proj_dir, f"v{version_number}_{branch_name}.parquet")
    table = pa.Table.from_pandas(df)
    pq.write_table(table, file_path)
    file_size = os.path.getsize(file_path)

    c = conn.cursor()
    c.execute("""
        INSERT INTO dataset_versions
        (project_id,version_number,version_label,commit_message,author_id,branch_id,parent_id,
         file_path,original_filename,file_size,row_count,col_count,schema_json,stats_json)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (project_id, version_number, f"V{version_number}", commit_message, author_id,
          branch_id, parent_id, file_path, original_filename, file_size,
          len(df), len(df.columns), json.dumps(schema), json.dumps(stats)))
    version_id = c.lastrowid
    conn.execute("UPDATE projects SET updated_at=datetime('now') WHERE id=?", (project_id,))
    conn.commit()
    conn.close()

    quality = compute_quality(df, version_id)
    log_activity(author_id, project_id, "commit", f"V{version_number} - {commit_message}")

    return {
        "version_id": version_id,
        "version_number": version_number,
        "version_label": f"V{version_number}",
        "branch_name": branch_name,
        "row_count": len(df),
        "col_count": len(df.columns),
        "quality": quality
    }

def get_versions(project_id: int) -> list:
    conn = get_db()
    rows = conn.execute("""
        SELECT dv.*, u.username as author_name, b.name as branch_name,
               qs.overall as quality_overall, qs.grade as quality_grade
        FROM dataset_versions dv
        LEFT JOIN users u ON dv.author_id = u.id
        LEFT JOIN branches b ON dv.branch_id = b.id
        LEFT JOIN quality_scores qs ON dv.id = qs.version_id
        WHERE dv.project_id=?
        ORDER BY dv.version_number ASC
    """, (project_id,)).fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        d["schema"] = json.loads(d.get("schema_json") or "{}")
        d["stats"] = json.loads(d.get("stats_json") or "{}")
        result.append(d)
    return result

def get_version_df(version_id: int) -> pd.DataFrame:
    conn = get_db()
    row = conn.execute("SELECT file_path FROM dataset_versions WHERE id=?", (version_id,)).fetchone()
    conn.close()
    if not row:
        return None
    return pq.read_table(row["file_path"]).to_pandas()

def get_version_preview(version_id: int, rows: int = 100) -> dict:
    conn = get_db()
    row = conn.execute("SELECT * FROM dataset_versions WHERE id=?", (version_id,)).fetchone()
    conn.close()
    if not row:
        return None
    r = dict(row)
    df = pq.read_table(r["file_path"]).to_pandas()
    preview = df.head(rows)
    return {
        "columns": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "rows": preview.fillna("").values.tolist(),
        "total_rows": len(df),
        "total_cols": len(df.columns)
    }

def get_version_profile(version_id: int) -> dict:
    conn = get_db()
    row = conn.execute("SELECT * FROM dataset_versions WHERE id=?", (version_id,)).fetchone()
    conn.close()
    if not row:
        return None
    r = dict(row)
    df = pq.read_table(r["file_path"]).to_pandas()
    profile = {}
    for col in df.columns:
        col_data = df[col]
        info = {
            "dtype": str(col_data.dtype),
            "null_count": int(col_data.isnull().sum()),
            "null_pct": round(col_data.isnull().mean() * 100, 2),
            "unique_count": int(col_data.nunique()),
        }
        if pd.api.types.is_numeric_dtype(col_data):
            s = col_data.dropna()
            info.update({
                "mean": float(s.mean()) if len(s) else None,
                "std": float(s.std()) if len(s) else None,
                "min": float(s.min()) if len(s) else None,
                "max": float(s.max()) if len(s) else None,
                "median": float(s.median()) if len(s) else None,
                "q25": float(s.quantile(0.25)) if len(s) else None,
                "q75": float(s.quantile(0.75)) if len(s) else None,
            })
            hist, edges = np.histogram(s.dropna(), bins=20)
            info["histogram"] = {"counts": hist.tolist(), "edges": edges.tolist()}
        else:
            vc = col_data.value_counts().head(10)
            info["top_values"] = vc.to_dict()
        profile[col] = info
    return profile

def create_branch(project_id: int, branch_name: str, user_id: int, base_version_id: int = None) -> dict:
    conn = get_db()
    existing = conn.execute("SELECT id FROM branches WHERE project_id=? AND name=?",
                             (project_id, branch_name)).fetchone()
    if existing:
        conn.close()
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Branch already exists")
    c = conn.cursor()
    c.execute("INSERT INTO branches (project_id,name,created_by,base_version_id) VALUES (?,?,?,?)",
              (project_id, branch_name, user_id, base_version_id))
    bid = c.lastrowid
    conn.commit()
    conn.close()
    log_activity(user_id, project_id, "branch_create", branch_name)
    return {"branch_id": bid, "name": branch_name}

def get_branches(project_id: int) -> list:
    conn = get_db()
    rows = conn.execute("""
        SELECT b.*, u.username as creator_name,
               COUNT(dv.id) as version_count
        FROM branches b
        LEFT JOIN users u ON b.created_by = u.id
        LEFT JOIN dataset_versions dv ON dv.branch_id = b.id
        WHERE b.project_id=?
        GROUP BY b.id
    """, (project_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def merge_branches(project_id: int, source_branch: str, target_branch: str,
                   strategy: str, user_id: int) -> dict:
    conn = get_db()
    src = conn.execute("SELECT id FROM branches WHERE project_id=? AND name=?",
                       (project_id, source_branch)).fetchone()
    tgt = conn.execute("SELECT id FROM branches WHERE project_id=? AND name=?",
                       (project_id, target_branch)).fetchone()
    if not src or not tgt:
        conn.close()
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Branch not found")

    versions_src = conn.execute(
        "SELECT * FROM dataset_versions WHERE project_id=? AND branch_id=? ORDER BY version_number DESC",
        (project_id, src["id"])
    ).fetchall()
    versions_tgt = conn.execute(
        "SELECT * FROM dataset_versions WHERE project_id=? AND branch_id=? ORDER BY version_number DESC",
        (project_id, tgt["id"])
    ).fetchall()
    conn.close()

    if not versions_src:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Source branch has no versions")

    if strategy == "latest":
        chosen = versions_src[0]
    elif strategy == "best_quality":
        best = None
        best_score = -1
        conn2 = get_db()
        for v in versions_src:
            qs = conn2.execute("SELECT overall FROM quality_scores WHERE version_id=?", (v["id"],)).fetchone()
            score = qs["overall"] if qs else 0
            if score > best_score:
                best_score = score
                best = v
        conn2.close()
        chosen = best or versions_src[0]
    elif strategy == "largest":
        chosen = max(versions_src, key=lambda x: x["row_count"])
    else:
        chosen = versions_src[0]

    df = get_version_df(chosen["id"])
    import io
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    result = commit_version(
        project_id, buf.getvalue(),
        f"merge_{source_branch}_to_{target_branch}.csv",
        f"Merge {source_branch} into {target_branch} ({strategy})",
        user_id, target_branch, chosen["id"]
    )
    log_activity(user_id, project_id, "merge", f"{source_branch} -> {target_branch}")
    return result

def get_lineage(project_id: int) -> list:
    conn = get_db()
    rows = conn.execute("""
        SELECT dv.id, dv.version_number, dv.version_label, dv.parent_id,
               dv.commit_message, b.name as branch_name, u.username as author_name,
               dv.created_at, qs.grade as quality_grade
        FROM dataset_versions dv
        LEFT JOIN branches b ON dv.branch_id = b.id
        LEFT JOIN users u ON dv.author_id = u.id
        LEFT JOIN quality_scores qs ON dv.id = qs.version_id
        WHERE dv.project_id=?
        ORDER BY dv.version_number
    """, (project_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def delete_version(version_id: int, user_id: int) -> bool:
    conn = get_db()
    row = conn.execute("SELECT * FROM dataset_versions WHERE id=?", (version_id,)).fetchone()
    if not row:
        conn.close()
        return False
    r = dict(row)
    try:
        if os.path.exists(r["file_path"]):
            os.remove(r["file_path"])
    except:
        pass
    conn.execute("DELETE FROM dataset_versions WHERE id=?", (version_id,))
    conn.execute("DELETE FROM quality_scores WHERE version_id=?", (version_id,))
    conn.commit()
    conn.close()
    log_activity(user_id, r["project_id"], "delete_version", f"V{r['version_number']}")
    return True
