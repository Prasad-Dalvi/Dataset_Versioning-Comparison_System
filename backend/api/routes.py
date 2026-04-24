import json
import asyncio
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import io

from core.database import get_db, log_activity
from core.auth import (get_current_user, register_user, login_user,
                        check_project_access)
from core.versioning import (commit_version, get_versions, get_version_preview,
                               get_version_profile, create_branch, get_branches,
                               merge_branches, get_lineage, delete_version, get_version_df)
from core.diff_engine import compute_diff, get_diff_result
from core.ml_engine import run_evaluation
from core.quality import compute_quality, get_quality
from core.predictor import train_predictor, run_prediction, run_batch_prediction
from core import copilot as copilot_mod

router = APIRouter()

# ─── Auth ───────────────────────────────────────────────────────────────────

class RegisterBody(BaseModel):
    username: str
    email: str
    password: str
    full_name: str = ""

class LoginBody(BaseModel):
    username: str
    password: str

@router.post("/auth/register")
def auth_register(body: RegisterBody):
    return register_user(body.username, body.email, body.password, body.full_name)

@router.post("/auth/login")
def auth_login(body: LoginBody):
    return login_user(body.username, body.password)

@router.get("/auth/me")
def auth_me(user=Depends(get_current_user)):
    return user

# ─── Projects ────────────────────────────────────────────────────────────────

class ProjectBody(BaseModel):
    name: str
    description: str = ""
    default_target_column: str = ""

@router.get("/projects")
def list_projects(user=Depends(get_current_user)):
    conn = get_db()
    rows = conn.execute("""
        SELECT p.*, u.username as owner_name,
               COUNT(DISTINCT dv.id) as version_count,
               COUNT(DISTINCT c.id) as collaborator_count
        FROM projects p
        LEFT JOIN users u ON p.owner_id = u.id
        LEFT JOIN dataset_versions dv ON dv.project_id = p.id
        LEFT JOIN collaborators c ON c.project_id = p.id
        WHERE p.owner_id=? OR EXISTS (
            SELECT 1 FROM collaborators cc WHERE cc.project_id=p.id AND cc.user_id=?
        )
        GROUP BY p.id
        ORDER BY p.updated_at DESC
    """, (user["id"], user["id"])).fetchall()
    conn.close()
    return [dict(r) for r in rows]

@router.post("/projects")
def create_project(body: ProjectBody, user=Depends(get_current_user)):
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        INSERT INTO projects (name, description, owner_id, default_target_column)
        VALUES (?,?,?,?)
    """, (body.name, body.description, user["id"], body.default_target_column))
    pid = c.lastrowid
    conn.commit()
    proj = conn.execute("SELECT * FROM projects WHERE id=?", (pid,)).fetchone()
    conn.close()
    log_activity(user["id"], pid, "create_project", body.name)
    return dict(proj)

@router.get("/projects/{project_id}")
def get_project(project_id: int, user=Depends(get_current_user)):
    proj = check_project_access(project_id, user["id"])
    conn = get_db()
    owner = conn.execute("SELECT username FROM users WHERE id=?", (proj["owner_id"],)).fetchone()
    conn.close()
    result = dict(proj)
    result["owner_name"] = owner["username"] if owner else ""
    return result

class ProjectUpdateBody(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    default_target_column: Optional[str] = None

@router.patch("/projects/{project_id}")
def update_project(project_id: int, body: ProjectUpdateBody, user=Depends(get_current_user)):
    check_project_access(project_id, user["id"], "editor")
    conn = get_db()
    allowed = ["name", "description", "default_target_column"]
    updates = {k: v for k, v in body.dict().items() if k in allowed and v is not None}
    if updates:
        sets = ", ".join(f"{k}=?" for k in updates)
        conn.execute(f"UPDATE projects SET {sets}, updated_at=datetime('now') WHERE id=?",
                     list(updates.values()) + [project_id])
        conn.commit()
    proj = conn.execute("SELECT * FROM projects WHERE id=?", (project_id,)).fetchone()
    conn.close()
    return dict(proj)

@router.delete("/projects/{project_id}")
def delete_project(project_id: int, user=Depends(get_current_user)):
    proj = check_project_access(project_id, user["id"])
    if proj["owner_id"] != user["id"] and user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Only owner can delete project")
    conn = get_db()
    conn.execute("DELETE FROM projects WHERE id=?", (project_id,))
    conn.commit()
    conn.close()
    return {"deleted": True}

@router.get("/projects/{project_id}/stats")
def project_stats(project_id: int, user=Depends(get_current_user)):
    check_project_access(project_id, user["id"])
    conn = get_db()
    versions = conn.execute("SELECT COUNT(*) as cnt FROM dataset_versions WHERE project_id=?", (project_id,)).fetchone()
    evals = conn.execute("SELECT COUNT(*) as cnt FROM ml_evaluations WHERE project_id=?", (project_id,)).fetchone()
    storage = conn.execute("SELECT SUM(file_size) as total FROM dataset_versions WHERE project_id=?", (project_id,)).fetchone()
    collabs = conn.execute("SELECT COUNT(*) as cnt FROM collaborators WHERE project_id=?", (project_id,)).fetchone()
    conn.close()
    return {
        "version_count": versions["cnt"],
        "evaluation_count": evals["cnt"],
        "storage_bytes": storage["total"] or 0,
        "collaborator_count": collabs["cnt"]
    }

@router.get("/projects/{project_id}/activity")
def project_activity(project_id: int, user=Depends(get_current_user)):
    check_project_access(project_id, user["id"])
    conn = get_db()
    rows = conn.execute("""
        SELECT al.*, u.username
        FROM activity_log al
        LEFT JOIN users u ON al.user_id = u.id
        WHERE al.project_id=?
        ORDER BY al.created_at DESC LIMIT 50
    """, (project_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

@router.get("/projects/{project_id}/radar-data")
def radar_data(project_id: int, target_column: str = "", user=Depends(get_current_user)):
    check_project_access(project_id, user["id"])
    conn = get_db()

    if not target_column:
        proj = conn.execute("SELECT default_target_column FROM projects WHERE id=?", (project_id,)).fetchone()
        if proj:
            target_column = proj["default_target_column"] or ""

    versions = conn.execute("""
        SELECT dv.id, dv.version_number, dv.version_label, dv.row_count, dv.col_count,
               dv.file_size, dv.created_at, qs.overall as quality, qs.grade,
               qs.completeness, qs.uniqueness, qs.consistency, qs.validity
        FROM dataset_versions dv
        LEFT JOIN quality_scores qs ON dv.id = qs.version_id
        WHERE dv.project_id=?
        ORDER BY dv.version_number
    """, (project_id,)).fetchall()
    versions = [dict(v) for v in versions]

    version_ids = [v["id"] for v in versions]
    performance_data = []
    champion = None
    best_score = -1

    for v in versions:
        v_data = {"version": v["version_number"], "label": v["version_label"],
                  "quality": round(v["quality"] or 0, 1)}
        cache = conn.execute("""
            SELECT vbc.*, me.task_type
            FROM version_best_cache vbc
            JOIN ml_evaluations me ON vbc.evaluation_id = me.id
            WHERE vbc.version_id=? AND (vbc.target_column=? OR ?='')
            ORDER BY vbc.best_score DESC LIMIT 1
        """, (v["id"], target_column, target_column)).fetchone()

        if cache:
            c = dict(cache)
            v_data["best_score"] = round(c["best_score"] or 0, 4)
            v_data["ensemble_score"] = round(c["ensemble_score"] or 0, 4)
            v_data["best_model"] = c["best_model"]
            v_data["task_type"] = c["task_type"]

            # Get individual model scores
            model_results = conn.execute("""
                SELECT mr.model_name, mr.metrics_json
                FROM ml_model_results mr
                WHERE mr.evaluation_id=? AND mr.status='completed'
            """, (c["evaluation_id"],)).fetchall()
            for mr in model_results:
                m = json.loads(mr["metrics_json"] or "{}")
                v_data[mr["model_name"]] = round(m.get("cv_score") or m.get("accuracy") or 0, 4)

            if (c["best_score"] or 0) > best_score:
                best_score = c["best_score"] or 0
                champion = {
                    "version": v["version_number"],
                    "label": v["version_label"],
                    "best_model": c["best_model"],
                    "best_score": round(c["best_score"] or 0, 4),
                    "ensemble_score": round(c["ensemble_score"] or 0, 4),
                    "quality_grade": v["grade"] or "N/A",
                    "quality_score": round(v["quality"] or 0, 1),
                    "task_type": c["task_type"]
                }
        performance_data.append(v_data)

    # Model win counter
    model_wins = {}
    all_caches = conn.execute("""
        SELECT best_model, COUNT(*) as wins
        FROM version_best_cache WHERE project_id=? AND best_model IS NOT NULL
        GROUP BY best_model ORDER BY wins DESC
    """, (project_id,)).fetchall()
    for r in all_caches:
        model_wins[r["best_model"]] = r["wins"]

    # Quality heatmap data
    quality_heatmap = []
    for v in versions:
        quality_heatmap.append({
            "version": v["version_number"],
            "label": v["version_label"],
            "completeness": round(v["completeness"] or 0, 1),
            "consistency": round(v["consistency"] or 0, 1),
            "uniqueness": round(v["uniqueness"] or 0, 1),
            "validity": round(v["validity"] or 0, 1),
            "overall": round(v["quality"] or 0, 1),
            "grade": v["grade"] or "N/A"
        })

    # Radar chart top 5 versions
    radar_versions = sorted(performance_data, key=lambda x: x.get("best_score", 0), reverse=True)[:5]

    conn.close()

    return {
        "champion": champion,
        "performance_timeline": performance_data,
        "model_wins": [{"model": k, "wins": v} for k, v in model_wins.items()],
        "quality_heatmap": quality_heatmap,
        "radar_versions": radar_versions,
        "has_evaluations": any(v.get("best_score") is not None for v in performance_data)
    }

# ─── Collaborators ───────────────────────────────────────────────────────────

@router.get("/projects/{project_id}/collaborators")
def list_collaborators(project_id: int, user=Depends(get_current_user)):
    check_project_access(project_id, user["id"])
    conn = get_db()
    rows = conn.execute("""
        SELECT c.*, u.username, u.email, u.full_name
        FROM collaborators c
        JOIN users u ON c.user_id = u.id
        WHERE c.project_id=?
    """, (project_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

class InviteBody(BaseModel):
    project_id: int
    username: str
    role: str = "viewer"

@router.post("/collaborators/invite")
def invite_collaborator(body: InviteBody, user=Depends(get_current_user)):
    check_project_access(body.project_id, user["id"])
    conn = get_db()
    target = conn.execute("SELECT id FROM users WHERE username=?", (body.username,)).fetchone()
    if not target:
        conn.close()
        raise HTTPException(status_code=404, detail="User not found")
    try:
        conn.execute("""
            INSERT OR REPLACE INTO collaborators (project_id, user_id, role)
            VALUES (?,?,?)
        """, (body.project_id, target["id"], body.role))
        conn.commit()
    except Exception as e:
        conn.close()
        raise HTTPException(status_code=400, detail=str(e))
    conn.close()
    log_activity(user["id"], body.project_id, "invite", body.username)
    return {"status": "invited"}

class RoleBody(BaseModel):
    project_id: int
    user_id: int
    role: str

@router.put("/collaborators/role")
def update_collab_role(body: RoleBody, user=Depends(get_current_user)):
    check_project_access(body.project_id, user["id"])
    conn = get_db()
    conn.execute("UPDATE collaborators SET role=? WHERE project_id=? AND user_id=?",
                 (body.role, body.project_id, body.user_id))
    conn.commit()
    conn.close()
    return {"status": "updated"}

@router.delete("/collaborators/{project_id}/{collab_user_id}")
def remove_collaborator(project_id: int, collab_user_id: int, user=Depends(get_current_user)):
    check_project_access(project_id, user["id"])
    conn = get_db()
    conn.execute("DELETE FROM collaborators WHERE project_id=? AND user_id=?",
                 (project_id, collab_user_id))
    conn.commit()
    conn.close()
    return {"status": "removed"}

# ─── Admin ───────────────────────────────────────────────────────────────────

def require_admin(user=Depends(get_current_user)):
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin only")
    return user

@router.get("/admin/stats")
def admin_stats(user=Depends(require_admin)):
    conn = get_db()
    stats = {
        "users": conn.execute("SELECT COUNT(*) as c FROM users").fetchone()["c"],
        "projects": conn.execute("SELECT COUNT(*) as c FROM projects").fetchone()["c"],
        "versions": conn.execute("SELECT COUNT(*) as c FROM dataset_versions").fetchone()["c"],
        "evaluations": conn.execute("SELECT COUNT(*) as c FROM ml_evaluations").fetchone()["c"],
        "storage_mb": round((conn.execute("SELECT SUM(file_size) as s FROM dataset_versions").fetchone()["s"] or 0) / 1024 / 1024, 2)
    }
    conn.close()
    return stats

@router.get("/admin/users")
def admin_users(user=Depends(require_admin)):
    conn = get_db()
    rows = conn.execute("""
        SELECT u.*, COUNT(DISTINCT p.id) as project_count
        FROM users u
        LEFT JOIN projects p ON p.owner_id = u.id
        GROUP BY u.id ORDER BY u.created_at DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]

class AdminUserPatch(BaseModel):
    role: str

@router.patch("/admin/users/{uid}")
def admin_update_user(uid: int, body: AdminUserPatch, user=Depends(require_admin)):
    conn = get_db()
    conn.execute("UPDATE users SET role=? WHERE id=?", (body.role, uid))
    conn.commit()
    conn.close()
    return {"status": "updated"}

@router.get("/admin/projects")
def admin_projects(user=Depends(require_admin)):
    conn = get_db()
    rows = conn.execute("""
        SELECT p.*, u.username as owner_name, COUNT(dv.id) as version_count
        FROM projects p
        LEFT JOIN users u ON p.owner_id = u.id
        LEFT JOIN dataset_versions dv ON dv.project_id = p.id
        GROUP BY p.id ORDER BY p.created_at DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]

@router.get("/admin/activity")
def admin_activity(user=Depends(require_admin)):
    conn = get_db()
    rows = conn.execute("""
        SELECT al.*, u.username, p.name as project_name
        FROM activity_log al
        LEFT JOIN users u ON al.user_id = u.id
        LEFT JOIN projects p ON al.project_id = p.id
        ORDER BY al.created_at DESC LIMIT 100
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]

# ─── Versions ────────────────────────────────────────────────────────────────

@router.get("/projects/{project_id}/versions")
def list_versions(project_id: int, user=Depends(get_current_user)):
    check_project_access(project_id, user["id"])
    return get_versions(project_id)

@router.post("/versions/commit")
async def version_commit(
    background_tasks: BackgroundTasks,
    project_id: int = Form(...),
    commit_message: str = Form(""),
    branch_name: str = Form("main"),
    parent_id: Optional[int] = Form(None),
    file: UploadFile = File(...),
    user=Depends(get_current_user)
):
    check_project_access(project_id, user["id"], "editor")
    content = await file.read()
    result = commit_version(project_id, content, file.filename,
                             commit_message, user["id"], branch_name, parent_id)

    # Auto-evaluate if default_target_column set
    conn = get_db()
    proj = conn.execute("SELECT default_target_column FROM projects WHERE id=?", (project_id,)).fetchone()
    conn.close()
    if proj and proj["default_target_column"]:
        target = proj["default_target_column"]
        conn2 = get_db()
        c = conn2.cursor()
        c.execute("""
            INSERT INTO ml_evaluations (project_id, version_id, target_column, status)
            VALUES (?,?,?,?)
        """, (project_id, result["version_id"], target, "pending"))
        eval_id = c.lastrowid
        conn2.commit()
        conn2.close()
        background_tasks.add_task(_run_eval_bg, eval_id)
        result["auto_eval_id"] = eval_id

    return result

async def _run_eval_bg(eval_id: int):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, run_evaluation, eval_id)

@router.get("/versions/{version_id}/preview")
def version_preview(version_id: int, rows: int = 100, user=Depends(get_current_user)):
    return get_version_preview(version_id, rows)

@router.get("/versions/{version_id}/quality")
def version_quality(version_id: int, user=Depends(get_current_user)):
    q = get_quality(version_id)
    if not q:
        conn = get_db()
        row = conn.execute("SELECT file_path FROM dataset_versions WHERE id=?", (version_id,)).fetchone()
        conn.close()
        if not row:
            raise HTTPException(status_code=404, detail="Version not found")
        import pyarrow.parquet as pq
        df = pq.read_table(row["file_path"]).to_pandas()
        q = compute_quality(df, version_id)
    return q

@router.get("/versions/{version_id}/profile")
def version_profile_route(version_id: int, user=Depends(get_current_user)):
    return get_version_profile(version_id)

@router.delete("/versions/{version_id}")
def version_delete(version_id: int, user=Depends(get_current_user)):
    ok = delete_version(version_id, user["id"])
    return {"deleted": ok}

# ─── Branches ────────────────────────────────────────────────────────────────

@router.get("/projects/{project_id}/branches")
def list_branches(project_id: int, user=Depends(get_current_user)):
    check_project_access(project_id, user["id"])
    return get_branches(project_id)

class BranchCreateBody(BaseModel):
    project_id: int
    branch_name: str
    base_version_id: Optional[int] = None

@router.post("/branches/create")
def branch_create(body: BranchCreateBody, user=Depends(get_current_user)):
    check_project_access(body.project_id, user["id"], "editor")
    return create_branch(body.project_id, body.branch_name, user["id"], body.base_version_id)

class MergeBody(BaseModel):
    project_id: int
    source_branch: str
    target_branch: str
    strategy: str = "latest"

@router.post("/branches/merge")
def branch_merge(body: MergeBody, user=Depends(get_current_user)):
    check_project_access(body.project_id, user["id"], "editor")
    return merge_branches(body.project_id, body.source_branch,
                           body.target_branch, body.strategy, user["id"])

@router.get("/projects/{project_id}/lineage")
def project_lineage(project_id: int, user=Depends(get_current_user)):
    check_project_access(project_id, user["id"])
    return get_lineage(project_id)

# ─── Diff ────────────────────────────────────────────────────────────────────

class DiffBody(BaseModel):
    project_id: int
    version_a_id: int
    version_b_id: int

@router.post("/diff")
def run_diff(body: DiffBody, user=Depends(get_current_user)):
    check_project_access(body.project_id, user["id"])
    return compute_diff(body.version_a_id, body.version_b_id, body.project_id)

@router.get("/diff/{diff_id}")
def get_diff(diff_id: int, user=Depends(get_current_user)):
    r = get_diff_result(diff_id)
    if not r:
        raise HTTPException(status_code=404, detail="Diff not found")
    return r

# ─── ML Evaluations ──────────────────────────────────────────────────────────

class EvalBody(BaseModel):
    project_id: int
    version_id: int
    target_column: str

@router.post("/evaluations")
async def start_evaluation(body: EvalBody, background_tasks: BackgroundTasks,
                            user=Depends(get_current_user)):
    check_project_access(body.project_id, user["id"])
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        INSERT INTO ml_evaluations (project_id, version_id, target_column, status)
        VALUES (?,?,?,?)
    """, (body.project_id, body.version_id, body.target_column, "pending"))
    eval_id = c.lastrowid
    conn.commit()
    conn.close()
    background_tasks.add_task(_run_eval_bg, eval_id)
    log_activity(user["id"], body.project_id, "start_eval", body.target_column)
    return {"eval_id": eval_id, "status": "pending"}

@router.get("/evaluations/{eval_id}")
def get_evaluation(eval_id: int, user=Depends(get_current_user)):
    conn = get_db()
    ev = conn.execute("SELECT * FROM ml_evaluations WHERE id=?", (eval_id,)).fetchone()
    if not ev:
        conn.close()
        raise HTTPException(status_code=404)
    ev = dict(ev)
    models = conn.execute("""
        SELECT model_name, metrics_json, feature_importances_json,
               confusion_matrix_json, training_time, status
        FROM ml_model_results WHERE evaluation_id=?
    """, (eval_id,)).fetchall()
    model_list = []
    for m in models:
        md = dict(m)
        md["metrics"] = json.loads(md.get("metrics_json") or "{}")
        md["feature_importances"] = json.loads(md.get("feature_importances_json") or "{}")
        md["confusion_matrix"] = json.loads(md.get("confusion_matrix_json") or "[]")
        model_list.append(md)
    ev["models"] = model_list
    ensemble = conn.execute(
        "SELECT * FROM ml_ensemble_results WHERE evaluation_id=? ORDER BY id DESC LIMIT 1",
        (eval_id,)
    ).fetchone()
    if ensemble:
        ens = dict(ensemble)
        ens["combined_metrics"] = json.loads(ens.get("combined_metrics_json") or "{}")
        ens["combined_feature_importances"] = json.loads(ens.get("combined_feature_importances_json") or "{}")
        ev["ensemble"] = ens
    conn.close()
    return ev

@router.get("/evaluations/{eval_id}/status")
def eval_status(eval_id: int, user=Depends(get_current_user)):
    conn = get_db()
    ev = conn.execute(
        "SELECT id,status,progress,current_model,error_message,task_type FROM ml_evaluations WHERE id=?",
        (eval_id,)
    ).fetchone()
    conn.close()
    if not ev:
        raise HTTPException(status_code=404)
    return dict(ev)

@router.get("/projects/{project_id}/evaluations")
def list_evaluations(project_id: int, user=Depends(get_current_user)):
    check_project_access(project_id, user["id"])
    conn = get_db()
    rows = conn.execute("""
        SELECT me.*, COUNT(mr.id) as model_count
        FROM ml_evaluations me
        LEFT JOIN ml_model_results mr ON mr.evaluation_id = me.id
        WHERE me.project_id=?
        GROUP BY me.id ORDER BY me.created_at DESC
    """, (project_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

# ─── Predictions ─────────────────────────────────────────────────────────────

class TrainBody(BaseModel):
    project_id: int
    version_id: int
    target_column: str

@router.post("/predict/train")
def predict_train(body: TrainBody, user=Depends(get_current_user)):
    check_project_access(body.project_id, user["id"])
    return train_predictor(body.project_id, body.version_id, body.target_column)

class PredictBody(BaseModel):
    project_id: int
    version_id: int
    target_column: str
    input_data: dict

@router.post("/predict/run")
def predict_run(body: PredictBody, user=Depends(get_current_user)):
    check_project_access(body.project_id, user["id"])
    return run_prediction(body.project_id, body.version_id, body.target_column, body.input_data)

@router.post("/predict/batch")
async def predict_batch(
    project_id: int = Form(...),
    version_id: int = Form(...),
    target_column: str = Form(...),
    file: UploadFile = File(...),
    user=Depends(get_current_user)
):
    check_project_access(project_id, user["id"])
    content = await file.read()
    result_bytes = run_batch_prediction(project_id, version_id, target_column, content)
    if result_bytes is None:
        raise HTTPException(status_code=400, detail="Prediction failed")
    return StreamingResponse(
        io.BytesIO(result_bytes),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=predictions.csv"}
    )

# ─── Copilot ─────────────────────────────────────────────────────────────────

class ChatBody(BaseModel):
    project_id: int
    message: str
    provider: str = "anthropic"
    api_key: str = ""

@router.post("/copilot/chat")
async def copilot_chat(body: ChatBody, user=Depends(get_current_user)):
    check_project_access(body.project_id, user["id"])
    history = copilot_mod.get_history(body.project_id)
    response = await copilot_mod.chat(
        body.project_id, user["id"], body.message,
        body.provider, body.api_key, history
    )
    return {"response": response}

@router.get("/copilot/history/{project_id}")
def copilot_history(project_id: int, user=Depends(get_current_user)):
    check_project_access(project_id, user["id"])
    return copilot_mod.get_history(project_id)

@router.get("/copilot/context/{project_id}")
def copilot_context(project_id: int, user=Depends(get_current_user)):
    """Returns what data the AI copilot actually sees — transparency endpoint."""
    check_project_access(project_id, user["id"])
    return copilot_mod.get_context_preview(project_id)

@router.delete("/copilot/history/{project_id}")
def copilot_clear_history(project_id: int, user=Depends(get_current_user)):
    """Clear chat history for a project."""
    check_project_access(project_id, user["id"])
    conn = get_db()
    conn.execute("DELETE FROM copilot_history WHERE project_id=?", (project_id,))
    conn.commit()
    conn.close()
    return {"cleared": True}

@router.get("/copilot/suggested-questions/{project_id}")
def copilot_suggested_questions(project_id: int, user=Depends(get_current_user)):
    """Return suggested questions based on project state."""
    check_project_access(project_id, user["id"])
    conn = get_db()
    has_eval = conn.execute(
        "SELECT id FROM ml_evaluations WHERE project_id=? AND status='completed' LIMIT 1",
        (project_id,)
    ).fetchone()
    has_diff = conn.execute(
        "SELECT id FROM diff_results WHERE project_id=? LIMIT 1",
        (project_id,)
    ).fetchone()
    version_count = conn.execute(
        "SELECT COUNT(*) as c FROM dataset_versions WHERE project_id=?",
        (project_id,)
    ).fetchone()["c"]
    conn.close()

    questions = [
        "Summarize the overall health of this project.",
        "What is the data quality score and what factors are dragging it down?",
    ]
    if version_count > 1:
        questions.append("How has my dataset changed across versions?")
    if has_eval:
        questions.append("Which model performed best and why?")
        questions.append("What are the most important features for my target variable?")
        questions.append("Is the ensemble significantly better than the best single model?")
        questions.append("Give me 3 actionable recommendations to improve model accuracy.")
    if has_diff:
        questions.append("Is there any data drift I should be worried about?")
        questions.append("Which columns drifted the most between versions?")
    questions.append("What steps should I take to improve data completeness?")
    questions.append("Are there any anomalies or outliers worth investigating?")

    return {"questions": questions}
