import hashlib
import jwt
import os
from datetime import datetime, timedelta
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from .database import get_db

SECRET_KEY = os.getenv("JWT_SECRET", "datavault-secret-key-2024")
ALGORITHM = "HS256"
security = HTTPBearer()

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def create_token(user_id: int, username: str, role: str) -> str:
    payload = {
        "user_id": user_id,
        "username": username,
        "role": role,
        "exp": datetime.utcnow() + timedelta(hours=48)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except:
        raise HTTPException(status_code=401, detail="Invalid token")

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    payload = decode_token(credentials.credentials)
    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE id=?", (payload["user_id"],)).fetchone()
    conn.close()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return dict(user)

def check_project_access(project_id: int, user_id: int, required_role: str = "viewer"):
    conn = get_db()
    project = conn.execute("SELECT * FROM projects WHERE id=?", (project_id,)).fetchone()
    if not project:
        conn.close()
        raise HTTPException(status_code=404, detail="Project not found")
    project = dict(project)
    if project["owner_id"] == user_id:
        conn.close()
        return project
    collab = conn.execute(
        "SELECT * FROM collaborators WHERE project_id=? AND user_id=?",
        (project_id, user_id)
    ).fetchone()
    conn.close()
    if not collab:
        raise HTTPException(status_code=403, detail="Access denied")
    role_levels = {"viewer": 0, "editor": 1, "owner": 2}
    if role_levels.get(collab["role"], 0) < role_levels.get(required_role, 0):
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    return project

def register_user(username: str, email: str, password: str, full_name: str = "") -> dict:
    conn = get_db()
    existing = conn.execute("SELECT id FROM users WHERE username=? OR email=?", (username, email)).fetchone()
    if existing:
        conn.close()
        raise HTTPException(status_code=400, detail="Username or email already exists")
    pwd_hash = hash_password(password)
    c = conn.cursor()
    c.execute("INSERT INTO users (username,email,password_hash,full_name) VALUES (?,?,?,?)",
              (username, email, pwd_hash, full_name))
    user_id = c.lastrowid
    conn.commit()
    user = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    conn.close()
    u = dict(user)
    token = create_token(u["id"], u["username"], u["role"])
    return {**u, "token": token}

def login_user(username: str, password: str) -> dict:
    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
    conn.close()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    u = dict(user)
    if u["password_hash"] != hash_password(password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_token(u["id"], u["username"], u["role"])
    return {**u, "token": token}
