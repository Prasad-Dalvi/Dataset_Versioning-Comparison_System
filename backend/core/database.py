import sqlite3
import os
import hashlib

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datavault.db")
STORAGE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "storage")

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn

def init_db():
    os.makedirs(STORAGE_PATH, exist_ok=True)
    conn = get_db()
    c = conn.cursor()
    c.executescript("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        full_name TEXT,
        role TEXT DEFAULT 'user',
        created_at TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS projects (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        description TEXT,
        owner_id INTEGER NOT NULL,
        default_target_column TEXT,
        created_at TEXT DEFAULT (datetime('now')),
        updated_at TEXT DEFAULT (datetime('now')),
        FOREIGN KEY (owner_id) REFERENCES users(id)
    );
    CREATE TABLE IF NOT EXISTS collaborators (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id INTEGER NOT NULL,
        user_id INTEGER NOT NULL,
        role TEXT DEFAULT 'viewer',
        invited_at TEXT DEFAULT (datetime('now')),
        FOREIGN KEY (project_id) REFERENCES projects(id),
        FOREIGN KEY (user_id) REFERENCES users(id),
        UNIQUE(project_id, user_id)
    );
    CREATE TABLE IF NOT EXISTS dataset_versions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id INTEGER NOT NULL,
        version_number INTEGER NOT NULL,
        version_label TEXT,
        commit_message TEXT,
        author_id INTEGER NOT NULL,
        branch_id INTEGER,
        parent_id INTEGER,
        file_path TEXT NOT NULL,
        original_filename TEXT,
        file_size INTEGER DEFAULT 0,
        row_count INTEGER DEFAULT 0,
        col_count INTEGER DEFAULT 0,
        schema_json TEXT,
        stats_json TEXT,
        created_at TEXT DEFAULT (datetime('now')),
        FOREIGN KEY (project_id) REFERENCES projects(id),
        FOREIGN KEY (author_id) REFERENCES users(id)
    );
    CREATE TABLE IF NOT EXISTS branches (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id INTEGER NOT NULL,
        name TEXT NOT NULL,
        created_by INTEGER NOT NULL,
        base_version_id INTEGER,
        created_at TEXT DEFAULT (datetime('now')),
        UNIQUE(project_id, name),
        FOREIGN KEY (project_id) REFERENCES projects(id)
    );
    CREATE TABLE IF NOT EXISTS diff_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id INTEGER NOT NULL,
        version_a_id INTEGER NOT NULL,
        version_b_id INTEGER NOT NULL,
        result_json TEXT,
        severity TEXT DEFAULT 'NONE',
        created_at TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS ml_evaluations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id INTEGER NOT NULL,
        version_id INTEGER NOT NULL,
        target_column TEXT NOT NULL,
        task_type TEXT,
        status TEXT DEFAULT 'pending',
        progress INTEGER DEFAULT 0,
        current_model TEXT,
        error_message TEXT,
        started_at TEXT,
        completed_at TEXT,
        created_at TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS ml_model_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        evaluation_id INTEGER NOT NULL,
        model_name TEXT NOT NULL,
        metrics_json TEXT,
        feature_importances_json TEXT,
        confusion_matrix_json TEXT,
        training_time REAL,
        status TEXT DEFAULT 'pending',
        FOREIGN KEY (evaluation_id) REFERENCES ml_evaluations(id)
    );
    CREATE TABLE IF NOT EXISTS ml_ensemble_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        evaluation_id INTEGER NOT NULL,
        combined_metrics_json TEXT,
        combined_feature_importances_json TEXT,
        created_at TEXT DEFAULT (datetime('now')),
        FOREIGN KEY (evaluation_id) REFERENCES ml_evaluations(id)
    );
    CREATE TABLE IF NOT EXISTS version_best_cache (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id INTEGER NOT NULL,
        version_id INTEGER NOT NULL,
        evaluation_id INTEGER NOT NULL,
        best_model TEXT,
        best_score REAL,
        ensemble_score REAL,
        task_type TEXT,
        target_column TEXT,
        updated_at TEXT DEFAULT (datetime('now')),
        UNIQUE(project_id, version_id, target_column)
    );
    CREATE TABLE IF NOT EXISTS quality_scores (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        version_id INTEGER UNIQUE NOT NULL,
        completeness REAL,
        uniqueness REAL,
        consistency REAL,
        validity REAL,
        overall REAL,
        grade TEXT,
        details_json TEXT,
        created_at TEXT DEFAULT (datetime('now')),
        FOREIGN KEY (version_id) REFERENCES dataset_versions(id)
    );
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id INTEGER NOT NULL,
        version_id INTEGER NOT NULL,
        model_name TEXT,
        input_json TEXT,
        output_json TEXT,
        created_at TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS copilot_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id INTEGER NOT NULL,
        user_id INTEGER NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        provider TEXT,
        created_at TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS activity_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        project_id INTEGER,
        action TEXT NOT NULL,
        details TEXT,
        created_at TEXT DEFAULT (datetime('now'))
    );
    """)
    pwd = hashlib.sha256("Admin@123".encode()).hexdigest()
    c.execute("INSERT OR IGNORE INTO users (username,email,password_hash,full_name,role) VALUES (?,?,?,?,?)",
              ("admin","admin@datavault.com",pwd,"Administrator","admin"))
    conn.commit()
    conn.close()

def log_activity(user_id, project_id, action, details=None):
    try:
        conn = get_db()
        conn.execute("INSERT INTO activity_log (user_id,project_id,action,details) VALUES (?,?,?,?)",
                     (user_id, project_id, action, details))
        conn.commit()
        conn.close()
    except:
        pass
