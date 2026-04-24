"""
DataVault V2 — Smart AI Copilot
Sends ONLY pre-computed statistics/metrics to the LLM — never raw dataset rows.

Supported providers (as of 2025):
  ⚡ xAI Grok       → grok-beta  (grok-3-mini does NOT exist → 400)
  🤖 Anthropic      → claude-3-haiku-20240307
  💚 OpenAI         → gpt-4o-mini  (gpt-3.5-turbo is deprecated → 404)
  ✨ Google Gemini  → gemini-2.0-flash  (free: 15 req/min → 429 if exceeded)
  🔮 DeepSeek       → deepseek-chat

Error codes handled per provider:
  400 → wrong model / malformed body  (Grok: usually wrong model name)
  401 → bad API key
  402 → insufficient balance          (DeepSeek)
  404 → model not found / deprecated
  429 → rate limited                  (Gemini free tier: 15 req/min)
  5xx → provider server error
"""

import json
import asyncio
import httpx
from .database import get_db

# ── Provider configuration ────────────────────────────────────────────────────

# Grok: OpenAI-compatible. "grok-3-mini" does NOT exist → causes 400.
# Valid models (2025): grok-beta, grok-2-1212, grok-2-mini, grok-3-beta, grok-3-mini-beta
# Use grok-beta as it's the most stable free-accessible model.
GROK_MODELS = ["grok-beta", "grok-2-mini", "grok-2-1212"]

# Gemini: free tier = 15 requests/min. Newest free model = gemini-2.0-flash.
# Try in order — skip any that return 404 (model unavailable in region).
GEMINI_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash-8b",
    "gemini-1.5-flash",
]
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# OpenAI: gpt-3.5-turbo is being phased out. gpt-4o-mini is the new cheap default.
OPENAI_MODELS = ["gpt-4o-mini", "gpt-3.5-turbo"]

PROVIDER_URLS = {
    "anthropic": "https://api.anthropic.com/v1/messages",
    "grok":      "https://api.x.ai/v1/chat/completions",
    "openai":    "https://api.openai.com/v1/chat/completions",
    "deepseek":  "https://api.deepseek.com/v1/chat/completions",
    # gemini URL is built dynamically per model
}

# ── Human-readable error messages ─────────────────────────────────────────────

def _friendly_error(provider: str, status: int, body: str) -> str:
    base = {
        400: {
            "grok":      "❌ **Grok 400 Bad Request** — wrong model name or malformed request. The model `grok-3-mini` does **not** exist. Using `grok-beta` instead.",
            "openai":    "❌ **OpenAI 400 Bad Request** — check your request format or model name.",
            "anthropic": "❌ **Anthropic 400 Bad Request** — check your API key format (`sk-ant-...`).",
            "deepseek":  "❌ **DeepSeek 400 Bad Request** — check your request format.",
            "gemini":    "❌ **Gemini 400 Bad Request** — check your API key.",
        },
        401: {
            "grok":      "❌ **Grok: Invalid API key** — get yours at https://console.x.ai",
            "openai":    "❌ **OpenAI: Invalid API key** — get yours at https://platform.openai.com/api-keys",
            "anthropic": "❌ **Anthropic: Invalid API key** — must start with `sk-ant-`. Get yours at https://console.anthropic.com",
            "deepseek":  "❌ **DeepSeek: Invalid API key** — get yours at https://platform.deepseek.com",
            "gemini":    "❌ **Gemini: Invalid API key** — get yours at https://aistudio.google.com/apikey",
        },
        402: {
            "deepseek":  "❌ **DeepSeek: Insufficient balance** — top up at https://platform.deepseek.com",
            "openai":    "❌ **OpenAI: Billing issue** — check https://platform.openai.com/account/billing",
        },
        404: {
            "grok":      "❌ **Grok: Model not found** — trying fallback models automatically.",
            "openai":    "❌ **OpenAI: Model not found** — `gpt-3.5-turbo` may be deprecated. Trying `gpt-4o-mini`.",
            "anthropic": "❌ **Anthropic: Model not found** — check model name.",
            "gemini":    "❌ **Gemini: Model not found** — trying next available model.",
            "deepseek":  "❌ **DeepSeek: Model not found** — check model name.",
        },
        429: {
            "gemini":    "⏳ **Gemini: Rate limited** — free tier allows 15 requests/min. Please wait 60 seconds and try again. Or upgrade at https://aistudio.google.com",
            "openai":    "⏳ **OpenAI: Rate limited** — you've hit your quota. Check https://platform.openai.com/account/limits",
            "anthropic": "⏳ **Anthropic: Rate limited** — please wait a moment and retry.",
            "grok":      "⏳ **Grok: Rate limited** — please wait a moment and retry.",
            "deepseek":  "⏳ **DeepSeek: Rate limited** — please wait a moment and retry.",
        },
        500: "❌ **Provider server error (500)** — the AI service is temporarily down. Try again in a moment.",
        503: "❌ **Provider unavailable (503)** — the AI service is overloaded. Try again shortly.",
    }.get(status, {})

    if isinstance(base, dict):
        msg = base.get(provider, f"❌ **{provider.title()} error {status}**: {body[:200]}")
    else:
        msg = base

    return msg


# ── Context builder — only computed stats, never raw rows ────────────────────

def _round(v, n=3):
    try:
        return round(float(v), n)
    except Exception:
        return v


def build_dataset_context(project_id: int) -> dict:
    """
    Pulls every pre-computed metric from DB — no parquet/CSV data ever touched.
    """
    conn = get_db()

    project = conn.execute("SELECT * FROM projects WHERE id=?", (project_id,)).fetchone()
    if not project:
        conn.close()
        return {}
    p = dict(project)

    version_rows = conn.execute("""
        SELECT dv.id, dv.version_number, dv.version_label, dv.row_count,
               dv.col_count, dv.file_size, dv.commit_message, dv.created_at,
               dv.schema_json, dv.stats_json,
               b.name AS branch_name,
               qs.completeness, qs.uniqueness, qs.consistency,
               qs.validity, qs.overall AS quality_overall, qs.grade,
               qs.details_json
        FROM dataset_versions dv
        LEFT JOIN branches b  ON dv.branch_id = b.id
        LEFT JOIN quality_scores qs ON dv.id = qs.version_id
        WHERE dv.project_id=?
        ORDER BY dv.version_number DESC LIMIT 10
    """, (project_id,)).fetchall()
    versions = []
    for r in version_rows:
        v = dict(r)
        v["schema"]          = json.loads(v.pop("schema_json")  or "{}")
        v["stats"]           = json.loads(v.pop("stats_json")   or "{}")
        v["quality_details"] = json.loads(v.pop("details_json") or "{}")
        versions.append(v)

    eval_row = conn.execute("""
        SELECT me.id, me.target_column, me.task_type, me.status,
               me.created_at, me.completed_at, me.version_id
        FROM ml_evaluations me
        WHERE me.project_id=? AND me.status='completed'
        ORDER BY me.completed_at DESC LIMIT 1
    """, (project_id,)).fetchone()

    latest_eval = None
    if eval_row:
        ev = dict(eval_row)
        model_rows = conn.execute("""
            SELECT model_name, metrics_json, feature_importances_json,
                   training_time, status
            FROM ml_model_results WHERE evaluation_id=? AND status='completed'
        """, (ev["id"],)).fetchall()
        models = []
        for m in model_rows:
            md = dict(m)
            md["metrics"]             = json.loads(md.pop("metrics_json") or "{}")
            md["feature_importances"] = json.loads(md.pop("feature_importances_json") or "{}")
            models.append(md)

        ensemble = conn.execute("""
            SELECT combined_metrics_json, combined_feature_importances_json
            FROM ml_ensemble_results WHERE evaluation_id=? ORDER BY id DESC LIMIT 1
        """, (ev["id"],)).fetchone()

        ev["models"] = models
        ev["ensemble"] = {
            "metrics":              json.loads(ensemble["combined_metrics_json"] or "{}"),
            "feature_importances":  json.loads(ensemble["combined_feature_importances_json"] or "{}")
        } if ensemble else None

        if models:
            best = max(models, key=lambda m: m["metrics"].get("cv_score", 0))
            ev["best_model"]    = best["model_name"]
            ev["best_cv_score"] = _round(best["metrics"].get("cv_score", 0))

        latest_eval = ev

    leaderboard = conn.execute("""
        SELECT vbc.version_id, vbc.best_model, vbc.best_score,
               vbc.ensemble_score, vbc.task_type, vbc.target_column,
               dv.version_number, dv.row_count
        FROM version_best_cache vbc
        JOIN dataset_versions dv ON vbc.version_id = dv.id
        WHERE vbc.project_id=?
        ORDER BY vbc.best_score DESC LIMIT 10
    """, (project_id,)).fetchall()

    diff_row = conn.execute("""
        SELECT result_json, severity, version_a_id, version_b_id, created_at
        FROM diff_results WHERE project_id=?
        ORDER BY created_at DESC LIMIT 1
    """, (project_id,)).fetchone()
    latest_diff = None
    if diff_row:
        d = dict(diff_row)
        result = json.loads(d.pop("result_json") or "{}")
        latest_diff = {**d, "summary":     result.get("summary", {}),
                           "schema_diff":  result.get("schema_diff", {}),
                           "drift_highlights": _top_drift(result.get("drift", {}))}

    history = conn.execute("""
        SELECT role, content FROM copilot_history
        WHERE project_id=? ORDER BY created_at DESC LIMIT 6
    """, (project_id,)).fetchall()

    conn.close()

    return {
        "project":               p,
        "versions":              versions,
        "latest_evaluation":     latest_eval,
        "performance_leaderboard": [dict(r) for r in leaderboard],
        "latest_diff":           latest_diff,
        "recent_history":        [dict(r) for r in reversed(history)],
    }


def _top_drift(drift: dict, n=5) -> list:
    ranked = sorted(
        [(col, info) for col, info in drift.items()
         if info.get("drift_level", "NONE") != "NONE"],
        key=lambda x: {"HIGH": 3, "MEDIUM": 2, "LOW": 1}.get(x[1].get("drift_level", "NONE"), 0),
        reverse=True
    )
    out = []
    for col, info in ranked[:n]:
        entry = {"column": col, "drift_level": info.get("drift_level")}
        if "ks_statistic" in info:
            entry["ks_stat"]   = _round(info["ks_statistic"])
            entry["psi"]       = _round(info.get("psi", 0))
            entry["mean_shift"] = f"{_round(info.get('mean_a',0))} → {_round(info.get('mean_b',0))}"
        elif "chi2_stat" in info:
            entry["chi2_stat"] = _round(info["chi2_stat"])
        out.append(entry)
    return out


# ── System prompt builder ─────────────────────────────────────────────────────

def build_system_prompt(ctx: dict) -> str:
    proj     = ctx.get("project", {})
    versions = ctx.get("versions", [])
    ev       = ctx.get("latest_evaluation")
    lb       = ctx.get("performance_leaderboard", [])
    diff     = ctx.get("latest_diff")

    lines = [
        "You are the DataVault AI Copilot — an expert data scientist assistant.",
        "You answer questions using ONLY the pre-computed statistics below. Never request raw data.",
        "Be concise, insightful, and actionable. Use markdown formatting.",
        "",
        f"## Project: {proj.get('name','Unknown')}",
        f"Description: {proj.get('description') or 'N/A'}",
        f"Default target column: {proj.get('default_target_column') or 'Not set'}",
        f"Total versions: {len(versions)}",
        "",
    ]

    if versions:
        lines.append("## Dataset Versions (latest first)")
        for v in versions[:6]:
            q     = v.get("quality_overall")
            grade = v.get("grade", "N/A")
            cols  = list(v.get("schema", {}).keys())
            qd    = v.get("quality_details", {})
            lines.append(
                f"- **{v['version_label']}** ({v.get('branch_name','main')}) | "
                f"{v['row_count']:,} rows × {v['col_count']} cols | "
                f"Quality: {grade} ({_round(q,1) if q else 'N/A'}%) | "
                f"\"{v.get('commit_message') or 'N/A'}\""
            )
            if qd:
                lines.append(
                    f"  → Completeness:{qd.get('completeness','N/A')}% "
                    f"Uniqueness:{qd.get('uniqueness','N/A')}% "
                    f"Consistency:{qd.get('consistency','N/A')}% "
                    f"Validity:{qd.get('validity','N/A')}% "
                    f"Nulls:{qd.get('null_count',0)} Dups:{qd.get('duplicate_rows',0)}"
                )
            if cols:
                lines.append(f"  Columns: {', '.join(cols[:20])}" + (" ..." if len(cols)>20 else ""))
            stats = v.get("stats", {})
            num = [
                f"{c}[μ={_round(s['mean'])},σ={_round(s.get('std',0))},min={_round(s['min'])},max={_round(s['max'])}]"
                for c, s in stats.items()
                if isinstance(s, dict) and "mean" in s and s["mean"] is not None
            ]
            if num:
                lines.append(f"  Stats: {' | '.join(num[:8])}")
        lines.append("")

    if ev:
        lines.append("## Latest ML Evaluation")
        lines.append(
            f"Target: `{ev['target_column']}` | Task: {ev['task_type']} | "
            f"Best model: {ev.get('best_model','N/A')} (CV={_round(ev.get('best_cv_score',0)*100,1)}%)"
        )
        if ev.get("models"):
            lines.append("Model leaderboard:")
            for m in sorted(ev["models"], key=lambda x: x["metrics"].get("cv_score",0), reverse=True):
                mt = m["metrics"]
                lines.append(
                    f"  - {m['model_name']}: CV={_round(mt.get('cv_score',0)*100,1)}% "
                    f"Acc={_round((mt.get('accuracy') or mt.get('r2') or 0)*100,1)}% "
                    f"F1={_round(mt.get('f1',0)*100,1)}% "
                    f"AUC={_round(mt.get('auc_roc',0)*100,1)}% "
                    f"time={_round(mt.get('training_time',0),2)}s"
                )
        ens = ev.get("ensemble")
        if ens and ens.get("metrics"):
            wa = ens["metrics"].get("weighted_avg", {})
            lines.append(
                f"Ensemble (weighted): Acc={_round((wa.get('accuracy') or wa.get('r2') or 0)*100,1)}% "
                f"F1={_round(wa.get('f1',0)*100,1)}%"
            )
            fi = ens.get("feature_importances", {})
            if fi:
                top = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:8]
                lines.append("Top features: " + " | ".join(f"{k}={_round(v*100,1)}%" for k,v in top))
        lines.append("")

    if lb:
        lines.append("## Version Performance Leaderboard")
        for e in lb[:5]:
            lines.append(
                f"  V{e['version_number']}: {e['best_model']} "
                f"score={_round(e['best_score']*100,1)}% "
                f"ensemble={_round(e['ensemble_score']*100,1)}%"
            )
        lines.append("")

    if diff:
        lines.append("## Latest Drift Analysis")
        lines.append(f"Severity: {diff['severity']}")
        s = diff.get("summary", {})
        lines.append(
            f"Schema issues:{s.get('schema_issues',0)} | "
            f"High-drift cols:{s.get('high_drift_cols',0)} | "
            f"Medium-drift cols:{s.get('medium_drift_cols',0)}"
        )
        sd = diff.get("schema_diff", {})
        if sd.get("added_columns"):
            lines.append(f"Added: {sd['added_columns']}")
        if sd.get("removed_columns"):
            lines.append(f"Removed: {sd['removed_columns']}")
        for d in diff.get("drift_highlights", []):
            detail = f"KS={d.get('ks_stat')} PSI={d.get('psi')} shift={d.get('mean_shift','')}" \
                     if "ks_stat" in d else f"Chi²={d.get('chi2_stat')}"
            lines.append(f"  Drift → {d['column']} [{d['drift_level']}] {detail}")
        lines.append("")

    lines += [
        "## Instructions",
        "- Use ONLY the above stats — never hallucinate numbers.",
        "- If something isn't in the context, say so clearly.",
        "- Be concise and actionable. Suggest improvements when relevant.",
    ]
    return "\n".join(lines)


# ── Chat entry point ──────────────────────────────────────────────────────────

async def chat(project_id: int, user_id: int, message: str, provider: str,
               api_key: str, history: list) -> str:

    ctx           = build_dataset_context(project_id)
    system_prompt = build_system_prompt(ctx)

    conn = get_db()
    conn.execute(
        "INSERT INTO copilot_history (project_id,user_id,role,content,provider) VALUES (?,?,?,?,?)",
        (project_id, user_id, "user", message, provider)
    )
    conn.commit()
    conn.close()

    try:
        response = await _call_llm(provider, api_key, system_prompt, history, message)
    except Exception as e:
        response = str(e)   # already formatted by _friendly_error

    conn = get_db()
    conn.execute(
        "INSERT INTO copilot_history (project_id,user_id,role,content,provider) VALUES (?,?,?,?,?)",
        (project_id, user_id, "assistant", response, provider)
    )
    conn.commit()
    conn.close()

    return response


# ── LLM router ───────────────────────────────────────────────────────────────

async def _call_llm(provider: str, api_key: str, system: str,
                    history: list, message: str) -> str:
    """
    Route to the correct provider with proper model names, auth headers,
    body shapes, and per-status-code error messages.
    """
    if not api_key or not api_key.strip():
        raise Exception(
            f"❌ No API key provided for **{provider}**. "
            "Enter your key in the field above."
        )

    # Build conversation turns (last 10)
    turns = []
    for h in history[-10:]:
        if h.get("role") in ("user", "assistant"):
            turns.append({"role": h["role"], "content": h["content"]})
    turns.append({"role": "user", "content": message})

    async with httpx.AsyncClient(timeout=60.0) as client:

        # ── Anthropic Claude ─────────────────────────────────────────────
        if provider == "anthropic":
            return await _call_anthropic(client, api_key, system, turns)

        # ── xAI Grok ─────────────────────────────────────────────────────
        elif provider == "grok":
            return await _call_grok(client, api_key, system, turns)

        # ── OpenAI GPT ───────────────────────────────────────────────────
        elif provider == "openai":
            return await _call_openai(client, api_key, system, turns)

        # ── Google Gemini ────────────────────────────────────────────────
        elif provider == "gemini":
            return await _call_gemini(client, api_key, system, turns)

        # ── DeepSeek ─────────────────────────────────────────────────────
        elif provider == "deepseek":
            return await _call_deepseek(client, api_key, system, turns)

        else:
            raise Exception(
                f"❌ Unknown provider `{provider}`. "
                "Choose: anthropic, grok, openai, gemini, deepseek"
            )


def _handle_http_error(provider: str, e: httpx.HTTPStatusError) -> str:
    status = e.response.status_code
    try:
        body = e.response.json()
        body_str = json.dumps(body)
    except Exception:
        body_str = e.response.text[:300]
    return _friendly_error(provider, status, body_str)


# ── Per-provider callers ──────────────────────────────────────────────────────

async def _call_anthropic(client, api_key, system, turns):
    """
    Anthropic uses its own schema:
    - system is a top-level string (NOT inside messages)
    - messages = [{role, content}]
    - auth: x-api-key header
    """
    try:
        resp = await client.post(
            PROVIDER_URLS["anthropic"],
            headers={
                "x-api-key":         api_key.strip(),
                "anthropic-version": "2023-06-01",
                "content-type":      "application/json",
            },
            json={
                "model":      "claude-3-haiku-20240307",
                "max_tokens": 1500,
                "system":     system,
                "messages":   turns,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data["content"][0]["text"]
    except httpx.HTTPStatusError as e:
        raise Exception(_handle_http_error("anthropic", e))
    except httpx.TimeoutException:
        raise Exception("⏳ **Anthropic: Request timed out.** Please try again.")


async def _call_grok(client, api_key, system, turns):
    """
    xAI Grok uses OpenAI-compatible schema.
    IMPORTANT: 'grok-3-mini' does NOT exist → 400 Bad Request.
    Valid models (2025): grok-beta, grok-2-mini, grok-2-1212, grok-3-beta, grok-3-mini-beta
    Try in fallback order.
    """
    last_err = "No models tried"
    for model in GROK_MODELS:
        try:
            resp = await client.post(
                PROVIDER_URLS["grok"],
                headers={
                    "Authorization": f"Bearer {api_key.strip()}",
                    "Content-Type":  "application/json",
                },
                json={
                    "model":      model,
                    "max_tokens": 1500,
                    "messages":   [{"role": "system", "content": system}] + turns,
                },
            )
            if resp.status_code == 400:
                # Wrong model name — try next
                last_err = _friendly_error("grok", 400, resp.text)
                continue
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 400:
                last_err = _friendly_error("grok", 400, e.response.text)
                continue
            raise Exception(_handle_http_error("grok", e))
        except httpx.TimeoutException:
            raise Exception("⏳ **Grok: Request timed out.** Please try again.")

    raise Exception(
        f"{last_err}\n\n"
        "All Grok models returned 400. Ensure your API key is valid at https://console.x.ai"
    )


async def _call_openai(client, api_key, system, turns):
    """
    OpenAI standard chat completions.
    gpt-3.5-turbo is being deprecated — try gpt-4o-mini first.
    """
    last_err = "No models tried"
    for model in OPENAI_MODELS:
        try:
            resp = await client.post(
                PROVIDER_URLS["openai"],
                headers={
                    "Authorization": f"Bearer {api_key.strip()}",
                    "Content-Type":  "application/json",
                },
                json={
                    "model":      model,
                    "max_tokens": 1500,
                    "messages":   [{"role": "system", "content": system}] + turns,
                },
            )
            if resp.status_code == 404:
                last_err = _friendly_error("openai", 404, resp.text)
                continue
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                last_err = _friendly_error("openai", 404, e.response.text)
                continue
            raise Exception(_handle_http_error("openai", e))
        except httpx.TimeoutException:
            raise Exception("⏳ **OpenAI: Request timed out.** Please try again.")

    raise Exception(f"{last_err}\n\nAll OpenAI models failed. Check https://platform.openai.com/account/billing")


async def _call_gemini(client, api_key, system, turns):
    """
    Google Gemini uses its own schema (NOT OpenAI-compatible).
    - Auth: API key as query param (?key=...)
    - Body: {"contents": [{"parts": [{"text": "..."}]}]}
    - Free tier: 15 requests/minute → 429 when exceeded
    - Model names change — try list in order, skip 404s
    """
    # Build Gemini-format content from system + turns
    parts = [{"text": f"System instructions:\n{system}\n\n---\n\n"}]
    for t in turns:
        role_label = "USER" if t["role"] == "user" else "ASSISTANT"
        parts.append({"text": f"{role_label}: {t['content']}\n\n"})

    payload = {
        "contents":         [{"parts": parts}],
        "generationConfig": {"maxOutputTokens": 1500, "temperature": 0.7},
    }

    last_err = "No models tried"
    for model in GEMINI_MODELS:
        url = f"{GEMINI_BASE.format(model=model)}?key={api_key.strip()}"
        try:
            resp = await client.post(url, json=payload)

            if resp.status_code == 404:
                # Model not available in this region/plan — try next
                last_err = f"Model `{model}` returned 404 (not available)"
                continue

            if resp.status_code == 429:
                raise Exception(
                    "⏳ **Gemini: Rate limit hit (429)**\n\n"
                    "The **free tier allows only 15 requests per minute**. "
                    "Please wait ~60 seconds and try again.\n\n"
                    "To remove this limit, upgrade at https://aistudio.google.com"
                )

            resp.raise_for_status()
            data = resp.json()

            # Parse Gemini response
            candidates = data.get("candidates", [])
            if not candidates:
                # Check for content blocking
                pf = data.get("promptFeedback", {})
                block = pf.get("blockReason", "unknown")
                raise Exception(f"❌ **Gemini blocked the request**: {block}")

            content = candidates[0].get("content", {})
            parts_out = content.get("parts", [{}])
            return parts_out[0].get("text", "")

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                last_err = f"Model `{model}` not found (404)"
                continue
            if e.response.status_code == 429:
                raise Exception(
                    "⏳ **Gemini: Rate limit hit (429)**\n\n"
                    "Free tier = 15 requests/min. Wait 60s then retry.\n"
                    "Upgrade at https://aistudio.google.com"
                )
            raise Exception(_handle_http_error("gemini", e))
        except httpx.TimeoutException:
            raise Exception("⏳ **Gemini: Request timed out.** Please try again.")

    raise Exception(
        f"❌ **All Gemini models unavailable.**\n"
        f"Last error: {last_err}\n\n"
        "Tried: " + ", ".join(GEMINI_MODELS) + "\n"
        "Verify your key at https://aistudio.google.com/apikey"
    )


async def _call_deepseek(client, api_key, system, turns):
    """
    DeepSeek uses OpenAI-compatible schema.
    Models: deepseek-chat (general), deepseek-reasoner (o1-style, slower)
    Error 402 = insufficient balance.
    """
    try:
        resp = await client.post(
            PROVIDER_URLS["deepseek"],
            headers={
                "Authorization": f"Bearer {api_key.strip()}",
                "Content-Type":  "application/json",
            },
            json={
                "model":      "deepseek-chat",
                "max_tokens": 1500,
                "messages":   [{"role": "system", "content": system}] + turns,
            },
        )
        if resp.status_code == 402:
            raise Exception(
                "❌ **DeepSeek: Insufficient balance (402)**\n"
                "Top up your account at https://platform.deepseek.com"
            )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except httpx.HTTPStatusError as e:
        raise Exception(_handle_http_error("deepseek", e))
    except httpx.TimeoutException:
        raise Exception("⏳ **DeepSeek: Request timed out.** Please try again.")


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_history(project_id: int) -> list:
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM copilot_history WHERE project_id=? ORDER BY created_at ASC LIMIT 100",
        (project_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_context_preview(project_id: int) -> dict:
    """Returns the exact structured context the AI receives — full transparency."""
    ctx    = build_dataset_context(project_id)
    system = build_system_prompt(ctx)
    return {
        "context":               ctx,
        "system_prompt_preview": system[:3000] + ("..." if len(system) > 3000 else ""),
        "system_prompt_chars":   len(system),
        "raw_data_sent":         False,
        "data_source":           "pre-computed statistics only — no raw CSV/parquet rows",
        "providers_config": {
            "grok":      {"models_tried": GROK_MODELS,   "note": "grok-3-mini does not exist"},
            "gemini":    {"models_tried": GEMINI_MODELS, "note": "free tier = 15 req/min"},
            "openai":    {"models_tried": OPENAI_MODELS, "note": "gpt-3.5-turbo deprecated → gpt-4o-mini"},
            "anthropic": {"model": "claude-3-haiku-20240307"},
            "deepseek":  {"model": "deepseek-chat", "note": "402 = no balance"},
        }
    }