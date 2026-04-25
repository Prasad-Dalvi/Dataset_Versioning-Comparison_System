"""
DataVault — AI Copilot (minimal, no dataset context)
Supported providers: groq | gemini | claude
"""

import httpx

# ── Groq SDK ──────────────────────────────────────────────────────────────────
try:
    from groq import Groq as GroqClient
    _groq_available = True
except ImportError:
    GroqClient = None
    _groq_available = False

# ── Anthropic SDK ─────────────────────────────────────────────────────────────
try:
    import anthropic as _anthropic
    _claude_available = True
except ImportError:
    _anthropic = None
    _claude_available = False

# ── Config ────────────────────────────────────────────────────────────────────
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
]
GEMINI_URL = (
    "https://generativelanguage.googleapis.com"
    "/v1beta/models/gemini-1.5-flash:generateContent"
)
CLAUDE_MODEL = "claude-sonnet-4-20250514"

SYSTEM_PROMPT = (
    "You are DataVault AI, a helpful data analysis assistant. "
    "Be concise and friendly."
)


def _normalise_provider(raw: str) -> str:
    """Map any reasonable provider string to groq | gemini | claude."""
    s = (raw or "").strip().lower()
    if any(x in s for x in ("groq", "grok", "llama", "mixtral", "gsk")):
        return "groq"
    if any(x in s for x in ("gemini", "google", "aiza")):
        return "gemini"
    if any(x in s for x in ("claude", "anthropic", "sk-ant")):
        return "claude"
    return s   # return as-is so the error shows the real value


# ── Main entry point ──────────────────────────────────────────────────────────

async def chat(
    project_id: int,
    user_id: int,
    message: str,
    provider: str,
    api_key: str,
    history: list,
) -> str:

    provider = _normalise_provider(provider)

    if not api_key or not api_key.strip():
        hints = {
            "groq":   ("Groq",   "https://console.groq.com/keys",      "gsk_"),
            "gemini": ("Gemini", "https://aistudio.google.com/apikey", "AIza"),
            "claude": ("Claude", "https://console.anthropic.com/",     "sk-ant-"),
        }
        name, url, prefix = hints.get(provider, ("your provider", "#", "..."))
        return (
            f"❌ Please enter your **{name}** API key.\n\n"
            f"Get one at {url}\n"
            f"Your key starts with **{prefix}**"
        )

    try:
        if provider == "groq":
            return await _groq(api_key.strip(), history, message)
        elif provider == "gemini":
            return await _gemini(api_key.strip(), history, message)
        elif provider == "claude":
            return await _claude(api_key.strip(), history, message)
        else:
            # Show the raw value so we can debug
            return (
                f"Unrecognised provider: {provider!r} "
                f"(bytes: {provider.encode()!r}). "
                f"Expected: groq | gemini | claude"
            )
    except Exception as e:
        return f"Error ({provider}): {str(e)[:300]}"


# ── Groq ──────────────────────────────────────────────────────────────────────

async def _groq(api_key: str, history: list, message: str) -> str:
    if not _groq_available:
        return "groq package not installed. Run: pip install groq"

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in history[-6:]:
        if h.get("role") in ("user", "assistant"):
            messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": message})

    client = GroqClient(api_key=api_key)
    last_error = "No models tried"
    for model in GROQ_MODELS:
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, max_tokens=600, temperature=0.7
            )
            return resp.choices[0].message.content
        except Exception as e:
            err = str(e).lower()
            if any(k in err for k in ("not found", "decommissioned", "does not exist")):
                last_error = str(e)
                continue
            if "401" in err or "authentication" in err:
                return (
                    "Groq: Invalid API key (401)\n"
                    "Key starts with gsk_ -- get one at https://console.groq.com/keys"
                )
            if "429" in err:
                return "Groq rate limited -- wait a moment and retry."
            raise

    return f"All Groq models unavailable: {last_error}"


# ── Gemini ────────────────────────────────────────────────────────────────────

async def _gemini(api_key: str, history: list, message: str) -> str:
    turns = []
    for h in history[-6:]:
        if h.get("role") in ("user", "assistant"):
            role = "model" if h["role"] == "assistant" else "user"
            turns.append({"role": role, "parts": [{"text": h["content"]}]})

    if not turns or turns[0]["role"] != "user":
        turns.insert(0, {"role": "user", "parts": [{"text": "Hello"}]})
    turns.append({"role": "user", "parts": [{"text": message}]})

    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": turns,
        "generationConfig": {"maxOutputTokens": 600, "temperature": 0.7},
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(f"{GEMINI_URL}?key={api_key}", json=payload)

    if resp.status_code == 429:
        return "Gemini rate limit -- free tier is 15 req/min. Wait ~60 s."
    if resp.status_code in (400, 401):
        return (
            "Gemini: Invalid API key\n"
            "Get one at https://aistudio.google.com/apikey (starts with AIza)"
        )
    resp.raise_for_status()

    data = resp.json()
    candidates = data.get("candidates", [])
    if not candidates:
        reason = data.get("promptFeedback", {}).get("blockReason", "unknown")
        return f"Gemini blocked the request: {reason}"
    return candidates[0]["content"]["parts"][0].get("text", "")


# ── Claude ────────────────────────────────────────────────────────────────────

async def _claude(api_key: str, history: list, message: str) -> str:
    if not _claude_available:
        return "anthropic package not installed. Run: pip install anthropic"

    messages = []
    for h in history[-6:]:
        if h.get("role") in ("user", "assistant"):
            messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": message})

    client = _anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=600,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    return resp.content[0].text


# ── History helpers ───────────────────────────────────────────────────────────

def get_history(project_id: int) -> list:
    try:
        from .database import get_db
        conn = get_db()
        rows = conn.execute(
            "SELECT role, content, provider, created_at FROM copilot_history "
            "WHERE project_id=? ORDER BY created_at ASC LIMIT 100",
            (project_id,),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


def get_context_preview(project_id: int) -> dict:
    return {
        "system_prompt": SYSTEM_PROMPT,
        "raw_data_sent": False,
        "supported_providers": ["groq", "gemini", "claude"],
        "groq_note":   "api.groq.com -- key starts with gsk_",
        "gemini_note": "generativelanguage.googleapis.com -- key starts with AIza",
        "claude_note": "api.anthropic.com -- key starts with sk-ant-",
    }
