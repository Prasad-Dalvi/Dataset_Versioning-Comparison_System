demo link https://youtu.be/sHF0ty1PTp8?si=NX5kVgbuP-v256mo 
# DataVault V2 — Dataset Versioning & ML Platform

See the full README below for setup, features, and API reference.

## Quick Start

### Windows
Double-click START.bat
Open index.html in browser from file manager

### Mac/Linux
```bash
cd backend && python -m venv venv && ./venv/Scripts/activate && pip install -r requirements.txt && python main.py
```
Open frontend/index.html in your browser.
Default admin: admin / Admin@123

## What Changed in This Upgrade

### copilot.py — Smart AI Agent (COMPLETE REWRITE)
- Privacy-first: AI only receives pre-computed statistics, never raw CSV rows
- Supports 5 providers: xAI Grok (default), Anthropic Claude, OpenAI GPT, Google Gemini, DeepSeek
- Rich structured context: quality scores, model leaderboards, drift highlights, feature importances
- Transparent context builder with /api/copilot/context/{id} inspection endpoint
- Smart suggested questions based on project state

### routes.py — Bug Fixes + New Endpoints
- Fixed PATCH /projects/{id} body type (was `dict`, now typed Pydantic model)
- Added DELETE /copilot/history/{project_id} — clear chat
- Added GET /copilot/context/{project_id} — AI transparency
- Added GET /copilot/suggested-questions/{project_id} — smart chips

### index.html — Copilot UI Overhaul
- Provider toggle bar with per-provider API key memory
- AI Context viewer (modal showing exactly what data AI sees)
- Context-aware suggested question chips
- Clear history button
- Rich markdown rendering in responses (headers, bullets, bold, code)
- Provider color-coded response bubbles with timestamps
- Textarea input (Shift+Enter newline, Enter to send)
