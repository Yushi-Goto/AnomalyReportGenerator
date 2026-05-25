# AGENTS.md

## Project
AnomalyReportGenerator is a FastAPI-based anomaly report generation project using Anomalib and OpenAI VLM structured outputs.

## Security
- Never read, print, edit, summarize, or copy `.env` or `.env.*`.
- Use `.env.example` only to understand required environment variables.
- Never expose API keys, tokens, credentials, or secrets.
- Never run commands that print environment variables, such as `env`, `printenv`, `set`, `Get-ChildItem Env:`, or `echo $env:OPENAI_API_KEY`.

## Files to Avoid
- Do not modify `datasets/`, `models/`, or `results/` unless explicitly requested.
- Do not modify large binary files or model checkpoints.
- Do not commit generated files unless explicitly requested.

## Development Style
- Prefer small, focused changes.
- Explain the planned change before large refactors.
- Keep API responses backward-compatible unless explicitly requested.
- Update README when commands, API behavior, or setup steps change.

## Project Commands
Install dependencies in this order:
```bash
pip install -r requirements-torch-cu121.txt
pip install -r requirements-anomalib.txt
pip install -r requirements.txt