# Flight
# SkyNote — Personal Flight Logger (Streamlit)

## What this is
A Flighty-like personal flight logger (auto + manual) that saves entries to `flights.json` in the repository root. Auto Mode uses AviationStack (free tier) if you provide an API key; Manual Mode always works.

## Files
- `app.py` — Streamlit app
- `flights.json` — (created automatically) saved entries; commit this to repo to preserve history
- `requirements.txt`

## Setup & run (locally)
1. Clone your repo and put these files there.
2. (Optional) Get a free AviationStack API key: https://aviationstack.com/ and set it in your environment:
   - Linux/macOS:
     ```bash
     export AVIATIONSTACK_KEY="your_key_here"
     ```
   - Windows (PowerShell):
     ```powershell
     $env:AVIATIONSTACK_KEY="your_key_here"
     ```
   If you don't set a key, use Manual Mode.
3. Install deps:
   ```bash
   pip install -r requirements.txt
