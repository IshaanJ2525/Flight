# app.py
"""
SkyNote — Flighty-like personal flight logger (free-first).
- Auto Mode uses AviationStack if env var AVIATIONSTACK_KEY is set.
- Manual Mode always available.
- Saves entries to flights.json in repo root.
- Optional auto git commit (requires git + credentials in host).
"""

import os
import json
import subprocess
from pathlib import Path
from datetime import datetime, date
from math import radians, cos, sin, asin, sqrt

import streamlit as st
import requests
import pandas as pd

# -----------------------
# Config
# -----------------------
st.set_page_config(page_title="SkyNote (Flight Logger)", layout="wide")
REPO_ROOT = Path(".").resolve()
DATA_FILE = REPO_ROOT / "flights.json"
API_KEY = os.environ.get("AVIATIONSTACK_KEY")  # set this env var to use auto mode
API_URL = "http://api.aviationstack.com/v1/flights"

# -----------------------
# Helpers: files & git
# -----------------------
def ensure_data_file():
    if not DATA_FILE.exists():
        DATA_FILE.write_text(json.dumps([], indent=2))

def load_flights():
    ensure_data_file()
    try:
        return json.loads(DATA_FILE.read_text())
    except Exception:
        return []

def save_flights(data, auto_git=False, commit_msg=None):
    DATA_FILE.write_text(json.dumps(data, indent=2, default=str))
    if auto_git:
        _ = git_add_commit_push([str(DATA_FILE)], commit_msg or "Update flights.json")

def git_add_commit_push(paths, commit_msg):
    """
    Try to git add/commit/push the given paths.
    Returns (ok: bool, message: str)
    """
    try:
        subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError:
        return False, "Not a git repository"

    try:
        subprocess.run(["git", "add"] + paths, check=True)
        subprocess.run(["git", "commit", "-m", commit_msg or "Update flights.json"], check=True)
    except subprocess.CalledProcessError as e:
        return False, f"Git commit failed: {e}"
    try:
        subprocess.run(["git", "push"], check=True)
        return True, "Committed and pushed"
    except subprocess.CalledProcessError as e:
        return True, f"Committed locally but push failed: {e}"

# -----------------------
# Helpers: distance/time
# -----------------------
def haversine_km(lat1, lon1, lat2, lon2):
    # return distance in km
    if any(v is None for v in [lat1, lon1, lat2, lon2]):
        return None
    # convert decimals to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    km = 6371 * c
    return round(km, 1)

def parse_iso(dt):
    if dt is None:
        return None
    try:
        return datetime.fromisoformat(dt.replace("Z", "+00:00"))
    except Exception:
        try:
            return pd.to_datetime(dt)
        except Exception:
            return None

# -----------------------
# API: AviationStack (Auto mode)
# -----------------------
def fetch_via_aviationstack(flight_iata: str, flight_date: str):
    """
    Query AviationStack flights endpoint for given IATA flight and date (YYYY-MM-DD).
    Returns a normalized dict or None.
    Note: free-tier requires signup and has limits.
    """
    if not API_KEY:
        return None
    params = {"access_key": API_KEY, "flight_iata": flight_iata, "flight_date": flight_date, "limit": 5}
    try:
        r = requests.get(API_URL, params=params, timeout=12)
        r.raise_for_status()
        j = r.json()
        if not j.get("data"):
            return None
        # choose the best match (first)
        f = j["data"][0]
        # normalize some fields (safe-get)
        airline_name = f.get("airline", {}).get("name")
        flight_number = f.get("flight", {}).get("iata") or f.get("flight", {}).get("number") or flight_iata
        departure_airport = f.get("departure", {}).get("airport")
        departure_iata = f.get("departure", {}).get("iata")
        departure_scheduled = f.get("departure", {}).get("scheduled")
        departure_lat = f.get("departure", {}).get("latitude")
        departure_lng = f.get("departure", {}).get("longitude")
        arrival_airport = f.get("arrival", {}).get("airport")
        arrival_iata = f.get("arrival", {}).get("iata")
        arrival_scheduled = f.get("arrival", {}).get("scheduled")
        arrival_lat = f.get("arrival", {}).get("latitude")
        arrival_lng = f.get("arrival", {}).get("longitude")
        aircraft = f.get("aircraft", {}).get("registration") or f.get("aircraft", {}).get("icao") or f.get("aircraft", {}).get("iata")
        # compute distance if coords exist
        distance_km = None
        if departure_lat and departure_lng and arrival_lat and arrival_lng:
            try:
                distance_km = haversine_km(float(departure_lat), float(departure_lng), float(arrival_lat), float(arrival_lng))
            except Exception:
                distance_km = None

        # Build entry
        entry = {
            "saved_at": datetime.utcnow().isoformat() + "Z",
            "auto_fetched": True,
            "airline": airline_name,
            "flight_number": flight_number,
            "departure_airport": departure_airport,
            "departure_iata": departure_iata,
            "departure_scheduled": departure_scheduled,
            "departure_lat": departure_lat,
            "departure_lng": departure_lng,
            "arrival_airport": arrival_airport,
            "arrival_iata": arrival_iata,
            "arrival_scheduled": arrival_scheduled,
            "arrival_lat": arrival_lat,
            "arrival_lng": arrival_lng,
            "aircraft": aircraft,
            "distance_km": distance_km,
            "source": "aviationstack"
        }
        return entry
    except Exception as e:
        st.warning(f"AviationStack fetch error: {e}")
        return None

# -----------------------
# UI
# -----------------------
st.title("SkyNote — Flight Logger (Auto + Manual)")

# top controls
col1, col2 = st.columns([2,1])
with col2:
    auto_git = st.checkbox("Auto-commit to git (if available)", value=False)
    st.caption("Auto-commit requires git + configured credentials on the host.")

mode = st.radio("Mode", options=["Auto (enter flight + date)", "Manual (enter all fields)"], index=0)

flights = load_flights()

if mode.startswith("Auto"):
    st.markdown("Enter **flight IATA** (e.g., AI101, BA2491) and **departure date**. If you don't have an AviationStack API key, use Manual mode.")
    cols = st.columns([2,1,1])
    flight_iata = cols[0].text_input("Flight IATA (callsign)", value="", placeholder="e.g., AI102")
    dep_date = cols[1].date_input("Departure date", value=date.today())
    if cols[2].button("Fetch & Save"):
        if not flight_iata:
            st.error("Enter flight IATA code first (e.g., AI102).")
        else:
            ds = dep_date.isoformat()
            entry = fetch_via_aviationstack(flight_iata.strip(), ds)
            if entry:
                # add requested / canonical fields & persist
                flights.insert(0, entry)
                save_flights(flights, auto_git=auto_git, commit_msg=f"Add flight {entry.get('flight_number')} {ds}")
                st.success("Flight auto-fetched and saved to flights.json")
                st.json(entry)
            else:
                st.warning("Auto fetch failed or no data found. Please switch to Manual mode or check your AVIATIONSTACK_KEY in environment variables.")
                st.info("To get a free key: https://aviationstack.com (free tier).")

else:
    st.markdown("Manual entry — fill the fields and Save.")
    with st.form("manual_form"):
        airline = st.text_input("Airline")
        flight_number = st.text_input("Flight number")
        departure_airport = st.text_input("Departure (full name)")
        departure_iata = st.text_input("Departure IATA/ICAO")
        departure_scheduled = st.text_input("Departure scheduled (ISO or text)")
        arrival_airport = st.text_input("Arrival (full name)")
        arrival_iata = st.text_input("Arrival IATA/ICAO")
        arrival_scheduled = st.text_input("Arrival scheduled (ISO or text)")
        aircraft = st.text_input("Aircraft type / reg")
        distance_km = st.number_input("Distance (km, optional)", min_value=0.0, format="%.1f")
        submitted = st.form_submit_button("Save manual entry")
        if submitted:
            entry = {
                "saved_at": datetime.utcnow().isoformat() + "Z",
                "auto_fetched": False,
                "airline": airline or None,
                "flight_number": flight_number or None,
                "departure_airport": departure_airport or None,
                "departure_iata": departure_iata or None,
                "departure_scheduled": departure_scheduled or None,
                "departure_lat": None,
                "departure_lng": None,
                "arrival_airport": arrival_airport or None,
                "arrival_iata": arrival_iata or None,
                "arrival_scheduled": arrival_scheduled or None,
                "arrival_lat": None,
                "arrival_lng": None,
                "aircraft": aircraft or None,
                "distance_km": float(distance_km) if distance_km else None,
                "source": "manual"
            }
            flights.insert(0, entry)
            save_flights(flights, auto_git=auto_git, commit_msg=f"Add manual flight {entry.get('flight_number')}")
            st.success("Manual flight saved to flights.json")
            st.json(entry)

# -----------------------
# Saved flights table + import/export
# -----------------------
st.markdown("---")
st.subheader("Saved flights (flights.json)")

if flights:
    df = pd.json_normalize(flights)
    # user-friendly column order
    cols_show = ["saved_at", "auto_fetched", "airline", "flight_number", "departure_iata", "departure_airport",
                 "departure_scheduled", "arrival_iata", "arrival_airport", "arrival_scheduled", "aircraft", "distance_km", "source"]
    cols_show = [c for c in cols_show if c in df.columns]
    st.dataframe(df[cols_show].fillna(""))

    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        if st.button("Export JSON"):
            b = json.dumps(flights, indent=2).encode("utf-8")
            st.download_button("Download JSON", data=b, file_name="flights_export.json", mime="application/json")
    with c2:
        csv_buf = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv_buf, file_name="flights_export.csv", mime="text/csv")
    with c3:
        uploaded = st.file_uploader("Import JSON (append)", type=["json"])
        if uploaded:
            try:
                new_items = json.load(uploaded)
                if isinstance(new_items, dict):
                    # if user uploaded {"entries": [...]}
                    if "entries" in new_items and isinstance(new_items["entries"], list):
                        new_items = new_items["entries"]
                    else:
                        new_items = [new_items]
                elif not isinstance(new_items, list):
                    st.error("Unsupported JSON structure.")
                    new_items = []
                # append and save
                for it in reversed(new_items):
                    flights.insert(0, it)
                save_flights(flights, auto_git=auto_git, commit_msg="Import flights.json")
                st.success(f"Imported {len(new_items)} entries.")
            except Exception as e:
                st.error(f"Error importing: {e}")
else:
    st.info("No saved flights yet. Use Auto or Manual mode to add entries.")

# -----------------------
# Simple stats
# -----------------------
st.markdown("---")
st.subheader("Quick stats")
flights_count = len(flights)
total_km = sum((f.get("distance_km") or 0) for f in flights)
unique_airlines = len(set(f.get("airline") for f in flights if f.get("airline")))
st.metric("Saved flights", flights_count)
st.metric("Total distance (km)", round(total_km, 1))
st.metric("Unique airlines", unique_airlines)

st.caption("flights.json is stored in the repo root. Commit/push manually if running on a host without git credentials.")
