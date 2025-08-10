# app.py
"""
SkyNote — Flighty-like Streamlit flight tracker (Flight-number-first).
- Uses AviationStack when API key is provided.
- Manual mode fallback.
- Saves to flights.json in repo root.
- Shows world map routes using pydeck.
"""

import os
import json
import subprocess
from pathlib import Path
from datetime import datetime, date
from math import radians, sin, cos, asin, sqrt

import requests
import pandas as pd
import streamlit as st
import pydeck as pdk

# =========================
# CONFIG — put your key here
# =========================
API_KEY = "7816bd490cd6dfaf7d6023dca442dd8b"   # <-- Replace with your AviationStack key
AVIATIONSTACK_URL = "http://api.aviationstack.com/v1/flights"

REPO_ROOT = Path(".").resolve()
DATA_FILE = REPO_ROOT / "flights.json"
AIRPORTS_FILE = REPO_ROOT / "airports.csv"  # optional (small sample will be generated)

# =========================
# Utility functions
# =========================
def ensure_files():
    if not DATA_FILE.exists():
        DATA_FILE.write_text("[]")
    # create small airports sample so distance often works (you can replace with full DB)
    if not AIRPORTS_FILE.exists():
        sample = [
            {"iata":"DEL","icao":"VIDP","name":"Indira Gandhi Intl","city":"Delhi","country":"India","lat":28.556163,"lon":77.100281},
            {"iata":"BOM","icao":"VABB","name":"Chhatrapati Shivaji Intl","city":"Mumbai","country":"India","lat":19.089559,"lon":72.865614},
            {"iata":"LHR","icao":"EGLL","name":"Heathrow","city":"London","country":"United Kingdom","lat":51.470020,"lon":-0.454295},
            {"iata":"JFK","icao":"KJFK","name":"John F. Kennedy Intl","city":"New York","country":"USA","lat":40.641311,"lon":-73.778139},
            {"iata":"DXB","icao":"OMDB","name":"Dubai Intl","city":"Dubai","country":"UAE","lat":25.253174,"lon":55.365673},
            {"iata":"SIN","icao":"WSSS","name":"Changi","city":"Singapore","country":"Singapore","lat":1.364420,"lon":103.991531},
            {"iata":"DOH","icao":"OTHH","name":"Hamad Intl","city":"Doha","country":"Qatar","lat":25.261819,"lon":51.565061},
            {"iata":"CDG","icao":"LFPG","name":"Charles de Gaulle","city":"Paris","country":"France","lat":49.009690,"lon":2.547925},
        ]
        pd.DataFrame(sample).to_csv(AIRPORTS_FILE, index=False)

def load_flights():
    ensure_files()
    try:
        return json.loads(DATA_FILE.read_text())
    except Exception:
        return []

def save_flights(flights, auto_git=False, commit_msg=None):
    DATA_FILE.write_text(json.dumps(flights, indent=2, default=str))
    if auto_git:
        ok, msg = git_add_commit_push([str(DATA_FILE)], commit_msg or "Update flights.json")
        st.info(f"Auto-git: {msg}")

def git_add_commit_push(paths, commit_msg):
    try:
        subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError:
        return False, "Not a git repository"
    try:
        subprocess.run(["git", "add"] + paths, check=True)
        subprocess.run(["git", "commit", "-m", commit_msg], check=True)
    except subprocess.CalledProcessError as e:
        return False, f"Git commit failed: {e}"
    try:
        subprocess.run(["git", "push"], check=True)
        return True, "Committed and pushed"
    except subprocess.CalledProcessError as e:
        return True, f"Committed locally; push failed or needs credentials: {e}"

def haversine_km(lat1, lon1, lat2, lon2):
    if None in (lat1, lon1, lat2, lon2):
        return None
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return round(6371 * c, 1)

# =========================
# Airports DB helpers
# =========================
def load_airports():
    ensure_files()
    try:
        df = pd.read_csv(AIRPORTS_FILE, dtype={"iata":str,"icao":str})
        if "iata" in df.columns:
            df["iata"] = df["iata"].str.upper()
        if "icao" in df.columns:
            df["icao"] = df["icao"].str.upper()
        return df
    except Exception:
        return pd.DataFrame()

airports_df = load_airports()

def lookup_airport(iata_or_icao):
    if airports_df.empty or not iata_or_icao:
        return None
    code = iata_or_icao.strip().upper()
    row = airports_df[(airports_df["iata"] == code) | (airports_df["icao"] == code)]
    if row.empty:
        return None
    return row.iloc[0].to_dict()

# =========================
# AviationStack autofill
# =========================
def fetch_aviationstack(flight_iata: str, flight_date: str):
    if not API_KEY or API_KEY.startswith("YOUR_"):
        return None
    params = {"access_key": API_KEY, "flight_iata": flight_iata, "flight_date": flight_date, "limit": 5}
    try:
        r = requests.get(AVIATIONSTACK_URL, params=params, timeout=12)
        r.raise_for_status()
        j = r.json()
        data = j.get("data") or []
        if not data:
            return None
        # choose a record closest to the date if multiple: prefer exact date match
        chosen = None
        for d in data:
            dep = d.get("departure", {}) or {}
            sched = dep.get("scheduled") or dep.get("estimated") or None
            if sched and flight_date in sched:
                chosen = d
                break
        if not chosen:
            chosen = data[0]
        f = chosen
        dep = f.get("departure", {}) or {}
        arr = f.get("arrival", {}) or {}
        aircraft = f.get("aircraft") or {}
        airline = f.get("airline") or {}
        # coords from API if present
        dep_lat, dep_lon = dep.get("latitude"), dep.get("longitude")
        arr_lat, arr_lon = arr.get("latitude"), arr.get("longitude")
        # attempt airport lookup if coords missing
        if (dep_lat is None or dep_lon is None) and dep.get("iata"):
            ap = lookup_airport(dep.get("iata"))
            if ap:
                dep_lat, dep_lon = ap.get("lat"), ap.get("lon")
        if (arr_lat is None or arr_lon is None) and arr.get("iata"):
            ap = lookup_airport(arr.get("iata"))
            if ap:
                arr_lat, arr_lon = ap.get("lat"), ap.get("lon")
        # distance
        distance_km = None
        if all(v is not None for v in (dep_lat, dep_lon, arr_lat, arr_lon)):
            try:
                distance_km = haversine_km(float(dep_lat), float(dep_lon), float(arr_lat), float(arr_lon))
            except Exception:
                distance_km = None
        entry = {
            "saved_at": datetime.utcnow().isoformat() + "Z",
            "auto_fetched": True,
            "airline": airline.get("name"),
            "flight_number": (f.get("flight") or {}).get("iata") or (f.get("flight") or {}).get("number"),
            "departure_airport": dep.get("airport"),
            "departure_iata": dep.get("iata"),
            "departure_scheduled": dep.get("scheduled") or dep.get("estimated"),
            "departure_lat": dep_lat,
            "departure_lng": dep_lon,
            "arrival_airport": arr.get("airport"),
            "arrival_iata": arr.get("iata"),
            "arrival_scheduled": arr.get("scheduled") or arr.get("estimated"),
            "arrival_lat": arr_lat,
            "arrival_lng": arr_lon,
            "aircraft": aircraft.get("registration") or aircraft.get("icao") or aircraft.get("iata"),
            "distance_km": distance_km,
            "status": f.get("flight_status"),
            "source": "aviationstack"
        }
        return entry
    except Exception as e:
        st.warning(f"AviationStack error: {e}")
        return None

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="SkyNote • Flight Tracker", layout="wide")
st.title("SkyNote — Flight Tracker (Flight No. first)")

# Sidebar options
st.sidebar.header("Options")
auto_git = st.sidebar.checkbox("Auto-commit saved JSON to git", value=False)
st.sidebar.caption("Requires git & credentials on host to push.")
st.sidebar.markdown("---")
st.sidebar.markdown("If you don't want to hardcode the key, set `API_KEY` in this file or export env var and restart app.")

# Mode
mode = st.radio("Mode", ["Auto (flight + date)", "Manual entry"], index=0)

col_l, col_r = st.columns([2,1])

if mode == "Auto (flight + date)":
    with col_l:
        st.markdown("Enter flight IATA (e.g., AI102) and select departure date (from year 2000).")
        flight_iata = st.text_input("Flight IATA (callsign)", value="")
        dep_date = st.date_input("Departure date", value=date.today(), min_value=date(2000,1,1), max_value=date(2100,12,31))
        if st.button("Auto-Fetch & Save"):
            if not flight_iata.strip():
                st.error("Enter flight IATA first.")
            else:
                ds = dep_date.isoformat()
                entry = fetch_aviationstack(flight_iata.strip(), ds)
                if entry:
                    flights = load_flights()
                    flights.insert(0, entry)
                    save_flights(flights, auto_git=auto_git, commit_msg=f"Add flight {entry.get('flight_number')} {ds}")
                    st.success("Auto-fetched and saved to flights.json")
                    st.json(entry)
                else:
                    st.warning("Auto fetch failed or API key missing/limit reached. Switch to Manual mode or check API key.")
else:
    with col_l:
        st.markdown("Manual entry (fill everything you want). Coordinates optional.")
        with st.form("manual"):
            airline = st.text_input("Airline")
            flight_number = st.text_input("Flight number")
            dep_airport = st.text_input("Departure airport (name)")
            dep_iata = st.text_input("Departure IATA/ICAO")
            dep_sched = st.text_input("Departure scheduled (ISO/text)")
            arr_airport = st.text_input("Arrival airport (name)")
            arr_iata = st.text_input("Arrival IATA/ICAO")
            arr_sched = st.text_input("Arrival scheduled (ISO/text)")
            aircraft = st.text_input("Aircraft type / reg")
            dist_km = st.number_input("Distance (km) — optional", min_value=0.0, value=0.0, step=1.0)
            submit = st.form_submit_button("Save manual entry")
            if submit:
                entry = {
                    "saved_at": datetime.utcnow().isoformat() + "Z",
                    "auto_fetched": False,
                    "airline": airline or None,
                    "flight_number": flight_number or None,
                    "departure_airport": dep_airport or None,
                    "departure_iata": dep_iata or None,
                    "departure_scheduled": dep_sched or None,
                    "departure_lat": None,
                    "departure_lng": None,
                    "arrival_airport": arr_airport or None,
                    "arrival_iata": arr_iata or None,
                    "arrival_scheduled": arr_sched or None,
                    "arrival_lat": None,
                    "arrival_lng": None,
                    "aircraft": aircraft or None,
                    "distance_km": float(dist_km) if dist_km else None,
                    "status": None,
                    "source": "manual"
                }
                # try to auto-fill coords from airports db
                if entry.get("departure_iata"):
                    ap = lookup_airport(entry["departure_iata"])
                    if ap:
                        entry["departure_lat"] = ap.get("lat"); entry["departure_lng"] = ap.get("lon")
                if entry.get("arrival_iata"):
                    ap = lookup_airport(entry["arrival_iata"])
                    if ap:
                        entry["arrival_lat"] = ap.get("lat"); entry["arrival_lng"] = ap.get("lon")
                if not entry.get("distance_km") and entry.get("departure_lat") and entry.get("arrival_lat"):
                    entry["distance_km"] = haversine_km(float(entry["departure_lat"]), float(entry["departure_lng"]),
                                                        float(entry["arrival_lat"]), float(entry["arrival_lng"]))
                flights = load_flights()
                flights.insert(0, entry)
                save_flights(flights, auto_git=auto_git, commit_msg=f"Add manual flight {entry.get('flight_number')}")
                st.success("Manual entry saved to flights.json")
                st.json(entry)

# =========================
# Map & Saved flights area
# =========================
st.markdown("---")
st.subheader("World Map — saved routes & airports")

flights = load_flights()

# Build points and lines for pydeck
points = []
lines = []
for f in flights:
    dep_lat = f.get("departure_lat")
    dep_lon = f.get("departure_lng")
    arr_lat = f.get("arrival_lat")
    arr_lon = f.get("arrival_lng")
    # attempt to fill coords from airports DB if missing
    if (not dep_lat or not dep_lon) and f.get("departure_iata"):
        ap = lookup_airport(f["departure_iata"])
        if ap:
            dep_lat = dep_lat or ap.get("lat"); dep_lon = dep_lon or ap.get("lon")
    if (not arr_lat or not arr_lon) and f.get("arrival_iata"):
        ap = lookup_airport(f["arrival_iata"])
        if ap:
            arr_lat = arr_lat or ap.get("lat"); arr_lon = arr_lon or ap.get("lon")
    label = f"{f.get('airline') or ''} {f.get('flight_number') or ''}"
    if dep_lat and dep_lon:
        points.append({"lat": dep_lat, "lon": dep_lon, "name": f"Dep: {f.get('departure_iata') or ''}", "label": label})
    if arr_lat and arr_lon:
        points.append({"lat": arr_lat, "lon": arr_lon, "name": f"Arr: {f.get('arrival_iata') or ''}", "label": label})
    if dep_lat and dep_lon and arr_lat and arr_lon:
        lines.append({"from_lon": dep_lon, "from_lat": dep_lat, "to_lon": arr_lon, "to_lat": arr_lat, "label": label})

if points or lines:
    point_df = pd.DataFrame(points) if points else pd.DataFrame(columns=["lat","lon","name","label"])
    lines_df = pd.DataFrame(lines) if lines else pd.DataFrame(columns=["from_lon","from_lat","to_lon","to_lat","label"])
    # layers
    layers = []
    if not lines_df.empty:
        layers.append(pdk.Layer(
            "ArcLayer",
            data=lines_df,
            get_source_position='[from_lon, from_lat]',
            get_target_position='[to_lon, to_lat]',
            get_source_color='[255, 140, 0]',
            get_target_color='[0, 200, 255]',
            stroke_width=2
        ))
    if not point_df.empty:
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=point_df,
            get_position='[lon, lat]',
            get_fill_color='[10, 100, 200]',
            get_radius=30000,
            pickable=True
        ))
    mid_lat = float(point_df["lat"].median()) if not point_df.empty else 20.0
    mid_lon = float(point_df["lon"].median()) if not point_df.empty else 0.0
    view_state = pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=2, pitch=20)
    tooltip = {"html": "<b>{label}</b><br/>{name}", "style": {"color":"white"}}
    st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view_state, tooltip=tooltip))
else:
    st.info("No routes/airport coordinates available yet. Add entries or provide a fuller airports.csv for better coverage.")

# =========================
# Saved flights table + import/export
# =========================
st.markdown("---")
st.subheader("Saved flights")

if flights:
    df = pd.json_normalize(flights)
    show_cols = [c for c in ["saved_at","auto_fetched","airline","flight_number","departure_iata","departure_airport","departure_scheduled",
                             "arrival_iata","arrival_airport","arrival_scheduled","aircraft","distance_km","status","source"] if c in df.columns]
    st.dataframe(df[show_cols].fillna(""), use_container_width=True)
    c1,c2,c3 = st.columns(3)
    with c1:
        st.download_button("Export JSON", data=json.dumps(flights, indent=2).encode("utf-8"), file_name="flights_export.json", mime="application/json")
    with c2:
        csv_buf = df.to_csv(index=False).encode("utf-8")
        st.download_button("Export CSV", data=csv_buf, file_name="flights_export.csv", mime="text/csv")
    with c3:
        uploaded = st.file_uploader("Import JSON (append)", type=["json"])
        if uploaded:
            try:
                new = json.load(uploaded)
                # accept list or {"entries":[...]} object
                if isinstance(new, dict) and "entries" in new:
                    new = new["entries"]
                if not isinstance(new, list):
                    st.error("Unsupported JSON format.")
                else:
                    for item in reversed(new):
                        flights.insert(0, item)
                    save_flights(flights, auto_git=auto_git, commit_msg="Import flights")
                    st.success(f"Imported {len(new)} flights.")
            except Exception as e:
                st.error(f"Import failed: {e}")
else:
    st.info("No saved flights yet. Add one using Auto or Manual mode.")

# =========================
# Quick stats
# =========================
st.markdown("---")
st.subheader("Quick stats")
total = len(flights)
total_km = sum((f.get("distance_km") or 0) for f in flights)
unique_airlines = len(set(f.get("airline") for f in flights if f.get("airline")))
st.metric("Saved flights", total)
st.metric("Total distance (km)", round(total_km,1))
st.metric("Unique airlines", unique_airlines)

st.caption("flights.json is stored in the repo root. Auto-commit will try to git add/commit/push (requires git & credentials). Replace airports.csv with a complete airports DB for best autofill and distances.")
