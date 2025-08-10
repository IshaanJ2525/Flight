# app.py
"""
SkyNote — Flighty-like personal flight logger (free-first) with world map routes.
Drop into your repo root. Writes flights.json in repo root.
Optional: set AVIATIONSTACK_KEY env var for autofill (free-tier).
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
import pydeck as pdk

# ----------------------
# CONFIG
# ----------------------
st.set_page_config(page_title="SkyNote • Flight Logger", layout="wide", initial_sidebar_state="expanded")
REPO_ROOT = Path(".").resolve()
DATA_FILE = REPO_ROOT / "flights.json"
AIRPORTS_FILE = REPO_ROOT / "airports.csv"   # optional full airports DB
API_KEY = os.environ.get("AVIATIONSTACK_KEY")  # optional; set for autofill
AVIATIONSTACK_URL = "http://api.aviationstack.com/v1/flights"
OPENSKY_URL = "https://opensky-network.org/api/states/all"

# ----------------------
# UTIL: files, git
# ----------------------
def ensure_files():
    if not DATA_FILE.exists():
        DATA_FILE.write_text(json.dumps([], indent=2))
    if not AIRPORTS_FILE.exists():
        # create a small sample airports CSV so distance works for major hubs
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

def save_flights(data, auto_git=False, commit_msg=None):
    DATA_FILE.write_text(json.dumps(data, indent=2, default=str))
    if auto_git:
        ok, msg = git_add_commit_push([str(DATA_FILE)], commit_msg or "Update flights.json")
        st.info(f"Auto-git: {msg}")

def git_add_commit_push(paths, commit_msg):
    try:
        subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception:
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

# ----------------------
# UTIL: distance / parse
# ----------------------
def haversine_km(lat1, lon1, lat2, lon2):
    if None in (lat1, lon1, lat2, lon2):
        return None
    # convert to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    km = 6371 * c
    return round(km, 1)

def parse_iso(dt):
    if dt is None:
        return None
    try:
        return pd.to_datetime(dt)
    except Exception:
        return None

# ----------------------
# AIRPORTS DB
# ----------------------
def load_airports():
    ensure_files()
    try:
        df = pd.read_csv(AIRPORTS_FILE, dtype={"iata":str,"icao":str})
        # normalize uppercase codes
        if "iata" in df.columns:
            df["iata"] = df["iata"].str.upper()
        if "icao" in df.columns:
            df["icao"] = df["icao"].str.upper()
        return df
    except Exception as e:
        st.warning(f"Could not load airports.csv: {e}")
        return pd.DataFrame()

airports_df = load_airports()

def lookup_airport_by_iata(iata):
    if airports_df.empty or not iata:
        return None
    row = airports_df[airports_df["iata"] == iata.upper()]
    if row.empty:
        # try ICAO match
        row = airports_df[airports_df["icao"] == iata.upper()]
    if row.empty:
        return None
    r = row.iloc[0].to_dict()
    return r

# ----------------------
# AUTOFILL via AviationStack
# ----------------------
def fetch_aviationstack(flight_iata: str, flight_date: str):
    """Return normalized flight dict or None. Uses AVIATIONSTACK_KEY env var if set."""
    if not API_KEY:
        return None
    params = {"access_key": API_KEY, "flight_iata": flight_iata, "flight_date": flight_date, "limit": 5}
    try:
        r = requests.get(AVIATIONSTACK_URL, params=params, timeout=12)
        r.raise_for_status()
        j = r.json()
        data = j.get("data") or []
        if not data:
            return None
        f = data[0]
        # gather fields
        dep = f.get("departure", {}) or {}
        arr = f.get("arrival", {}) or {}
        aircraft = f.get("aircraft") or {}
        airline = f.get("airline", {}) or {}
        dep_lat = dep.get("latitude")
        dep_lon = dep.get("longitude")
        arr_lat = arr.get("latitude")
        arr_lon = arr.get("longitude")
        distance_km = None
        if all(v is not None for v in [dep_lat, dep_lon, arr_lat, arr_lon]):
            try:
                distance_km = haversine_km(float(dep_lat), float(dep_lon), float(arr_lat), float(arr_lon))
            except Exception:
                distance_km = None
        entry = {
            "saved_at": datetime.utcnow().isoformat() + "Z",
            "auto": True,
            "airline": airline.get("name"),
            "flight_number": (f.get("flight") or {}).get("iata") or (f.get("flight") or {}).get("number"),
            "departure_airport": dep.get("airport"),
            "departure_iata": dep.get("iata"),
            "departure_scheduled": dep.get("scheduled"),
            "departure_lat": dep_lat,
            "departure_lng": dep_lon,
            "arrival_airport": arr.get("airport"),
            "arrival_iata": arr.get("iata"),
            "arrival_scheduled": arr.get("scheduled"),
            "arrival_lat": arr_lat,
            "arrival_lng": arr_lon,
            "aircraft": aircraft.get("registration") or aircraft.get("iata") or aircraft.get("icao"),
            "distance_km": distance_km,
            "source": "aviationstack"
        }
        return entry
    except Exception as e:
        st.warning(f"AviationStack fetch error: {e}")
        return None

# ----------------------
# OpenSky live positions (optional)
# ----------------------
@st.cache_data(ttl=20)
def fetch_opensky_positions(bbox=None):
    try:
        params = {}
        if bbox:
            params = {"lamin": bbox[0], "lamax": bbox[1], "lomin": bbox[2], "lomax": bbox[3]}
        r = requests.get(OPENSKY_URL, params=params, timeout=10)
        r.raise_for_status()
        j = r.json()
        states = j.get("states") or []
        cols = ["icao24","callsign","origin_country","time_position","last_contact","longitude","latitude","baro_altitude","on_ground","velocity","heading","vertical_rate","sensors","geo_altitude","squawk","spi","position_source"]
        df = pd.DataFrame(states, columns=cols)
        if not df.empty:
            df["callsign"] = df["callsign"].fillna("").str.strip()
            df["last_contact"] = pd.to_datetime(df["last_contact"], unit="s", utc=True, errors="coerce")
        return df
    except Exception as e:
        st.session_state["_opensky_error"] = str(e)
        return pd.DataFrame()

# ----------------------
# UI
# ----------------------
ensure_files()
st.title("SkyNote — Flight Logger & Routes Map")
st.markdown("Enter flight IATA + departure date (Auto Mode) or switch to Manual Mode. Saved entries are stored in `flights.json` in the repo root.")

# Sidebar
st.sidebar.header("Settings & Options")
auto_git = st.sidebar.checkbox("Auto-commit (git add/commit/push)", value=False)
st.sidebar.caption("Auto-commit works only if git is installed and repo has credentials on host.")
show_live = st.sidebar.checkbox("Show live aircraft positions (OpenSky)", value=False)
st.sidebar.markdown("---")
st.sidebar.markdown(f"Data file: `{DATA_FILE.name}`")
st.sidebar.markdown(f"Airports DB: `{AIRPORTS_FILE.name}` (replace with full dataset for better coverage)")

# Mode and inputs
mode = st.radio("Mode", ["Auto (flight + date)", "Manual (full)"], index=0)
col1, col2 = st.columns([3,1])

if mode == "Auto (flight + date)":
    with col1:
        flight_iata = st.text_input("Flight IATA (e.g., AI102, QR700)", value="")
        dep_date = st.date_input("Departure date", value=date.today(), min_value=date(2000,1,1), max_value=date(2100,12,31))
    with col2:
        if st.button("Auto Fill & Save"):
            if not flight_iata:
                st.error("Enter flight IATA first.")
            else:
                ds = dep_date.isoformat()
                entry = fetch_aviationstack(flight_iata.strip(), ds)
                if entry:
                    # fill missing coords via airports DB if available
                    if not entry.get("departure_lat") and entry.get("departure_iata"):
                        ap = lookup_airport_by_iata(entry["departure_iata"])
                        if ap:
                            entry["departure_lat"] = ap.get("lat"); entry["departure_lng"] = ap.get("lon")
                    if not entry.get("arrival_lat") and entry.get("arrival_iata"):
                        ap = lookup_airport_by_iata(entry["arrival_iata"])
                        if ap:
                            entry["arrival_lat"] = ap.get("lat"); entry["arrival_lng"] = ap.get("lon")
                    # if still no distance but have coords, compute
                    if not entry.get("distance_km") and entry.get("departure_lat") and entry.get("arrival_lat"):
                        try:
                            entry["distance_km"] = haversine_km(float(entry["departure_lat"]), float(entry["departure_lng"]),
                                                               float(entry["arrival_lat"]), float(entry["arrival_lng"]))
                        except Exception:
                            entry["distance_km"] = None
                    flights = load_flights()
                    flights.insert(0, entry)
                    save_flights(flights, auto_git=auto_git, commit_msg=f"Add flight {entry.get('flight_number')} {ds}")
                    st.success("Auto-fetched and saved to flights.json")
                    st.json(entry)
                else:
                    st.warning("Auto fetch failed or API key missing. Switch to Manual mode, or set AVIATIONSTACK_KEY in your environment.")

elif mode == "Manual (full)":
    with st.form("manual_form"):
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
        submitted = st.form_submit_button("Save manual entry")
        if submitted:
            entry = {
                "saved_at": datetime.utcnow().isoformat() + "Z",
                "auto": False,
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
                "source": "manual"
            }
            # try to fill coords from airports DB
            if entry.get("departure_iata"):
                ap = lookup_airport_by_iata(entry["departure_iata"])
                if ap:
                    entry["departure_lat"] = ap.get("lat"); entry["departure_lng"] = ap.get("lon")
            if entry.get("arrival_iata"):
                ap = lookup_airport_by_iata(entry["arrival_iata"])
                if ap:
                    entry["arrival_lat"] = ap.get("lat"); entry["arrival_lng"] = ap.get("lon")
            # compute distance if possible
            if not entry.get("distance_km") and entry.get("departure_lat") and entry.get("arrival_lat"):
                entry["distance_km"] = haversine_km(float(entry["departure_lat"]), float(entry["departure_lng"]),
                                                  float(entry["arrival_lat"]), float(entry["arrival_lng"]))
            flights = load_flights()
            flights.insert(0, entry)
            save_flights(flights, auto_git=auto_git, commit_msg=f"Add manual flight {entry.get('flight_number')}")
            st.success("Manual entry saved to flights.json")
            st.json(entry)

# ----------------------
# MAP: show saved routes + airports + optional live states
# ----------------------
st.markdown("---")
st.subheader("World Map — Saved routes")

flights = load_flights()
# create layers data
points = []
lines = []
for f in flights:
    dep_lat = f.get("departure_lat")
    dep_lon = f.get("departure_lng")
    arr_lat = f.get("arrival_lat")
    arr_lon = f.get("arrival_lng")
    # fallback: if airports DB present but no coords, try fill (non-destructive)
    if (not dep_lat or not dep_lon) and f.get("departure_iata"):
        ap = lookup_airport_by_iata(f["departure_iata"])
        if ap:
            dep_lat = dep_lat or ap.get("lat"); dep_lon = dep_lon or ap.get("lon")
    if (not arr_lat or not arr_lon) and f.get("arrival_iata"):
        ap = lookup_airport_by_iata(f["arrival_iata"])
        if ap:
            arr_lat = arr_lat or ap.get("lat"); arr_lon = arr_lon or ap.get("lon")
    # add points
    label = f"{f.get('airline') or ''} {f.get('flight_number') or ''}"
    if dep_lat and dep_lon:
        points.append({"lat": dep_lat, "lon": dep_lon, "name": f"Dep: {f.get('departure_iata') or ''}", "label": label})
    if arr_lat and arr_lon:
        points.append({"lat": arr_lat, "lon": arr_lon, "name": f"Arr: {f.get('arrival_iata') or ''}", "label": label})
    # add line
    if dep_lat and dep_lon and arr_lat and arr_lon:
        lines.append({"from_lon": dep_lon, "from_lat": dep_lat, "to_lon": arr_lon, "to_lat": arr_lat, "label": label})

# Optional live aircraft overlay
if show_live:
    with st.spinner("Fetching live aircraft (OpenSky)..."):
        states_df = fetch_opensky_positions()
    if not states_df.empty:
        for _, r in states_df.sample(min(50, len(states_df))).iterrows():
            if pd.notna(r.get("latitude")) and pd.notna(r.get("longitude")):
                points.append({"lat": float(r["latitude"]), "lon": float(r["longitude"]), "name": f"AC: {r['callsign']}", "label": r["callsign"]})
    else:
        st.info("OpenSky unavailable or rate-limited; live positions not shown.")

# Create pydeck layers
if points:
    points_df = pd.DataFrame(points)
    point_layer = pdk.Layer(
        "ScatterplotLayer",
        data=points_df,
        get_position='[lon, lat]',
        get_radius=50000,
        radius_scale=0.6,
        get_fill_color="[30,144,255, 180]",
        pickable=True
    )
else:
    point_layer = None

if lines:
    lines_df = pd.DataFrame(lines)
    arc_layer = pdk.Layer(
        "ArcLayer",
        data=lines_df,
        get_width=3,
        get_source_position='[from_lon, from_lat]',
        get_target_position='[to_lon, to_lat]',
        get_source_color='[255, 140, 0]',
        get_target_color='[0, 200, 255]'
    )
else:
    arc_layer = None

# initial view - center world or median
if points:
    mid_lat = float(points_df["lat"].median())
    mid_lon = float(points_df["lon"].median())
else:
    mid_lat, mid_lon = 20.0, 0.0

layers = [l for l in (arc_layer, point_layer) if l is not None]
if layers:
    view_state = pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=2, pitch=20)
    tooltip = {"html": "<b>{label}</b><br/>{name}", "style":{"color":"white"}}
    deck = pdk.Deck(layers=layers, initial_view_state=view_state, tooltip=tooltip)
    st.pydeck_chart(deck)
else:
    st.info("No routes or airports with coordinates available yet. Add flights or provide a full airports.csv for better coverage.")

# ----------------------
# Saved flights table + import/export
# ----------------------
st.markdown("---")
st.subheader("Saved flights & import/export")

if flights:
    df = pd.json_normalize(flights)
    # choose convenient columns
    show_cols = [c for c in ["saved_at","auto","airline","flight_number","departure_iata","departure_airport","departure_scheduled",
                             "arrival_iata","arrival_airport","arrival_scheduled","aircraft","distance_km","source"] if c in df.columns]
    st.dataframe(df[show_cols].fillna("").head(200), use_container_width=True)

    c1,c2,c3 = st.columns([1,1,1])
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
    st.info("No saved flights yet. Use Auto/Manual mode to add entries.")

# ----------------------
# Quick stats
# ----------------------
st.markdown("---")
st.subheader("Quick stats")
total = len(flights)
total_km = sum((f.get("distance_km") or 0) for f in flights)
airlines = len(set([f.get("airline") for f in flights if f.get("airline")]))
st.metric("Saved flights", total)
st.metric("Total distance (km)", round(total_km,1))
st.metric("Unique airlines", airlines)

st.caption("flights.json is created/updated in the repo root. Commit/push manually or enable Auto-commit (requires git & credentials). Replace airports.csv with a full airports dataset for best autofill/distance coverage.")
