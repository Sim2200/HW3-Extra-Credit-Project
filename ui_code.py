import ast
import base64
import os
from difflib import get_close_matches

import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# ==============================
# CONFIG
# ==============================
st.set_page_config(page_title="Spotify Explorer", page_icon="🎵", layout="wide")

CSV_PATH = "processed_spotify_data.csv"  # change if needed
CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID", "fcc0a4cf00044bf2a7b1bb28e79f5883")
CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "d76a5d47a16e44ea9d4a9dbd63d4ee4f")

FEATURE_COLS = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo"
]

# ==============================
# STYLING
# ==============================
st.markdown(
    """
    <style>
    .main {background-color: #121212; color: white;}
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    .hero-card {
        background: linear-gradient(135deg, #1e1e1e, #161616);
        border-radius: 22px;
        padding: 24px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.35);
        margin-bottom: 1.25rem;
    }
    .section-title {
        border-left: 6px solid #1db954;
        padding-left: 12px;
        font-size: 1.4rem;
        font-weight: 700;
        margin: 1rem 0;
    }
    .mini-card {
        background: #1a1a1a;
        border-radius: 18px;
        padding: 14px;
        min-height: 130px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.22);
        margin-bottom: 12px;
    }
    .meta {
        color: #cfcfcf;
        font-size: 0.92rem;
        margin-top: 4px;
    }
    .muted {
        color: #a9a9a9;
    }
    .pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        background: #1db95422;
        border: 1px solid #1db95455;
        color: #d7ffe5;
        font-size: 0.84rem;
        margin-right: 8px;
        margin-top: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================
# HELPERS
# ==============================

def ensure_list(x):
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    if isinstance(x, str):
        try:
            parsed = ast.literal_eval(x)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            return [x]
    return [x]


def safe_title(text):
    if pd.isna(text):
        return "Unknown"
    return str(text).title()


def normalize_text(x):
    return str(x).lower().strip()


def fuzzy_candidates(query, choices, n=8, cutoff=0.55):
    query = normalize_text(query)
    if query in choices:
        return [query]

    matches = get_close_matches(query, list(choices), n=n, cutoff=cutoff)

    contains = []
    for c in choices:
        if query in c or c in query:
            contains.append(c)

    merged = []
    seen = set()
    for item in matches + contains:
        if item not in seen:
            merged.append(item)
            seen.add(item)
    return merged[:n]


def spotify_enabled():
    return bool(CLIENT_ID and CLIENT_SECRET)


def get_token():
    if not spotify_enabled():
        return None
    auth_str = f"{CLIENT_ID}:{CLIENT_SECRET}"
    b64_auth = base64.b64encode(auth_str.encode()).decode()
    url = "https://accounts.spotify.com/api/token"
    headers = {"Authorization": f"Basic {b64_auth}"}
    data = {"grant_type": "client_credentials"}
    response = requests.post(url, headers=headers, data=data, timeout=30)
    response.raise_for_status()
    return response.json()["access_token"]


def spotify_fetch_by_id(item_id, fetch_type="artist", token=None):
    if not token or not item_id or str(item_id).lower() == "nan":
        return None
    url = f"https://api.spotify.com/v1/{fetch_type}s/{item_id}"
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    data = response.json()

    if fetch_type == "artist":
        return {
            "name": data.get("name"),
            "id": data.get("id"),
            "image": data["images"][0]["url"] if data.get("images") else None,
            "spotify_url": data.get("external_urls", {}).get("spotify"),
        }
    if fetch_type == "album":
        return {
            "name": data.get("name"),
            "id": data.get("id"),
            "image": data["images"][0]["url"] if data.get("images") else None,
            "spotify_url": data.get("external_urls", {}).get("spotify"),
        }
    if fetch_type == "track":
        return {
            "name": data.get("name"),
            "id": data.get("id"),
            "album": data.get("album", {}).get("name"),
            "image": (
                data.get("album", {}).get("images", [{}])[0].get("url")
                if data.get("album", {}).get("images") else None
            ),
            "spotify_url": data.get("external_urls", {}).get("spotify"),
        }
    return data


# ==============================
# DATA LOAD + INDEXES
# ==============================
@st.cache_data(show_spinner=False)
def load_and_prepare_data(csv_path):
    df = pd.read_csv(csv_path)

    df["artists"] = df["artists"].apply(ensure_list)
    if "artist_ids" in df.columns:
        df["artist_ids"] = df["artist_ids"].apply(ensure_list)
    else:
        df["artist_ids"] = [[] for _ in range(len(df))]

    for col in ["name", "album"]:
        df[col] = df[col].astype(str).str.lower().str.strip()

    # Store as tuples — lists are unhashable and break st.cache_data + pandas ops
    df["artists"]    = df["artists"].apply(lambda x: tuple(normalize_text(a) for a in x))
    df["artist_ids"] = df["artist_ids"].apply(lambda x: tuple(str(a).strip() for a in x))

    if "album_id" in df.columns:
        df["album_id"] = df["album_id"].astype(str).str.strip()
    else:
        df["album_id"] = ""

    if "id" in df.columns:
        df["track_id"] = df["id"].astype(str).str.strip()
    else:
        df["track_id"] = ""

    existing_feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    if existing_feature_cols:
        feature_frame = df.dropna(subset=existing_feature_cols).copy()
        scaler_avg = StandardScaler()
        scaled = scaler_avg.fit_transform(feature_frame[existing_feature_cols])
        feature_frame["audio_feature_avg"] = scaled.mean(axis=1)
        df = df.merge(
            feature_frame[["audio_feature_avg"]],
            left_index=True, right_index=True, how="left"
        )
    else:
        df["audio_feature_avg"] = np.nan

    return df


@st.cache_resource(show_spinner=False)
def build_indexes(df):
    artist_to_id = {}
    artist_to_songs = {}
    artist_to_albums = {}
    album_to_songs = {}
    album_to_artists = {}
    song_to_records = {}

    artist_names = set()
    album_names = set()
    song_names = set()

    for _, row in df.iterrows():
        song = row["name"]
        album = row["album"]
        year = row.get("year", "Unknown")
        album_id = row.get("album_id", "")
        track_id = row.get("track_id", "")
        artists    = list(row["artists"])
        artist_ids = list(row.get("artist_ids", ()))

        record = {
            "song": song,
            "album": album,
            "album_id": album_id,
            "track_id": track_id,
            "year": year,
            "artists": artists,
            "artist_ids": artist_ids,
        }

        song_to_records.setdefault(song, []).append(record)
        album_to_songs.setdefault(album, [])
        album_to_artists.setdefault(album, set())

        if song not in album_to_songs[album]:
            album_to_songs[album].append(song)
        for a in artists:
            album_to_artists[album].add(a)

        artist_names.update(artists)
        album_names.add(album)
        song_names.add(song)

        for i, artist in enumerate(artists):
            raw_id = artist_ids[i] if i < len(artist_ids) else None
            # Only store if it looks like a real Spotify ID (22 chars, alphanumeric)
            current_artist_id = None
            if raw_id and str(raw_id).strip() not in ("", "nan", "None"):
                cleaned = str(raw_id).strip()
                if len(cleaned) >= 10:
                    current_artist_id = cleaned
            if current_artist_id and artist not in artist_to_id:
                artist_to_id[artist] = current_artist_id

            artist_to_songs.setdefault(artist, {"solo": [], "main": [], "feature": []})
            artist_to_albums.setdefault(artist, set())
            artist_to_albums[artist].add(album)

            if len(artists) == 1:
                role = "solo"
            elif i == 0:
                role = "main"
            else:
                role = "feature"

            artist_to_songs[artist][role].append(record)

    artist_to_albums = {k: sorted(v) for k, v in artist_to_albums.items()}
    album_to_artists = {k: sorted(v) for k, v in album_to_artists.items()}

    return {
        "artist_to_id": artist_to_id,
        "artist_to_songs": artist_to_songs,
        "artist_to_albums": artist_to_albums,
        "album_to_songs": album_to_songs,
        "album_to_artists": album_to_artists,
        "song_to_records": song_to_records,
        "artist_names": sorted(artist_names),
        "album_names": sorted(album_names),
        "song_names": sorted(song_names),
    }


@st.cache_resource(show_spinner=False)
def prepare_similarity(df):
    df_clean = df.dropna(subset=FEATURE_COLS).copy()
    if df_clean.empty:
        return None, None, None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean[FEATURE_COLS])

    kmeans = KMeans(n_clusters=20, random_state=42, n_init=10)
    df_clean["cluster"] = kmeans.fit_predict(X_scaled)
    # Store as tuples — lists are unhashable and break Streamlit's cache hashing
    df_clean["_vec"] = [tuple(row) for row in X_scaled]

    return df_clean, scaler, kmeans


def get_similar_songs(song_name, df_clean, kmeans, limit=10, expand_clusters=1):
    name_norm = normalize_text(song_name)

    match = df_clean[df_clean["name"].str.lower().str.strip() == name_norm]
    if match.empty:
        match = df_clean[df_clean["name"].str.lower().str.contains(name_norm, na=False)]
    if match.empty:
        return pd.DataFrame()

    query_row = match.iloc[0]
    query_vec = np.array(query_row["_vec"]).reshape(1, -1)
    query_cluster = query_row["cluster"]

    if expand_clusters > 0:
        centroid_q = kmeans.cluster_centers_[query_cluster].reshape(1, -1)
        centroid_sim = cosine_similarity(centroid_q, kmeans.cluster_centers_)[0]
        top_clusters = np.argsort(centroid_sim)[::-1][:1 + expand_clusters]
        candidates = df_clean[df_clean["cluster"].isin(top_clusters)]
    else:
        candidates = df_clean[df_clean["cluster"] == query_cluster]

    candidates = candidates[
        candidates["name"].str.lower().str.strip() != name_norm
    ].copy()

    if candidates.empty:
        return pd.DataFrame()

    candidate_vecs = np.vstack(candidates["_vec"].values)
    sims = cosine_similarity(query_vec, candidate_vecs)[0]
    candidates["similarity"] = sims

    out = (
        candidates.sort_values("similarity", ascending=False)
        .head(limit)[["name", "artists", "album", "year", "cluster", "similarity", "album_id", "track_id", "artist_ids"]]
        .reset_index(drop=True)
    )
    # tuples → lists so downstream display code can iterate normally
    out["artists"]    = out["artists"].apply(list)
    out["artist_ids"] = out["artist_ids"].apply(list)
    out.index += 1
    return out


# ==============================
# RENDER HELPERS
# ==============================

def fetch_album_image(album_id, token):
    try:
        info = spotify_fetch_by_id(album_id, fetch_type="album", token=token)
        return info["image"] if info else None
    except Exception:
        return None


def fetch_artist_image(artist_id, token):
    """Try fetching by stored ID first; fall back to name search if needed."""
    if not token:
        return None
    try:
        if artist_id and str(artist_id).strip() not in ("", "nan", "None"):
            info = spotify_fetch_by_id(artist_id.strip(), fetch_type="artist", token=token)
            if info and info.get("image"):
                return info
    except Exception:
        pass
    return None


def search_artist_on_spotify(artist_name, token):
    """Search Spotify by artist name and return info dict."""
    if not token or not artist_name:
        return None
    try:
        url = "https://api.spotify.com/v1/search"
        headers = {"Authorization": f"Bearer {token}"}
        params = {"q": artist_name, "type": "artist", "limit": 1}
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        resp.raise_for_status()
        items = resp.json().get("artists", {}).get("items", [])
        if not items:
            return None
        a = items[0]
        return {
            "name":  a.get("name"),
            "id":    a.get("id"),
            "image": a["images"][0]["url"] if a.get("images") else None,
            "spotify_url": a.get("external_urls", {}).get("spotify"),
        }
    except Exception:
        return None


def render_song_card(song, album, year, cover=None, role=None, similarity=None, artists=None):
    img_html = (
        f"<img src='{cover}' style='width:100%;border-radius:10px;object-fit:cover;aspect-ratio:1'>"
        if cover else
        "<div style='background:#2a2a2a;border-radius:10px;aspect-ratio:1;display:flex;"
        "align-items:center;justify-content:center;color:#666;font-size:0.75rem'>No Image</div>"
    )
    pills = ""
    if role:
        pills += f"<span class='pill'>{role.title()}</span>"
    if similarity is not None:
        pills += f"<span class='pill'>{similarity:.3f} match</span>"
    artists_line = ""
    if artists:
        artists_line = f"<div class='meta'>{', '.join(safe_title(a) for a in artists[:2])}</div>"
    st.markdown(
        f"<div class='mini-card'>"
        f"{img_html}"
        f"<div style='margin-top:8px;font-weight:600;font-size:0.9rem;color:white;"
        f"white-space:nowrap;overflow:hidden;text-overflow:ellipsis'>{safe_title(song)}</div>"
        f"<div class='meta'>{safe_title(album)}</div>"
        f"{artists_line}"
        f"<div class='meta'>{year}</div>"
        f"{pills}"
        f"</div>",
        unsafe_allow_html=True,
    )


def render_artist_view(artist_key, indexes, token):
    artist_to_id = indexes["artist_to_id"]
    artist_to_songs = indexes["artist_to_songs"]
    artist_to_albums = indexes["artist_to_albums"]

    artist_id   = artist_to_id.get(artist_key)
    artist_info = fetch_artist_image(artist_id, token) if artist_id else None
    # Fall back: search Spotify by name if ID lookup gave no image
    if (not artist_info or not artist_info.get("image")) and token:
        artist_info = search_artist_on_spotify(artist_key, token) or artist_info

    display_name = safe_title(artist_info["name"] if artist_info else artist_key)
    num_albums = len(artist_to_albums.get(artist_key, []))
    num_songs = sum(len(v) for v in artist_to_songs.get(artist_key, {}).values())

    c1, c2 = st.columns([1, 3])
    with c1:
        if artist_info and artist_info.get("image"):
            st.image(artist_info["image"], width="stretch")
        else:
            st.markdown('<div class="mini-card muted">No Artist Image</div>', unsafe_allow_html=True)
    with c2:
        hero_html = (
            "<div class='hero-card'>"
            f"<h1 style='margin:0'>{display_name}</h1>"
            f"<div class='meta'>Albums: {num_albums} | Songs: {num_songs}</div>"
            "</div>"
        )
        st.markdown(hero_html, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Albums</div>", unsafe_allow_html=True)
    albums = artist_to_albums.get(artist_key, [])
    if not albums:
        st.info("No albums found.")
    else:
        # Fetch album covers then display in 4-column grid
        album_records = indexes["album_to_songs"]
        cols_per_row = 4
        rows = [albums[i:i+cols_per_row] for i in range(0, len(albums), cols_per_row)]
        for row_albums in rows:
            cols = st.columns(cols_per_row)
            for col, alb in zip(cols, row_albums):
                with col:
                    # grab album_id from first song in this album
                    first_songs = album_records.get(alb, [])
                    alb_id = None
                    for s in first_songs:
                        recs = indexes["song_to_records"].get(s, [])
                        for r in recs:
                            if r.get("album_id"):
                                alb_id = r["album_id"]
                                break
                        if alb_id:
                            break
                    cover = fetch_album_image(alb_id, token)
                    img_html = (
                        f"<img src='{cover}' style='width:100%;border-radius:10px;"
                        f"object-fit:cover;aspect-ratio:1'>"
                        if cover else
                        "<div style='background:#2a2a2a;border-radius:10px;aspect-ratio:1;"
                        "display:flex;align-items:center;justify-content:center;"
                        "color:#666;font-size:0.75rem'>No Image</div>"
                    )
                    st.markdown(
                        f"<div style='text-align:center;padding:6px'>"
                        f"{img_html}"
                        f"<div style='margin-top:6px;font-size:0.82rem;color:white;"
                        f"white-space:nowrap;overflow:hidden;text-overflow:ellipsis'>"
                        f"{safe_title(alb)}</div></div>",
                        unsafe_allow_html=True,
                    )

    for role, title in [
        ("solo",    "Solo Songs"),
        ("main",    "Main Artist Songs"),
        ("feature", "Featured Songs"),
    ]:
        st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)
        records = artist_to_songs.get(artist_key, {}).get(role, [])
        if not records:
            st.info(f"No {title.lower()} found.")
            continue
        cols_per_row = 4
        rows = [records[:40][i:i+cols_per_row] for i in range(0, min(len(records),40), cols_per_row)]
        for row_recs in rows:
            cols = st.columns(cols_per_row)
            for col, rec in zip(cols, row_recs):
                with col:
                    cover = fetch_album_image(rec.get("album_id"), token)
                    render_song_card(rec["song"], rec["album"], rec["year"], cover=cover, role=role)


def render_song_view(record, indexes, token, similarity_bundle, song_key):
    album_cover = fetch_album_image(record.get("album_id"), token)

    song_display = safe_title(record["song"])
    album_display = safe_title(record["album"])
    artists_display = ", ".join(safe_title(a) for a in record["artists"])

    c1, c2 = st.columns([1, 3])
    with c1:
        if album_cover:
            st.image(album_cover, width="stretch")
        else:
            st.markdown('<div class="mini-card muted">No Album Image</div>', unsafe_allow_html=True)
    with c2:
        hero_html = (
            "<div class='hero-card'>"
            f"<h1 style='margin:0'>{song_display}</h1>"
            f"<div class='meta'>Album: {album_display}</div>"
            f"<div class='meta'>Year: {record['year']}</div>"
            f"<div class='meta'>Artists: {artists_display}</div>"
            "</div>"
        )
        st.markdown(hero_html, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Other Songs In The Same Album</div>", unsafe_allow_html=True)
    same_album_songs = indexes["album_to_songs"].get(record["album"], [])
    for s in same_album_songs:
        if s != song_key:
            st.markdown(f"- {safe_title(s)}")

    st.markdown("<div class='section-title'>Other Albums By The Same Artists</div>", unsafe_allow_html=True)
    seen_albums = {record["album"]}
    for artist in record["artists"]:
        for album in indexes["artist_to_albums"].get(artist, []):
            if album not in seen_albums:
                st.markdown(f"- {safe_title(album)}")
                seen_albums.add(album)

    if similarity_bundle is not None and similarity_bundle[0] is not None:
        sim_key = f"sim_results_{song_key}"
        if st.button("Explore Similar Songs"):
            df_clean, _, kmeans = similarity_bundle
            st.session_state[sim_key] = get_similar_songs(song_key, df_clean, kmeans, limit=20)

        if sim_key in st.session_state:
            results = st.session_state[sim_key]
            st.markdown("<div class='section-title'>Similar Songs</div>", unsafe_allow_html=True)
            if results.empty:
                st.info("No similar songs found.")
            else:
                cols_per_row = 4
                result_list = list(results.iterrows())
                rows = [result_list[i:i+cols_per_row] for i in range(0, len(result_list), cols_per_row)]
                for row_items in rows:
                    cols = st.columns(cols_per_row)
                    for col, (_, row) in zip(cols, row_items):
                        with col:
                            cover = fetch_album_image(row.get("album_id"), token)
                            render_song_card(
                                row["name"], row["album"], row["year"],
                                cover=cover,
                                similarity=float(row["similarity"]),
                                artists=row["artists"],
                            )


def render_album_view(album_key, indexes, token):
    songs = indexes["album_to_songs"].get(album_key, [])
    artists = indexes["album_to_artists"].get(album_key, [])
    song_records = []
    for s in songs:
        for r in indexes["song_to_records"].get(s, []):
            if r["album"] == album_key:
                song_records.append(r)
                break

    album_id = song_records[0].get("album_id") if song_records else None
    album_info = None
    if album_id:
        try:
            album_info = spotify_fetch_by_id(album_id, fetch_type="album", token=token)
        except Exception:
            album_info = None

    album_display = safe_title(album_key)
    artists_display = ", ".join(safe_title(a) for a in artists)

    c1, c2 = st.columns([1, 3])
    with c1:
        if album_info and album_info.get("image"):
            st.image(album_info["image"], width="stretch")
        else:
            st.markdown('<div class="mini-card muted">No Album Image</div>', unsafe_allow_html=True)
    with c2:
        hero_html = (
            "<div class='hero-card'>"
            f"<h1 style='margin:0'>{album_display}</h1>"
            f"<div class='meta'>Artists: {artists_display}</div>"
            f"<div class='meta'>Songs: {len(songs)}</div>"
            "</div>"
        )
        st.markdown(hero_html, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Songs In This Album</div>", unsafe_allow_html=True)
    shared_cover = album_info.get("image") if album_info else None
    cols_per_row = 4
    rows = [song_records[i:i+cols_per_row] for i in range(0, len(song_records), cols_per_row)]
    for row_recs in rows:
        cols = st.columns(cols_per_row)
        for col, rec in zip(cols, row_recs):
            with col:
                render_song_card(rec["song"], rec["album"], rec["year"],
                                 cover=shared_cover, artists=rec["artists"])

    st.markdown("<div class='section-title'>Other Albums By These Artists</div>", unsafe_allow_html=True)
    seen = {album_key}
    other_albums = []
    for artist in artists:
        for other_album in indexes["artist_to_albums"].get(artist, []):
            if other_album not in seen:
                other_albums.append(other_album)
                seen.add(other_album)
    if not other_albums:
        st.info("No other albums found.")
    else:
        cols_per_row = 4
        rows = [other_albums[i:i+cols_per_row] for i in range(0, len(other_albums), cols_per_row)]
        for row_albs in rows:
            cols = st.columns(cols_per_row)
            for col, alb in zip(cols, row_albs):
                with col:
                    first_songs = indexes["album_to_songs"].get(alb, [])
                    alb_id = None
                    for s in first_songs:
                        for r in indexes["song_to_records"].get(s, []):
                            if r.get("album_id"):
                                alb_id = r["album_id"]
                                break
                        if alb_id:
                            break
                    cover = fetch_album_image(alb_id, token)
                    img_html = (
                        f"<img src='{cover}' style='width:100%;border-radius:10px;"
                        f"object-fit:cover;aspect-ratio:1'>"
                        if cover else
                        "<div style='background:#2a2a2a;border-radius:10px;aspect-ratio:1;"
                        "display:flex;align-items:center;justify-content:center;"
                        "color:#666;font-size:0.75rem'>No Image</div>"
                    )
                    st.markdown(
                        f"<div style='text-align:center;padding:6px'>"
                        f"{img_html}"
                        f"<div style='margin-top:6px;font-size:0.82rem;color:white;"
                        f"white-space:nowrap;overflow:hidden;text-overflow:ellipsis'>"
                        f"{safe_title(alb)}</div></div>",
                        unsafe_allow_html=True,
                    )


# ==============================
# APP
# ==============================
df = load_and_prepare_data(CSV_PATH)
indexes = build_indexes(df)
similarity_bundle = prepare_similarity(df)

token = None
if spotify_enabled():
    try:
        token = get_token()
    except Exception:
        token = None

st.title("🎵 Spotify Relationship Explorer")
st.caption("Search by artist, song, or album. Then choose what you meant.")

query = st.text_input("Enter an artist, song, or album")

col_a, col_b, col_c = st.columns(3)
with col_a:
    artist_btn = st.button("I entered an artist", width="stretch")
with col_b:
    song_btn = st.button("I entered a song", width="stretch")
with col_c:
    album_btn = st.button("I entered an album", width="stretch")

NOT_FOUND_MSGS = {
    "artist": (
        "😔 Sorry, we couldn't find **{query}** in our dataset.\n\n"
        "Our dataset covers artists who were active between **1997 – 2020**. "
        "Your favourite artist might not be included yet — try searching for "
        "someone from that era, or check the spelling!"
    ),
    "song": (
        "😔 Sorry, **'{query}'** doesn't appear to be in our dataset.\n\n"
        "We have songs released between **1997 and 2020**. "
        "Try searching for a track from that period, or double-check the song title."
    ),
    "album": (
        "😔 We couldn't find the album **'{query}'** in our dataset.\n\n"
        "Our collection spans albums released from **1997 to 2020**. "
        "The album might be outside that range — try another one from that era!"
    ),
}

# Persist entity_type across reruns so dropdown changes don't wipe it
if artist_btn:
    st.session_state["entity_type"] = "artist"
    st.session_state["search_query"] = query
    st.session_state.pop("confirmed_choice", None)
elif song_btn:
    st.session_state["entity_type"] = "song"
    st.session_state["search_query"] = query
    st.session_state.pop("confirmed_choice", None)
elif album_btn:
    st.session_state["entity_type"] = "album"
    st.session_state["search_query"] = query
    st.session_state.pop("confirmed_choice", None)

entity_type  = st.session_state.get("entity_type")
active_query = st.session_state.get("search_query", "")

if active_query and entity_type:
    if entity_type == "artist":
        candidates = fuzzy_candidates(active_query, indexes["artist_names"], n=8, cutoff=0.55)
        if not candidates:
            st.warning(NOT_FOUND_MSGS["artist"].format(query=active_query.title()))
        else:
            labels = [f"Artist — {safe_title(c)}" for c in candidates]
            # Prepend a blank prompt option so nothing renders until user picks
            options = ["— Select an artist —"] + labels
            sel = st.selectbox("Did you mean?", options, key="artist_select")
            if sel != "— Select an artist —":
                choice = candidates[labels.index(sel)]
                render_artist_view(choice, indexes, token)

    elif entity_type == "song":
        candidates = fuzzy_candidates(active_query, indexes["song_names"], n=8, cutoff=0.5)
        option_map = {}
        for song in candidates:
            for rec in indexes["song_to_records"].get(song, [])[:3]:
                label = (
                    f"Song — {safe_title(song)} | "
                    f"Artist — {', '.join(safe_title(a) for a in rec['artists'])} | "
                    f"Album — {safe_title(rec['album'])}"
                )
                option_map[label] = rec
        if not option_map:
            st.warning(NOT_FOUND_MSGS["song"].format(query=active_query.title()))
        else:
            options = ["— Select a song —"] + list(option_map.keys())
            sel = st.selectbox("Did you mean?", options, key="song_select")
            if sel != "— Select a song —":
                render_song_view(option_map[sel], indexes, token, similarity_bundle, option_map[sel]["song"])

    elif entity_type == "album":
        candidates = fuzzy_candidates(active_query, indexes["album_names"], n=8, cutoff=0.5)
        option_map = {}
        for album in candidates:
            artists = indexes["album_to_artists"].get(album, [])
            label = (
                f"Album — {safe_title(album)} | "
                f"Artist — {', '.join(safe_title(a) for a in artists[:3])}"
            )
            option_map[label] = album
        if not option_map:
            st.warning(NOT_FOUND_MSGS["album"].format(query=active_query.title()))
        else:
            options = ["— Select an album —"] + list(option_map.keys())
            sel = st.selectbox("Did you mean?", options, key="album_select")
            if sel != "— Select an album —":
                render_album_view(option_map[sel], indexes, token)

with st.sidebar:
    st.markdown("### Notes")
    st.write("- Artist search shows albums and songs grouped as solo, main, and feature.")
    st.write("- Song search shows album, year, artists, same-album songs, other albums, and similar songs.")
    st.write("- Album search shows songs in that album and other albums by the same artists.")
    if not spotify_enabled():
        st.warning("Spotify API credentials not found. Images will be unavailable.")