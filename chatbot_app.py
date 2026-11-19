
import json
import io
from datetime import datetime
from functools import lru_cache
import os, urllib.request

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Chatbot Analytics", layout="wide")

st.markdown("""
<style>
:root { --ink:var(--text-color); --muted:var(--text-color); --line:rgba(127,127,127,.24); --surface:var(--secondary-background-color); --canvas:var(--background-color); }
.stApp { background:var(--background-color); color:var(--text-color); font-family:Inter,ui-sans-serif,system-ui,sans-serif; }
[data-testid="stHeader"] { background:transparent; }
[data-testid="stSidebar"] { background:var(--secondary-background-color); border-right:1px solid var(--line); }
[data-testid="stSidebar"] > div:first-child { padding-top:1.5rem; }
h1,h2,h3 { font-family:Inter,ui-sans-serif,system-ui,sans-serif !important; letter-spacing:-.035em; color:var(--text-color); }
h1 { font-size:2.1rem !important; font-weight:800 !important; }
[data-testid="stMetric"] { background:var(--surface); border:1px solid var(--line); border-radius:16px; padding:18px 20px; box-shadow:0 3px 12px rgba(0,0,0,.08); }
[data-testid="stMetricLabel"] { color:var(--muted); opacity:.75; font-weight:600; }
[data-testid="stMetricValue"] { color:var(--text-color); font-family:Inter,ui-sans-serif,system-ui,sans-serif; font-weight:700; }
div[data-testid="stPlotlyChart"], [data-testid="stDataFrame"] { background:var(--secondary-background-color); border:1px solid var(--line); border-radius:16px; padding:8px; }
div[data-testid="stRadio"] [role="radiogroup"] { display:flex; flex-wrap:wrap; align-items:center; gap:6px; width:fit-content; max-width:100%; padding:6px; background:var(--secondary-background-color); border:1px solid var(--line); border-radius:14px; }
div[data-testid="stRadio"] [role="radiogroup"] label { margin:0; padding:9px 14px; border-radius:10px; cursor:pointer; transition:background .15s ease,color .15s ease; }
div[data-testid="stRadio"] [role="radiogroup"] label > div:first-child { display:none; }
div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) { background:var(--primary-color); }
div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) * { color:#fff !important; }
div[data-testid="stRadio"] [role="radiogroup"] label:hover:not(:has(input:checked)) { background:rgba(127,127,127,.12); }
.hero { padding:26px 30px; margin:4px 0 22px; background:linear-gradient(120deg,#25285f 0%,#5356c9 55%,#7778ed 100%); border-radius:20px; color:#fff; box-shadow:0 14px 32px rgba(67,70,170,.18); }
.hero-kicker { text-transform:uppercase; letter-spacing:.14em; font-size:.72rem; font-weight:700; color:#d6d8ff; margin-bottom:9px; }
.hero-title { color:#fff; font:800 1.85rem Inter,ui-sans-serif,system-ui,sans-serif; letter-spacing:-.04em; }
.hero-copy { color:#e7e8ff; margin-top:7px; font-size:.98rem; }
.section-note { color:var(--muted); opacity:.78; font-size:.9rem; margin-top:-8px; margin-bottom:18px; }
div.stButton > button, div.stDownloadButton > button { border-radius:10px; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ---------------------- Helpers ----------------------
def _to_ts(s):
    try:
        return pd.to_datetime(s, utc=True)
    except Exception:
        return pd.NaT

def _user_key(s):
    return s.get("user_id") or s.get("anonymous_id") or None

def _has_submission(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"", "false", "no", "0", "null"}
    return bool(value)

@lru_cache(maxsize=8)
def load_data(json_bytes: bytes):
    raw = json.loads(json_bytes.decode("utf-8"))
    sessions = raw.get("conversations", [])
    conv_rows, msg_rows = [], []

    for s in sessions:
        conv_id = s.get("id")
        created_at = _to_ts(s.get("created_at"))
        last_at    = _to_ts(s.get("last_message_at"))
        duration   = (last_at - created_at).total_seconds() if (pd.notna(created_at) and pd.notna(last_at)) else np.nan

        conv_rows.append({
            "conversation_id": conv_id,
            "title": s.get("title"),
            "chatbot_id": s.get("chatbot_id"),
            "account_id": s.get("account_id"),
            "country": s.get("country"),
            "source": s.get("source"),
            "user_key": _user_key(s),
            "created_at": created_at,
            "last_message_at": last_at,
            "duration_seconds": None if np.isnan(duration) else int(duration),
            "min_score": s.get("min_score"),
            "sentiment": s.get("sentiment"),
            "form_submission": s.get("form_submission"),
            "external_conversation_id": s.get("external_conversation_id"),
            "topics": s.get("topics"),
        })

        for m in s.get("messages", []):
            msg_rows.append({
                "conversation_id": conv_id,
                "role": m.get("role"),
                "type": m.get("type"),
                "content": m.get("content"),
                "message_id": m.get("id"),
                "step_id": m.get("stepId"),
                "assistant_source": m.get("source") if m.get("role") == "assistant" else None,
                "assistant_score": m.get("score") if m.get("role") == "assistant" else None,
                "matched_sources": m.get("matchedSources"),
            })

    conversations_df = pd.DataFrame(conv_rows)
    messages_df      = pd.DataFrame(msg_rows)

    # counts per conversation
    if not messages_df.empty:
        counts = messages_df.groupby(["conversation_id", "role"]).size().unstack(fill_value=0)
        conversations_df = conversations_df.merge(counts, left_on="conversation_id", right_index=True, how="left")
        conversations_df.rename(columns={"assistant":"num_assistant_messages","user":"num_user_messages"}, inplace=True)
    else:
        conversations_df["num_assistant_messages"] = 0
        conversations_df["num_user_messages"] = 0

    conversations_df["num_messages"] = conversations_df["num_assistant_messages"].fillna(0) + conversations_df["num_user_messages"].fillna(0)

    # tidy types
    for c in ["num_messages","num_user_messages","num_assistant_messages","duration_seconds"]:
        conversations_df[c] = pd.to_numeric(conversations_df[c], errors="coerce")

    # add day/time features
    if not conversations_df["created_at"].isna().all():
        ca = conversations_df["created_at"].dt.tz_convert("UTC")
        conversations_df["date"] = ca.dt.date
        conversations_df["hour"] = ca.dt.hour
        conversations_df["weekday"] = ca.dt.day_name()

    return conversations_df, messages_df

def kpis(df):
    if df.empty:
        return dict(
            sessions=0, total_messages=0, avg_msgs=0, avg_dur=None, bounce=None, engage=None,
            unique_users=0, new_users=0, returning_users=0
        )
    sessions = len(df)
    total_messages = int(df["num_messages"].fillna(0).sum())
    avg_msgs = float(df["num_messages"].fillna(0).mean())
    avg_dur = float(df["duration_seconds"].dropna().mean()) if df["duration_seconds"].notna().any() else None

    # user stats
    if "user_key" in df and df["user_key"].notna().any():
        counts = df["user_key"].dropna().value_counts()
        unique_users = int(counts.size)
        new_users = int((counts == 1).sum())
        returning_users = int((counts > 1).sum())
    else:
        unique_users = new_users = returning_users = 0

    bounces = int((df["num_user_messages"].fillna(0) == 1).sum())
    engaged = int((df["num_user_messages"].fillna(0) >= 2).sum())
    bounce = bounces / sessions if sessions else None
    engage = engaged / sessions if sessions else None

    return dict(
        sessions=sessions, total_messages=total_messages, avg_msgs=round(avg_msgs,2),
        avg_dur=None if avg_dur is None else round(avg_dur,2),
        bounce=None if bounce is None else round(bounce,4),
        engage=None if engage is None else round(engage,4),
        unique_users=unique_users, new_users=new_users, returning_users=returning_users
    )

def df_to_csv_download(df, filename):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download " + filename, data=csv, file_name=filename, mime="text/csv")

st.markdown("""
<div class="hero">
  <div class="hero-kicker">Conversation intelligence</div>
  <div class="hero-title">Chatbot analytics</div>
  <div class="hero-copy">Explore engagement, assistant quality, and user behavior in one place.</div>
</div>
""", unsafe_allow_html=True)


# Option A: local file bundled with the app
JSON_PATH = os.environ.get("JSON_PATH", "data/chats_new(19-11-2026).json")

# Option B: remote file (e.g., S3 public URL). If set, this overrides local.
JSON_URL = os.environ.get("JSON_URL")

conversations_df = messages_df = None
try:
    if JSON_URL:
        with urllib.request.urlopen(JSON_URL) as resp:
            data_bytes = resp.read()
        conversations_df, messages_df = load_data(data_bytes)
    else:
        with open(JSON_PATH, "rb") as fh:
            data_bytes = fh.read()
        conversations_df, messages_df = load_data(data_bytes)
except Exception as e:
    st.error(f"Could not load the analytics JSON. Details: {e}")
    st.stop()



# Filters
with st.sidebar:
    st.markdown("### Explore data")
    st.caption("Choose a date window and narrow the conversations shown across the dashboard.")

    # Date range
    if "created_at" in conversations_df and not conversations_df["created_at"].isna().all():
        min_date = conversations_df["created_at"].min().date()
        max_date = conversations_df["created_at"].max().date()
        date_mode = st.selectbox("Date window", ["All dates", "Last 7 days", "Last 30 days", "Last 90 days", "Custom range"])
        if date_mode == "Custom range":
            dr = st.date_input("Choose dates", value=(min_date, max_date), min_value=min_date, max_value=max_date)
        elif date_mode == "All dates":
            dr = (min_date, max_date)
        else:
            days = {"Last 7 days": 7, "Last 30 days": 30, "Last 90 days": 90}[date_mode]
            dr = (max(min_date, max_date - pd.Timedelta(days=days - 1)), max_date)
    else:
        dr = None

    countries = sorted([c for c in conversations_df["country"].dropna().unique()])
    sel_countries = st.multiselect("Country", countries, default=countries)

    sources = sorted([c for c in conversations_df["source"].dropna().unique()])
    sel_sources = st.multiselect("Source", sources, default=sources)

    bots = sorted([c for c in conversations_df["chatbot_id"].dropna().unique()])
    sel_bots = st.multiselect("Chatbot", bots, default=bots)

    sentiments = sorted([str(value) for value in conversations_df["sentiment"].dropna().unique()])
    sel_sentiments = st.multiselect("Sentiment", sentiments, default=sentiments)

    form_status = st.selectbox("Form submission", ["All", "Submitted", "Not submitted"])

    max_user_messages = max(1, int(conversations_df["num_user_messages"].fillna(0).max()) if not conversations_df.empty else 1)
    user_message_range = st.slider("User messages per conversation", 0, max_user_messages, (0, max_user_messages))

    # user segment
    seg = st.selectbox("User segment", ["All", "New users", "Returning users"])

# The search widget is shown beside the main navigation below. Reading its
# stored value here keeps the search in sync with KPIs and every data view.
search_text = st.session_state.get("conversation_search", "")

# Apply filters
df = conversations_df.copy()
if dr is not None and not df["created_at"].isna().all():
    if isinstance(dr, (tuple, list)) and len(dr) == 2:
        start, end = dr
        mask = (df["created_at"].dt.date >= start) & (df["created_at"].dt.date <= end)
        df = df[mask]

if sel_countries:
    df = df[df["country"].isin(sel_countries)]
if sel_sources:
    df = df[df["source"].isin(sel_sources)]
if sel_bots:
    df = df[df["chatbot_id"].isin(sel_bots)]

if sel_sentiments:
    df = df[df["sentiment"].astype(str).isin(sel_sentiments)]

has_submission = df["form_submission"].map(_has_submission)
if form_status == "Submitted":
    df = df[has_submission]
elif form_status == "Not submitted":
    df = df[~has_submission]

df = df[df["num_user_messages"].fillna(0).between(user_message_range[0], user_message_range[1])]

if search_text.strip():
    matching_ids = messages_df.loc[
        messages_df["content"].fillna("").astype(str).str.contains(search_text.strip(), case=False, regex=False),
        "conversation_id",
    ].unique()
    df = df[df["conversation_id"].isin(matching_ids)]

if seg != "All" and "user_key" in df:
    counts = df["user_key"].dropna().value_counts()
    if seg == "New users":
        keep = set(counts[counts == 1].index)
    else:
        keep = set(counts[counts > 1].index)
    df = df[df["user_key"].isin(keep)]

# ---------------------- KPIs ----------------------
stats = kpis(df)
st.markdown(f'<div class="section-note">Showing <b>{len(df):,}</b> of {len(conversations_df):,} conversations</div>', unsafe_allow_html=True)
col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Sessions", stats["sessions"])
col2.metric("Total Messages", stats["total_messages"])
col3.metric("Avg Msgs/Session", stats["avg_msgs"])
col4.metric("Avg Duration (s)", stats["avg_dur"] if stats["avg_dur"] is not None else "-")
col5.metric("Bounce Rate", f"{stats['bounce']*100:.1f}%" if stats["bounce"] is not None else "-")
col6.metric("Engagement Rate", f"{stats['engage']*100:.1f}%" if stats["engage"] is not None else "-")

col7, col8, col9 = st.columns(3)
col7.metric("Unique Users", stats["unique_users"])
col8.metric("New Users", stats["new_users"])
col9.metric("Returning Users", stats["returning_users"])

st.markdown("---")

# ---------------------- Charts ----------------------
nav_col, search_col = st.columns([1.2, 0.8])
with nav_col:
    dashboard_section = st.radio(
        "Dashboard menu",
        ["Overview", "Assistant quality", "Users", "Conversation data"],
        horizontal=True,
        label_visibility="collapsed",
        key="dashboard_section",
    )
with search_col:
    st.text_input(
        "Search conversations",
        placeholder="Search conversations…",
        label_visibility="collapsed",
        key="conversation_search",
    )

if dashboard_section == "Overview":
    st.subheader("Usage overview")
    st.caption("Conversation volume and message patterns for the selected filters.")
    c1, c2 = st.columns(2)
    if "date" in df:
        by_day = df.groupby("date")["conversation_id"].nunique().reset_index(name="sessions")
        if not by_day.empty:
            fig = px.line(by_day, x="date", y="sessions", title="Sessions by Day")
            c1.plotly_chart(fig, use_container_width=True)

    if not df.empty:
        fig = px.histogram(df, x="num_messages", nbins=20, title="Messages per Session")
        c2.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    vc = (
        df["country"]
        .dropna()
        .value_counts()
        .reset_index(name="count")
        .rename(columns={"index": "country"})
    )
    if not vc.empty:
        fig = px.bar(vc, x="country", y="count", title="Sessions by Country")
        c3.plotly_chart(fig, use_container_width=True)


    vs = (
        df["source"]
        .dropna()
        .value_counts()
        .reset_index(name="count")
        .rename(columns={"index": "source"})
    )
    if not vs.empty:
        fig = px.bar(vs, x="source", y="count", title="Sessions by Source")
        c4.plotly_chart(fig, use_container_width=True)


elif dashboard_section == "Assistant quality":
    st.subheader("Assistant quality")
    st.caption("Response scores and the sources used by assistant messages.")
    asst = messages_df[
        messages_df["conversation_id"].isin(df["conversation_id"])
        & (messages_df["role"] == "assistant")
    ]
    if not asst.empty and "assistant_score" in asst:
        fig = px.histogram(asst.dropna(subset=["assistant_score"]), x="assistant_score", nbins=30, title="Assistant Score Distribution")
        st.plotly_chart(fig, use_container_width=True)

        src_counts = asst["assistant_source"].fillna("unknown").value_counts().reset_index()
        src_counts.columns = ["assistant_source","count"]
        fig = px.pie(src_counts, values="count", names="assistant_source", title="Assistant Message Sources")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No assistant messages with scores found.")

elif dashboard_section == "Users":
    st.subheader("User behavior")
    st.caption("Repeat usage and the times conversations are most active.")
    users_df = df.copy()
    if not users_df.empty and "user_key" in users_df:
        sessions_per_user = users_df["user_key"].value_counts().reset_index()
        sessions_per_user.columns = ["user_key","sessions"]
        fig = px.histogram(sessions_per_user, x="sessions", nbins=20, title="Sessions per User (distribution)")
        st.plotly_chart(fig, use_container_width=True)

        if "hour" in users_df:
            by_hour = users_df.groupby("hour")["conversation_id"].nunique().reset_index(name="sessions")
            fig = px.bar(by_hour, x="hour", y="sessions", title="Sessions by Hour (UTC)")
            st.plotly_chart(fig, use_container_width=True)

        if "weekday" in users_df:
            by_wd = users_df.groupby("weekday")["conversation_id"].nunique().reset_index(name="sessions")
            order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            by_wd["weekday"] = pd.Categorical(by_wd["weekday"], categories=order, ordered=True)
            by_wd = by_wd.sort_values("weekday")
            fig = px.bar(by_wd, x="weekday", y="sessions", title="Sessions by Weekday")
            st.plotly_chart(fig, use_container_width=True)

elif dashboard_section == "Conversation data":
    st.subheader("Conversations")
    st.caption("Filtered conversation records. Use the download buttons to export the current view.")
    st.dataframe(df.sort_values("created_at", ascending=False), use_container_width=True, height=400)
    df_to_csv_download(df, "conversations_filtered.csv")

    st.subheader("Messages")
    # filter messages for the selected conversations
    msg = messages_df[messages_df["conversation_id"].isin(df["conversation_id"])].copy()
    st.dataframe(msg, use_container_width=True, height=400)
    df_to_csv_download(msg, "messages_filtered.csv")

st.caption("Chatbot Analytics · Filtered views update across every tab and export.")
