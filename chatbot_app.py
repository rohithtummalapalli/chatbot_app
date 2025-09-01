
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

# ---------------------- Helpers ----------------------
def _to_ts(s):
    try:
        return pd.to_datetime(s, utc=True)
    except Exception:
        return pd.NaT

def _user_key(s):
    return s.get("user_id") or s.get("anonymous_id") or None

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

st.title("🤖 Chatbot Analytics Dashboard")


# Option A: local file bundled with the app
JSON_PATH = os.environ.get("JSON_PATH", "data/chats_new.json")

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
    st.header("Filters")

    # Date range
    if "created_at" in conversations_df and not conversations_df["created_at"].isna().all():
        min_date = conversations_df["created_at"].min().date()
        max_date = conversations_df["created_at"].max().date()
        dr = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    else:
        dr = None

    countries = sorted([c for c in conversations_df["country"].dropna().unique()])
    sel_countries = st.multiselect("Country", countries, default=countries)

    sources = sorted([c for c in conversations_df["source"].dropna().unique()])
    sel_sources = st.multiselect("Source", sources, default=sources)

    bots = sorted([c for c in conversations_df["chatbot_id"].dropna().unique()])
    sel_bots = st.multiselect("Chatbot", bots, default=bots)

    # user segment
    seg = st.selectbox("User segment", ["All", "New users", "Returning users"])

# Apply filters
df = conversations_df.copy()
if dr is not None and not df["created_at"].isna().all():
    start, end = dr
    mask = (df["created_at"].dt.date >= start) & (df["created_at"].dt.date <= end)
    df = df[mask]

if sel_countries:
    df = df[df["country"].isin(sel_countries)]
if sel_sources:
    df = df[df["source"].isin(sel_sources)]
if sel_bots:
    df = df[df["chatbot_id"].isin(sel_bots)]

if seg != "All" and "user_key" in df:
    counts = df["user_key"].dropna().value_counts()
    if seg == "New users":
        keep = set(counts[counts == 1].index)
    else:
        keep = set(counts[counts > 1].index)
    df = df[df["user_key"].isin(keep)]

# ---------------------- KPIs ----------------------
stats = kpis(df)
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
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Assistant Quality", "Users", "Data Tables"])

with tab1:
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


with tab2:
    asst = messages_df[messages_df["role"]=="assistant"]
    if not asst.empty and "assistant_score" in asst:
        fig = px.histogram(asst.dropna(subset=["assistant_score"]), x="assistant_score", nbins=30, title="Assistant Score Distribution")
        st.plotly_chart(fig, use_container_width=True)

        src_counts = asst["assistant_source"].fillna("unknown").value_counts().reset_index()
        src_counts.columns = ["assistant_source","count"]
        fig = px.pie(src_counts, values="count", names="assistant_source", title="Assistant Message Sources")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No assistant messages with scores found.")

with tab3:
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

with tab4:
    st.subheader("Conversations")
    st.dataframe(df.sort_values("created_at", ascending=False), use_container_width=True, height=400)
    df_to_csv_download(df, "conversations_filtered.csv")

    st.subheader("Messages")
    # filter messages for the selected conversations
    msg = messages_df[messages_df["conversation_id"].isin(df["conversation_id"])].copy()
    st.dataframe(msg, use_container_width=True, height=400)
    df_to_csv_download(msg, "messages_filtered.csv")

st.caption("Built with Streamlit • Drop in your JSON export and use the filters to explore everything.")
