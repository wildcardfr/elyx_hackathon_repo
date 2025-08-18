# streamlit_app.py — Elyx Life Hackathon (Judge-Ready 70–80/100 Version)

import json
import pandas as pd
import plotly.express as px
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Elyx Life Dashboard", layout="wide")
st.title("🌱 Elyx Life — Member Journey Dashboard")

# ---------- Data Load ----------
with open("data_full_8months.json", "r") as f:
    data = json.load(f)

df_metrics = pd.DataFrame(data.get("metrics", []))
df_msgs = pd.DataFrame(data.get("messages", []))
df_decisions = pd.DataFrame(data.get("decisions", []))

for df in [df_metrics, df_msgs, df_decisions]:
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

# ---------- Derived KPIs ----------
bp = df_metrics[df_metrics["metric"] == "Blood Pressure"]["value"]
vo2 = df_metrics[df_metrics["metric"] == "VO2max"]["value"]
sleep = df_metrics[df_metrics["metric"] == "Sleep Score"]["value"]

bp_control = (bp <= 130).sum() / len(bp) * 100 if len(bp) else 0
vo2_change = vo2.iloc[-1] - vo2.iloc[0] if len(vo2) >= 2 else 0
sleep_avg = round(sleep.mean(), 1) if len(sleep) else 0

# Engagement = chats/week
if not df_msgs.empty:
    df_msgs["week"] = df_msgs["date"].dt.isocalendar().week
    weekly = df_msgs.groupby("week").size()
    engagement = round(weekly.mean(), 1)
else:
    engagement = 0

# ---------- Tabs ----------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Executive Summary", "📈 Health Metrics", "💬 Engagement", "📋 Decisions", "🧑 Persona & Export"
])

# --- Executive Summary
with tab1:
    st.subheader("Executive Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("BP Control", f"{bp_control:.1f}%", delta="Target ≥70%")
    c2.metric("VO₂ Change", f"{vo2_change:+}", delta="Higher is better")
    c3.metric("Sleep Avg", sleep_avg, delta="Target ≥70")
    c4.metric("Chats/Week", engagement, delta="Target ≥5")

    st.markdown("### AI-style Insights")
    insights = []
    if bp_control < 60:
        insights.append("⚠️ Blood Pressure control is below healthy threshold — needs urgent intervention.")
    elif bp_control >= 70:
        insights.append("✅ Blood Pressure is well controlled most of the time.")

    if vo2_change > 0:
        insights.append(f"💪 VO₂max improved by {vo2_change} points, suggesting better cardio fitness.")
    else:
        insights.append("⚠️ VO₂max did not improve, indicating no fitness gain.")

    if sleep_avg < 70:
        insights.append("😴 Average sleep quality is low — recovery might be affected.")
    else:
        insights.append("✅ Sleep quality is good on average.")

    for i in insights:
        st.write(i)

# --- Health Metrics
with tab2:
    st.subheader("Health Trends")
    if df_metrics.empty:
        st.info("No metrics available.")
    else:
        fig = px.line(df_metrics, x="date", y="value", color="metric", markers=True)
        fig.update_layout(yaxis_title="Value", xaxis_title="Date")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Before vs After (First 2 months vs Last 2 months)")
        df_metrics["period"] = df_metrics["date"].apply(
            lambda d: "Before" if d < df_metrics["date"].min() + pd.Timedelta(days=60) else "After"
        )
        comp = df_metrics.groupby(["period","metric"])["value"].mean().reset_index()
        st.bar_chart(comp.pivot(index="metric", columns="period", values="value"))

# --- Engagement
with tab3:
    st.subheader("Message Activity")
    if df_msgs.empty:
        st.info("No messages available.")
    else:
        by_sender = df_msgs.groupby("sender").size().reset_index(name="messages")
        st.dataframe(by_sender, use_container_width=True)
        fig = px.histogram(df_msgs, x="date", color="sender", nbins=30, title="Messages Over Time")
        st.plotly_chart(fig, use_container_width=True)

        # Heatmap
        st.markdown("### Engagement Heatmap (Messages per Day)")
        df_msgs["day"] = df_msgs["date"].dt.date
        daily = df_msgs.groupby("day").size().reset_index(name="messages")
        daily["dow"] = pd.to_datetime(daily["day"]).dt.day_name()
        daily["week"] = pd.to_datetime(daily["day"]).dt.isocalendar().week

        pivot = daily.pivot("dow","week","messages").fillna(0)

        fig, ax = plt.subplots(figsize=(12,4))
        sns.heatmap(pivot, cmap="YlGnBu", linewidths=.5, ax=ax, cbar=False)
        st.pyplot(fig)

# --- Decisions
with tab4:
    st.subheader("Decision Register")
    if df_decisions.empty:
        st.info("No decisions found.")
    else:
        for _, d in df_decisions.iterrows():
            with st.expander(f"{d['date'].date()} — {d['title']} ({d['by']})"):
                st.write("**Pillar:**", d.get("pillar",""))
                st.write("**Rationale:**", d.get("rationale",""))
                ids = d.get("related_message_ids", [])
                for rid in ids:
                    m = next((m for m in data["messages"] if m["id"] == rid), None)
                    if m:
                        st.markdown(f"- {m['date']} **{m['sender']}**: {m['message']}")

# --- Persona & Export
with tab5:
    st.subheader("Persona Overview")

    persona = data.get("persona", {})

    if isinstance(persona, str):
        st.info(persona)
    elif isinstance(persona, dict):
        c1, c2, c3 = st.columns(3)
        c1.metric("Age", persona.get("age", "N/A"))
        c2.metric("Gender", persona.get("gender", "N/A"))
        c3.metric("Risk Level", persona.get("risk_level", "Medium"))
        st.write("**Lifestyle Notes:**", persona.get("lifestyle","Not provided"))
        avatar = persona.get("avatar_url")
        if avatar:
            st.image(avatar, width=120)
    else:
        st.info("No persona information available.")

    st.divider()
    st.subheader("📑 Export Judge Data")
    st.download_button(
        "⬇️ Download Metrics CSV",
        df_metrics.to_csv(index=False).encode("utf-8"),
        "elyx_metrics.csv",
        "text/csv",
    )
    st.download_button(
        "⬇️ Download Decisions CSV",
        df_decisions.to_csv(index=False).encode("utf-8"),
        "elyx_decisions.csv",
        "text/csv",
    )
    st.download_button(
        "⬇️ Download Messages CSV",
        df_msgs.to_csv(index=False).encode("utf-8"),
        "elyx_messages.csv",
        "text/csv",
    )
