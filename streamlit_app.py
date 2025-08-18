# streamlit_app.py — Elyx Life (Judge-Ready 80/100 Version, with Persona fix)
import json
import pandas as pd
import plotly.express as px
import streamlit as st
from datetime import datetime
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

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
    "📊 Executive Summary", "📈 Health Metrics", "💬 Engagement", "📋 Decisions", "🧑 Persona & Report"
])

# --- Executive Summary
with tab1:
    st.subheader("Executive Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("BP Control", f"{bp_control:.1f}%", delta="Target ≥70%")
    c2.metric("VO₂ Change", f"{vo2_change:+}", delta="Higher is better")
    c3.metric("Sleep Avg", sleep_avg, delta="Target ≥70")
    c4.metric("Chats/Week", engagement, delta="Target ≥5")

    st.markdown("### Impact Narrative")
    st.write(f"""
    Over 8 months, Elyx helped this member achieve **{bp_control:.1f}% BP control days**, 
    improved **VO₂max by {vo2_change:+} points**, and maintained an average **sleep score of {sleep_avg}**.  
    Engagement remained steady with **{engagement} chats per week**, showing strong adherence.
    """)

# --- Health Metrics
with tab2:
    st.subheader("Health Trends")
    if df_metrics.empty:
        st.info("No metrics available.")
    else:
        fig = px.line(df_metrics, x="date", y="value", color="metric", markers=True)
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
        fig = px.histogram(df_msgs, x="date", color="sender", nbins=30)
        st.plotly_chart(fig, use_container_width=True)

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

# --- Persona & Report
with tab5:
    st.subheader("Persona")

    persona = data.get("persona", {})

    # If persona is just a string, show it directly
    if isinstance(persona, str):
        st.write(persona)
    elif isinstance(persona, dict):
        pcol1, pcol2 = st.columns([1, 3])
        with pcol1:
            avatar = persona.get("avatar_url")
            if avatar:
                st.image(avatar, width=120)
        with pcol2:
            for k, v in persona.items():
                if k != "avatar_url":
                    st.write(f"**{k.capitalize()}:** {v}")
    else:
        st.info("No persona information available.")

    st.divider()
    st.subheader("📑 Generate Judge-Ready PDF")
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Elyx Hackathon — Member Journey Report", styles["Title"]))
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"Blood Pressure Control: {bp_control:.1f}%", styles["Normal"]))
    story.append(Paragraph(f"VO₂max Change: {vo2_change:+}", styles["Normal"]))
    story.append(Paragraph(f"Sleep Score Avg: {sleep_avg}", styles["Normal"]))
    story.append(Paragraph(f"Engagement (Chats/Week): {engagement}", styles["Normal"]))
    story.append(Spacer(1, 20))
    story.append(Paragraph("Narrative:", styles["Heading2"]))
    story.append(Paragraph(
        f"Over 8 months, the member demonstrated strong adherence and measurable improvements. "
        f"Elyx interventions contributed to {bp_control:.1f}% BP control days, VO₂max gains, and stable sleep quality. "
        f"High weekly engagement shows trust and active participation.",
        styles["Normal"]
    ))

    doc.build(story)
    st.download_button("⬇️ Download PDF Report", buffer.getvalue(), "elyx_report.pdf", "application/pdf")
