import streamlit as st
import json
import plotly.express as px
import pandas as pd

st.set_page_config(page_title="Elyx Member Journey", layout="wide")
st.title("Elyx Life — Member Journey Visualization")

with open("data_full_8months.json", "r") as f:
    data = json.load(f)

tab1, tab2, tab3, tab4 = st.tabs(["📈 Metrics Timeline", "💬 WhatsApp Chats", "📋 Decision Tracker", "🧑 Persona & Ops"])

with tab1:
    st.subheader("Trends")
    if len(data["metrics"]) == 0:
        st.warning("No metrics found.")
    else:
        df = pd.DataFrame(data["metrics"])
        fig = px.line(df, x="date", y="value", color="metric", markers=True)
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("WhatsApp-style Threads")
    q = st.text_input("Search text")
    sender = st.selectbox("Sender", ["All"] + sorted({m["sender"] for m in data["messages"]}))
    pillar = st.selectbox("Pillar", ["All"] + sorted({m.get("pillar","") for m in data["messages"] if m.get("pillar")}))

    msgs = data["messages"]
    if sender != "All":
        msgs = [m for m in msgs if m["sender"] == sender]
    if pillar != "All":
        msgs = [m for m in msgs if m.get("pillar") == pillar]
    if q.strip():
        msgs = [m for m in msgs if q.lower() in m["message"].lower()]

    st.caption(f"{len(msgs)} messages")
    for m in msgs:
        st.markdown(f"**{m['date']} — {m['sender']} ({m['role']})**")
        st.write(m["message"])
        st.divider()

with tab3:
    st.subheader("Decision Register")
    for d in data["decisions"]:
        with st.expander(f"{d['date']} — {d['title']} ({d['by']})"):
            st.write("**Pillar:**", d["pillar"])
            st.write("**Rationale:**", d["rationale"])
            st.write("**Linked chats:**")
            for rid in d.get("related_message_ids", []):
                m = next((m for m in data["messages"] if m["id"] == rid), None)
                if m:
                    st.markdown(f"- {m['date']} **{m['sender']}**: {m['message']}")

with tab4:
    st.subheader("Persona")
    st.write(data["persona"])
    st.subheader("Internal Metrics")
    for k, v in data["internal_metrics"].items():
        st.write(f"- **{k}:** {v}")
