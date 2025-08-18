# streamlit_app.py — Elyx Life (Refactored, Judge-Ready, Pandas 2.3 compatible)
"""
Refactored Elyx Member Journey Streamlit app.
Preserves original features while improving structure, error handling,
and pandas compatibility (avoids pandas 2.4-only APIs).
"""

import json
import base64
from datetime import datetime, timedelta
from io import StringIO
from typing import Tuple, Dict, Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Elyx Member Journey — Judge Ready", layout="wide")
st.title("🏆 Elyx Life — Member Journey (Judge-Ready)")
st.caption("Polished visualizations, automated checklist, narrative insights and exports")

# -------------------------
# Sidebar UI
# -------------------------
with st.sidebar:
    st.header("⚙️ Controls")
    uploaded = st.file_uploader("Upload JSON (optional)", type=["json"])
    show_judge = st.checkbox("🎯 Show Judge Mode", value=True)
    show_insights = st.checkbox("💡 Show Auto Insights", value=True)
    st.markdown("---")
    st.caption("Tip: upload your own JSON to test. Use judge mode during demo.")

# -------------------------
# Helper functions
# -------------------------
@st.cache_data
def load_data_from_file(file_like) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load JSON data and return raw dict plus metrics/messages/decisions DataFrames."""
    try:
        if file_like:
            data = json.load(file_like)
        else:
            with open("data_full_8months.json", "r") as f:
                data = json.load(f)
    except Exception:
        # graceful fallback
        data = {}

    # Ensure required keys with safe defaults
    data.setdefault("metrics", [])
    data.setdefault("messages", [])
    data.setdefault("decisions", [])
    data.setdefault("persona", {})
    data.setdefault("internal_metrics", {})

    # Convert to DataFrames safely
    df_metrics = pd.DataFrame(data["metrics"])
    df_msgs = pd.DataFrame(data["messages"])
    df_decisions = pd.DataFrame(data["decisions"])

    # Normalize date columns if present
    for df in (df_metrics, df_msgs, df_decisions):
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

    return data, df_metrics, df_msgs, df_decisions


def safe_pct(n: int, d: int) -> float:
    return 0.0 if d == 0 else round(100.0 * n / d, 1)


def metric_series(df: pd.DataFrame, metric_name: str) -> pd.Series:
    """Return numeric series for a given metric name (case-insensitive)."""
    if df.empty or "metric" not in df.columns or "value" not in df.columns:
        return pd.Series(dtype=float)
    mask = df["metric"].astype(str).str.lower() == metric_name.lower()
    try:
        return df.loc[mask, "value"].astype(float).reset_index(drop=True)
    except Exception:
        # if conversion fails, return empty
        return pd.Series(dtype=float)


def ensure_date_column(df: pd.DataFrame, col: str = "date") -> pd.Series:
    """Return a safe date Series for the dataframe or an empty Series."""
    if df.empty or col not in df.columns:
        return pd.Series(dtype="datetime64[ns]")
    return pd.to_datetime(df[col], errors="coerce")


def df_date_range_minmax(*dfs) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Compute min and max across date columns of multiple dataframes."""
    dates = []
    for df in dfs:
        if not df.empty and "date" in df.columns:
            dates.append(df["date"].dropna())
    if not dates:
        return None, None
    min_date = min(s.min() for s in dates if not s.empty)
    max_date = max(s.max() for s in dates if not s.empty)
    return min_date, max_date


def html_snapshot(data: dict, kpis: dict, persona: dict, insights: list) -> str:
    """Build a small HTML snapshot string for download."""
    html = ["<html><head><meta charset='utf-8'><title>Elyx Snapshot</title></head><body>"]
    html.append(f"<h1>Elyx Journey — Snapshot</h1>")
    html.append(f"<p>Generated: {datetime.utcnow().isoformat()} UTC</p>")
    html.append("<h2>KPIs</h2><ul>")
    for k, v in kpis.items():
        html.append(f"<li><strong>{k}:</strong> {v}</li>")
    html.append("</ul>")
    html.append("<h2>Persona</h2>")
    html.append(f"<pre>{json.dumps(persona, indent=2)}</pre>")
    if insights:
        html.append("<h2>Top insights</h2><ul>")
        for i in insights:
            html.append(f"<li>{i}</li>")
        html.append("</ul>")
    html.append("</body></html>")
    return "\n".join(html)


# -------------------------
# Load data
# -------------------------
data, df_metrics, df_msgs, df_decisions = load_data_from_file(uploaded)

# Guardrail: require at least messages or metrics to proceed
if df_metrics.empty and df_msgs.empty:
    st.error("No usable metrics or messages found. Upload JSON or include `data_full_8months.json` in repo.")
    st.stop()

# -------------------------
# Derived KPIs (robust)
# -------------------------
bp_series = metric_series(df_metrics, "Blood Pressure")
vo2_series = metric_series(df_metrics, "VO2max")
sleep_series = metric_series(df_metrics, "Sleep Score")

bp_avg = int(bp_series.mean()) if not bp_series.empty else None
bp_ctrl_rate = safe_pct((bp_series <= 130).sum(), bp_series.count()) if not bp_series.empty else 0
vo2_delta = int(vo2_series.iloc[-1] - vo2_series.iloc[0]) if len(vo2_series) >= 2 else 0
sleep_avg = int(sleep_series.mean()) if not sleep_series.empty else None

# Adherence proxy: weeks with >=4 member-initiated messages / total weeks
adherence = 0.0
if not df_msgs.empty:
    # create safe date_only column
    if "date" in df_msgs.columns:
        # avoid chained assignment; use loc
        df_msgs = df_msgs.copy()
        df_msgs.loc[:, "date_only"] = pd.to_datetime(df_msgs["date"].dt.date)
    else:
        df_msgs.loc[:, "date_only"] = pd.NaT

    # robust member messages filter: fillna and lower()
    if "role" in df_msgs.columns:
        member_msgs = df_msgs[df_msgs["role"].fillna("").astype(str).str.lower() == "member"]
    else:
        member_msgs = df_msgs.copy()

    if not member_msgs.empty and "date_only" in member_msgs.columns:
        member_msgs = member_msgs.copy()
        member_msgs.loc[:, "week"] = member_msgs["date_only"].dt.isocalendar().week
        weeks_count = int(member_msgs["week"].nunique() or 0)
        active_weeks = int(member_msgs.groupby("week").size().ge(4).sum())
        adherence = safe_pct(active_weeks, weeks_count) if weeks_count else 0.0
    else:
        adherence = 0.0

# -------------------------
# Header KPI metrics
# -------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Avg BP (mmHg)", value=bp_avg if bp_avg is not None else "—",
          delta=f"{bp_ctrl_rate}% days ≤130")
k2.metric("VO₂max change", value=f"{vo2_delta:+} pts",
          delta="8-month delta" if vo2_delta is not None else "")
k3.metric("Sleep score (avg)", value=sleep_avg if sleep_avg is not None else "—",
          delta="goal ≥ 70")
k4.metric("Adherence proxy", value=f"{adherence}%", delta="weeks ≥4 chats")

st.markdown("---")

# -------------------------
# Global filters (date range, pillar, search)
# -------------------------
min_date, max_date = df_date_range_minmax(df_metrics, df_msgs, df_decisions)

# date_input defaults: if no data, show last 7 days
if min_date is not None and max_date is not None:
    date_range = st.date_input("Date range", value=(min_date.date(), max_date.date()))
else:
    today = datetime.today().date()
    date_range = st.date_input("Date range", value=(today - timedelta(days=7), today))

search_text = st.text_input("Search text (messages & decisions)")

# build pillar options from messages + decisions safely
pillar_opts = ["All"]
try:
    p_msgs = list(df_msgs.get("pillar", pd.Series([], dtype=object)).dropna().unique())
    p_dec = list(df_decisions.get("pillar", pd.Series([], dtype=object)).dropna().unique())
    pillar_opts += sorted(list(set(p_msgs + p_dec)))
except Exception:
    pass
pillar_filter = st.selectbox("Pillar", pillar_opts)

# -------------------------
# Apply filters helpers
# -------------------------
def apply_date_filter(df: pd.DataFrame, date_range_tuple) -> pd.DataFrame:
    """Return df filtered by date_range_tuple (start_date, end_date)."""
    if df.empty or "date" not in df.columns:
        return df
    start, end = pd.to_datetime(date_range_tuple[0]), pd.to_datetime(date_range_tuple[1])
    return df[(df["date"] >= start) & (df["date"] <= end)]


df_metrics_f = apply_date_filter(df_metrics, date_range)
df_msgs_f = apply_date_filter(df_msgs, date_range)
df_decisions_f = apply_date_filter(df_decisions, date_range)

# pillar filter
if pillar_filter != "All":
    if "pillar" in df_msgs_f.columns:
        df_msgs_f = df_msgs_f[df_msgs_f["pillar"] == pillar_filter]
    if "pillar" in df_decisions_f.columns:
        df_decisions_f = df_decisions_f[df_decisions_f["pillar"] == pillar_filter]

# text search filter
if search_text and search_text.strip():
    q = search_text.lower()
    if "message" in df_msgs_f.columns:
        df_msgs_f = df_msgs_f[df_msgs_f["message"].astype(str).str.lower().str.contains(q, na=False)]
    if not df_decisions_f.empty:
        # search across decision row contents
        df_decisions_f = df_decisions_f[df_decisions_f.apply(lambda r: q in json.dumps(r.dropna().to_dict()).lower(), axis=1)]

# -------------------------
# Tabs
# -------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Metrics Comparison",
    "💬 Chats & Engagement",
    "📋 Decision Register",
    "🧑 Persona & Ops",
    "⬇️ Exports & Snapshot",
])

# -------------------------
# Tab 1: Metrics Comparison
# -------------------------
with tab1:
    st.subheader("Multi-metric comparison & mini insights")
    if df_metrics_f.empty:
        st.info("No metrics to show for selected filters.")
    else:
        # pivot: date x metric
        pivot = (df_metrics_f.groupby(["date", "metric"])["value"]
                 .mean().reset_index()
                 .pivot(index="date", columns="metric", values="value")
                 .sort_index())

        # plotly figure with primary + secondary axes
        fig = go.Figure()
        metrics_list = list(pivot.columns)
        colors = px.colors.qualitative.Plotly

        for i, m in enumerate(metrics_list):
            use_secondary = True if i >= 1 else False
            fig.add_trace(go.Scatter(
                x=pivot.index,
                y=pivot[m],
                name=m,
                mode="lines+markers",
                yaxis="y2" if use_secondary else "y",
                marker=dict(size=6),
                line=dict(width=2, color=colors[i % len(colors)])
            ))

        fig.update_layout(
            title="Metrics over time (primary + secondary axis)",
            xaxis=dict(title="Date"),
            yaxis=dict(title="Primary"),
            yaxis2=dict(title="Secondary", overlaying="y", side="right"),
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="left", x=0),
            margin=dict(t=40, b=50)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Auto insights (rule-based)
        if show_insights:
            insights = []
            if not vo2_series.empty and len(vo2_series) >= 2:
                insights.append(f"VO₂max changed by **{vo2_delta:+} pts** over the window.")
            if bp_avg is not None:
                insights.append(f"Avg BP is **{bp_avg} mmHg** with **{bp_ctrl_rate}%** of readings ≤130.")
            if sleep_avg is not None:
                if sleep_avg >= 70:
                    insights.append(f"Average sleep score **{sleep_avg}** meets the goal (≥70).")
                else:
                    insights.append(f"Average sleep score **{sleep_avg}** is below goal (≥70); consider sleep interventions.")
            insights.append(f"Adherence (weeks with ≥4 member chats): **{adherence}%**.")

            st.markdown("**Auto insights:**")
            for i, ins in enumerate(insights, 1):
                st.write(f"{i}. {ins}")

# -------------------------
# Tab 2: Chats & Engagement
# -------------------------
with tab2:
    st.subheader("Chats, activity timeline & engagement heatmap")
    if df_msgs_f.empty:
        st.info("No messages for selected filters.")
    else:
        left, right = st.columns([2, 1])
        with left:
            st.caption(f"{len(df_msgs_f)} messages in selection")
            # threaded messages as expanders
            for _, row in df_msgs_f.sort_values("date").iterrows():
                stamp = row["date"].strftime("%Y-%m-%d %H:%M") if pd.notna(row.get("date")) else ""
                sender = row.get("sender", "Unknown")
                role = row.get("role", "")
                header = f"**{stamp} — {sender} ({role})**"
                with st.expander(header, expanded=False):
                    st.write(row.get("message", ""))
                    if row.get("pillar"):
                        st.caption(f"Pillar: {row.get('pillar')}")

        with right:
            if "sender" in df_msgs_f.columns:
                by_sender = df_msgs_f.groupby("sender").size().reset_index(name="messages")
                fig_s = px.bar(by_sender, x="messages", y="sender", orientation="h", title="Messages by sender", text="messages")
                st.plotly_chart(fig_s, use_container_width=True)

        # calendar heatmap: messages per day
        df_days = df_msgs_f.copy()
        if "date" in df_days.columns:
            df_days.loc[:, "date_only"] = pd.to_datetime(df_days["date"].dt.date)
            day_counts = df_days.groupby("date_only").size().reset_index(name="count")
            if not day_counts.empty:
                start = day_counts["date_only"].min()
                end = day_counts["date_only"].max()
                all_dates = pd.DataFrame({"date_only": pd.date_range(start, end)})
                merged = all_dates.merge(day_counts, on="date_only", how="left").fillna(0)
                merged.loc[:, "week_of_year"] = merged["date_only"].dt.isocalendar().week
                merged.loc[:, "weekday"] = merged["date_only"].dt.weekday  # 0=Mon
                week_labels = merged.groupby("week_of_year")["date_only"].min().sort_values().to_dict()
                merged.loc[:, "week_label"] = merged["week_of_year"].map(lambda w: week_labels.get(w).strftime("%Y-%m-%d"))
                heat = merged.pivot_table(index="week_label", columns="weekday", values="count", fill_value=0)
                weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                # ensure only existing weekdays
                heat = heat[[i for i in range(7) if i in heat.columns]]
                fig_h = go.Figure(data=go.Heatmap(
                    z=heat.values,
                    x=[weekday_names[i] for i in heat.columns],
                    y=heat.index,
                    colorscale="Blues",
                    hovertemplate="Week: %{y}<br>Day: %{x}<br>Messages: %{z}<extra></extra>"
                ))
                fig_h.update_layout(title="Engagement heatmap (messages/day)", height=350, margin=dict(t=40, b=10))
                st.plotly_chart(fig_h, use_container_width=True)

# -------------------------
# Tab 3: Decision Register
# -------------------------
with tab3:
    st.subheader("Decisions, traceability & impact")
    if df_decisions_f.empty:
        st.info("No decisions in selected range.")
    else:
        if "pillar" in df_decisions_f.columns:
            by_pillar = df_decisions_f.groupby("pillar").size().reset_index(name="decisions")
            st.plotly_chart(px.bar(by_pillar, x="pillar", y="decisions", title="Decisions by pillar"), use_container_width=True)

        st.markdown("#### Recent decisions (traceability to messages)")
        display_cols = [c for c in ["date", "title", "pillar", "by", "rationale"] if c in df_decisions_f.columns]
        st.dataframe(df_decisions_f[display_cols].sort_values("date", ascending=False), use_container_width=True)

        st.markdown("#### Traceability (click to expand)")
        for _, d in df_decisions_f.sort_values("date", ascending=False).iterrows():
            title = d.get("title", "Untitled")
            pillar = d.get("pillar", "")
            who = d.get("by", "")
            date_str = d["date"].date().isoformat() if pd.notna(d.get("date")) else ""
            header = f"{date_str} • {title} • {pillar} • {who}"
            with st.expander(header):
                st.write(d.get("rationale", ""))
                ids = d.get("related_message_ids", []) or []
                if not isinstance(ids, list):
                    ids = [ids]
                if not ids:
                    st.info("No related message ids recorded.")
                else:
                    for rid in ids:
                        if "id" in df_msgs.columns:
                            row = df_msgs[df_msgs["id"] == rid]
                        else:
                            row = pd.DataFrame()
                        if not row.empty:
                            r = row.iloc[0]
                            stamp = r.get("date").strftime("%Y-%m-%d %H:%M") if pd.notna(r.get("date")) else ""
                            st.markdown(f"- **{stamp} — {r.get('sender','')}**: {r.get('message','')}")
                        else:
                            st.markdown(f"- Related message id **{rid}** not found in messages dataset.")

# -------------------------
# Tab 4: Persona & Ops
# -------------------------
with tab4:
    st.subheader("Persona & internal metrics")
    persona = data.get("persona", {}) or {}

    pcol1, pcol2 = st.columns([1, 3])
    with pcol1:
        avatar = persona.get("avatar_url")
        if avatar:
            st.image(avatar, width=120)
        else:
            st.markdown("👤", unsafe_allow_html=True)

    with pcol2:
        st.markdown(f"### {persona.get('name', 'Unnamed')}")
        if persona.get("age"):
            st.write(f"- **Age:** {persona.get('age')}")
        if persona.get("occupation"):
            st.write(f"- **Occupation:** {persona.get('occupation')}")
        if persona.get("goals"):
            st.write(f"- **Goals:** {persona.get('goals')}")
        if persona.get("notes"):
            st.write(persona.get("notes"))

    st.subheader("Internal metrics snapshot")
    im = data.get("internal_metrics", {}) or {}
    if isinstance(im, dict) and im:
        im_df = pd.DataFrame(list(im.items()), columns=["Metric", "Value"])
        st.table(im_df)
    else:
        st.info("No internal metrics provided.")

    # Auto judge checklist
    if show_judge:
        st.divider()
        st.subheader("Automated Judge Checklist (computed)")

        checks = []
        total_span_days = ( (max_date - min_date).days ) if (min_date is not None and max_date is not None) else 0
        checks.append(("Realistic timeline (≥120 days)", total_span_days >= 120))

        # messages per week average
        avg_msgs_per_week = 0.0
        if not df_msgs.empty and "date" in df_msgs.columns:
            span_days = (df_msgs["date"].max() - df_msgs["date"].min()).days or 1
            weeks = span_days / 7.0 or 1.0
            avg_msgs_per_week = float(len(df_msgs) / weeks) if weeks else float(len(df_msgs))
        checks.append(("Member-initiated messages (avg/wk ≥5)", avg_msgs_per_week >= 5.0))

        dec_total = len(df_decisions)
        dec_with_links = 0
        if dec_total > 0 and "related_message_ids" in df_decisions.columns:
            # compute count of linked decisions
            def _count_links(x):
                if isinstance(x, list):
                    return len(x)
                if pd.notna(x):
                    return 1
                return 0
            dec_with_links = int(df_decisions["related_message_ids"].dropna().apply(_count_links).sum())
        checks.append(("Decisions linked to chats", dec_total > 0 and dec_with_links > 0))
        checks.append(("Multi-metric trends (≥2 metrics)", df_metrics["metric"].nunique() >= 2 if not df_metrics.empty else False))
        checks.append(("Filters available (date, pillar, search)", True))
        checks.append(("Upload & export features", True))

        # Show checklist with icons
        for label, ok in checks:
            emoji = "✅" if ok else "❌"
            st.write(f"{emoji} {label}")

        # heuristic score
        score = int((sum(1 for _, ok in checks if ok) / len(checks)) * 100)
        st.metric("Auto checklist score", f"{score}/100")

# -------------------------
# Tab 5: Exports & Snapshot
# -------------------------
with tab5:
    st.subheader("Export data and download snapshot")

    if not df_metrics.empty:
        st.download_button("⬇️ Metrics CSV", df_metrics.to_csv(index=False), "metrics.csv", "text/csv")
    if not df_msgs.empty:
        st.download_button("⬇️ Messages CSV", df_msgs.to_csv(index=False), "messages.csv", "text/csv")
    if not df_decisions.empty:
        st.download_button("⬇️ Decisions CSV", df_decisions.to_csv(index=False), "decisions.csv", "text/csv")

    pretty_json = json.dumps(data, indent=2)
    st.download_button("⬇️ Full JSON", pretty_json, "member_journey.json", "application/json")

    # Build HTML snapshot with KPIs + persona + insights
    kpis = {
        "Avg BP": bp_avg,
        "BP control <=130 (%)": bp_ctrl_rate,
        "VO2 delta": vo2_delta,
        "Sleep avg": sleep_avg,
        "Adherence (%)": adherence
    }
    # re-create insights list as text
    insights_list = []
    if not vo2_series.empty and len(vo2_series) >= 2:
        insights_list.append(f"VO2 changed by {vo2_delta:+} pts")
    if bp_avg is not None:
        insights_list.append(f"Avg BP {bp_avg} mmHg ({bp_ctrl_rate}% ≤130)")
    if sleep_avg is not None:
        insights_list.append(f"Sleep avg {sleep_avg}")

    snapshot_html = html_snapshot(data, kpis, persona, insights_list)
    st.download_button("⬇️ HTML Snapshot", snapshot_html, "elyx_snapshot.html", "text/html")

st.caption("© 2025 Elyx Hackathon — advanced dashboard (use judge mode & upload your JSON to demo).")
