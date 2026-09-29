"""Streamlit front end for InsightPilot."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

from insight_pilot.data import DATA_PATH, INDICATORS, load_data, refresh_data  # noqa: E402
from insight_pilot.agent import ask_agent  # noqa: E402
from insight_pilot.extensions import run_all  # noqa: E402

st.set_page_config(page_title="InsightPilot | Agentic Data Analyst", page_icon="🧭", layout="wide")
st.title("🧭 InsightPilot")
st.caption("An agentic data analyst that plans questions, calls bounded analytics tools, and explains the evidence.")

with st.sidebar:
    st.header("Dataset")
    st.write("World Bank development indicators · 8 countries · 5 measures")
    st.caption("Weekly snapshot via the World Bank API. Latest observation years vary by indicator.")
    if st.button("Refresh data now", use_container_width=True):
        with st.spinner("Fetching the latest available observations…"):
            try:
                refresh_data()
                st.cache_data.clear()
                st.success("Snapshot refreshed.")
                st.rerun()
            except Exception as exc:
                st.error(f"Refresh failed: {exc}")
    uploaded = st.file_uploader("Or analyze your own tidy CSV", type=["csv"], help="Required columns: country_code, country, year, indicator, indicator_code, value")

@st.cache_data(ttl=3600)
def cached_load(path: str, modified: float) -> pd.DataFrame:
    return pd.read_csv(path)

try:
    if uploaded is not None:
        frame = pd.read_csv(uploaded)
        required = {"country_code", "country", "year", "indicator", "indicator_code", "value"}
        missing = required.difference(frame.columns)
        if missing:
            st.error(f"Uploaded CSV is missing: {', '.join(sorted(missing))}")
            st.stop()
    else:
        if not DATA_PATH.exists():
            with st.spinner("First run: downloading public indicators…"):
                refresh_data()
        frame = cached_load(str(DATA_PATH), DATA_PATH.stat().st_mtime)
except Exception as exc:
    st.error(f"Data could not be loaded: {exc}")
    st.info("Check your connection, then try Refresh data now.")
    st.stop()

latest_year = int(frame["year"].max())
country_count = frame["country"].nunique()
indicator_count = frame["indicator"].nunique()
c1, c2, c3 = st.columns(3)
c1.metric("Observations", f"{len(frame):,}")
c2.metric("Countries", country_count)
c3.metric("Indicators", indicator_count)

left, right = st.columns([1.2, 1])
with left:
    indicator = st.selectbox("Explore an indicator", sorted(frame["indicator"].dropna().unique()))
    plot_data = frame[frame["indicator"] == indicator]
    figure = px.line(plot_data, x="year", y="value", color="country", markers=True, title=indicator,
                     labels={"year": "Year", "value": indicator, "country": "Country"})
    figure.update_layout(legend_title_text="Country", margin=dict(l=10, r=10, t=48, b=10))
    st.plotly_chart(figure, use_container_width=True)
with right:
    st.subheader("Ask the analyst")
    st.write("Questions are answered with tool-calculated results, not free-form code execution.")
    examples = [
        "How has GDP per capita in India changed since 2000?",
        "Compare internet use across India, Bangladesh, and Indonesia in the latest year with coverage.",
        "Is internet use associated with GDP per capita across countries? What are the limitations?",
    ]
    prompt = st.text_area("Your question", placeholder=examples[0], height=100)
    st.caption("Try: " + " · ".join(examples))
    if st.button("Analyze", type="primary", disabled=not prompt.strip(), use_container_width=True):
        with st.spinner("Planning the analysis and calculating evidence…"):
            try:
                st.session_state["answer"] = ask_agent(prompt.strip(), frame)
            except Exception as exc:
                st.error(str(exc))
    if st.session_state.get("answer"):
        st.markdown(st.session_state["answer"])

with st.expander("Data preview and caveats"):
    st.dataframe(frame.head(100), use_container_width=True, hide_index=True)
    st.markdown(
        "- Source: World Bank Open Data API. Values use each indicator's published units.\n"
        "- Country indicators are reported at different times; the latest year is not the same for every measure.\n"
        "- Missing observations are omitted, not filled. Correlation does not show causation."
    )

st.subheader("Automated insight modules")
st.caption("Small, deterministic analytics plug-ins. New modules can be proposed through the daily review workflow.")
for extension_name, result in run_all(frame).items():
    with st.expander(extension_name.replace("_", " ").title()):
        st.json(result)
