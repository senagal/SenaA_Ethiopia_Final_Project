# -*- coding: utf-8 -*-
"""
streamlit_ethiopia.py
=====================
Dashboard: Guardian Coverage of Ethiopia 1995–2021

Research questions:
  1. How has Guardian coverage of Ethiopia changed over time?
  2. How has sentiment toward Ethiopia changed over time?
  3. Which topics dominate Ethiopia-related coverage?
  4. Do years with more focused coverage correspond to higher/lower
     tourist arrivals the following year?
  5. Do years with more negative coverage correspond to lower tourism
     the following year?
  6. Do years with more positive coverage correspond to higher tourism
     the following year?

Run with:
    streamlit run streamlit_ethiopia.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import plotly.graph_objects as go
import streamlit as st
from statsmodels.formula.api import ols

# ============================================================
# Page setup
# ============================================================

st.set_page_config(
    page_title="Guardian Coverage of Ethiopia",
    layout="wide",
)

st.title("Guardian Coverage of Ethiopia 1995–2021")
st.markdown(
    "How has Guardian media coverage, sentiment, and topics about Ethiopia "
    "changed over time — and are they associated with tourist arrivals?"
)

# ============================================================
# Load data
# ============================================================

INPUT_FILE = "data/ethiopia_analysis.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(INPUT_FILE)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    numeric_cols = [
        "total_articles", "focused_articles", "focus_rate",
        "average_sentiment_score",
        "positive_rate", "negative_rate", "neutral_rate",
        "positive_articles", "negative_articles", "neutral_articles",
        "tourist_arrivals", "tourism_growth_rate",
        "tourist_arrivals_next_year", "tourism_growth_rate_next_year",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    topic_rate_cols = [
        c for c in df.columns
        if c.endswith("_rate")
        and c not in {
            "focus_rate", "positive_rate", "negative_rate",
            "neutral_rate", "tourism_growth_rate",
            "tourism_growth_rate_next_year",
        }
    ]
    topic_count_cols = [c for c in df.columns if c.endswith("_count")]

    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    df = df.sort_values("year").reset_index(drop=True)

    return df, topic_rate_cols, topic_count_cols

df, topic_rate_cols, topic_count_cols = load_data()

year_min = int(df["year"].min())
year_max = int(df["year"].max())

# ── Clean topic names for display ─────────────────────────────────────────────
def pretty_topic(col, suffix="_rate"):
    return (col.replace(suffix, "").replace("_", " ").title())

# ============================================================
# Helper: simple linear regression plot (matplotlib)
# ============================================================

def regression_plot(x, y, x_label, y_label, title, year_labels=None):
    """
    Fits y = m*x + k and plots data points, fitted line, and residual lines.
    Labels each point with its year.
    Returns slope, R², p-value.
    """
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = np.array(x)[mask], np.array(y)[mask]
    if year_labels is not None:
        year_labels = np.array(year_labels)[mask]

    if len(x) < 3:
        st.info("Not enough data points for this regression.")
        return

    # Fit line — same approach as the Kujenga course
    m, k  = np.polyfit(x, y, 1)
    model = ols(f"y ~ x", data=pd.DataFrame({"x": x, "y": y})).fit()
    r2    = model.rsquared
    pval  = model.pvalues["x"]

    predicted = m * x + k
    y_range   = y.max() - y.min() if y.max() != y.min() else 1

    fig, ax = plt.subplots(figsize=(10, 4))

    # Residual dotted lines
    for xi, yi, pi in zip(x, y, predicted):
        ax.plot([xi, xi], [yi, pi], linestyle=":", color="gray", linewidth=0.8, zorder=1)

    # Data points
    ax.scatter(x, y, color="#1A6B5A", s=60, zorder=3)

    # Year labels
    if year_labels is not None:
        for xi, yi, yr in zip(x, y, year_labels):
            ax.text(xi, yi + y_range * 0.025, str(int(yr)),
                    fontsize=7, ha="center", alpha=0.85)

    # Fitted line
    x_line = np.linspace(x.min(), x.max(), 200)
    ax.plot(x_line, m * x_line + k, color="black", linewidth=1.5, zorder=2)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    direction = "positive ↑" if m > 0 else "negative ↓"
    sig = "statistically significant ✓" if pval < 0.05 else "not statistically significant ✗"
    st.caption(
        f"Slope m = **{m:.4f}** · R² = **{r2:.3f}** · p-value = **{pval:.4f}** — "
        f"**{direction}** association, **{sig}**"
    )
    return {"slope": m, "r_squared": r2, "p_value": pval}


# ============================================================
# Tabs
# ============================================================

tabs = st.tabs([
    "Pipeline",
    "Q1  Coverage over time",
    "Q2  Sentiment over time",
    "Q3  Topics",
    "Q4  Coverage → Tourism",
    "Q5  Negative → Tourism",
    "Q6  Positive → Tourism",
])

# ── Pipeline ──────────────────────────────────────────────────────────────────
with tabs[0]:
    st.header("Project Pipeline")
    st.markdown(f"""
### Research question
How has Guardian coverage, sentiment, and topics about Ethiopia changed over time,
and are they associated with international tourist arrivals?

### Data sources
- **Guardian API** — all articles mentioning Ethiopia, 1995–2021
- **International tourist arrivals** — Ethiopia, 1995–2021

### Pipeline

**Step 1 — Fetch (`fetch_ethiopia.py`)**  
Queries the Guardian API for any article mentioning Ethiopia (or "Ethiopian", "Ethiopians", "Walias").
Fetches in 30-day chunks from 1995 to 2021. Saves progress so it can be safely interrupted and resumed.

**Step 2 — Label (`label_ethiopia.py`)**  
Sends each article to GPT-4o-mini in batches of 20. Assigns:
- `is_focus` — is Ethiopia the primary subject, or just mentioned?
- `topic` — one of: Politics & Governance, Economy & Business, Sport & Football,
  Tourism & Culture, Health & Development, Conflict & Security, Environment & Climate, Other
- `sentiment` — Positive, Neutral, or Negative toward Ethiopia

Saves progress per year so you can resume without re-calling GPT.

**Step 3 — Prepare (`prepare_ethiopia.py`)**  
Aggregates labelled articles to one row per year. Calculates:
- Total and focused article counts
- Sentiment rates and average sentiment score
- Topic counts and rates (focused articles only)
- Tourist arrivals and year-on-year growth rate
- Lagged tourism (year t+1) for the regression questions

**Step 4 — Dashboard (`streamlit_ethiopia.py`)**  
Visualises all six research questions. Uses simple linear regression
(one data point per year) with slope, R², and p-value shown below each chart.

---
*Loaded {len(df)} years of data · {year_min}–{year_max}*
    """)
    st.info("All regression results show association, not causation.")

# ── Q1: Coverage over time ────────────────────────────────────────────────────
with tabs[1]:
    st.header("Q1. How has Guardian coverage of Ethiopia changed over time?")
    st.markdown(
        "Each bar shows the number of Guardian articles mentioning Ethiopia in that year, "
        "split into articles where Ethiopia is the **main focus** (dark teal) "
        "and articles where Ethiopia is only **mentioned in passing** (light blue)."
    )

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Focused (main subject)",
        x=df["year"], y=df["focused_articles"],
        marker_color="#1A6B5A",
        customdata=np.stack([
            df["total_articles"], df["focused_articles"],
            df["focus_rate"].fillna(0),
        ], axis=1),
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Focused articles: <b>%{y}</b><br>"
            "Total articles: %{customdata[0]}<br>"
            "Focus rate: %{customdata[2]:.1%}<extra></extra>"
        ),
    ))

    fig.add_trace(go.Bar(
        name="Mention only",
        x=df["year"],
        y=(df["total_articles"] - df["focused_articles"]).clip(lower=0),
        marker_color="#90CAF9",
        customdata=np.stack([df["total_articles"]], axis=1),
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Mention-only articles: <b>%{y}</b><br>"
            "Total articles: %{customdata[0]}<extra></extra>"
        ),
    ))

    fig.update_layout(
        barmode="stack",
        title="Guardian articles mentioning Ethiopia per year",
        xaxis_title="Year",
        yaxis_title="Number of articles",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=420,
        xaxis=dict(dtick=1, tickangle=45),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Coverage trend line
    st.subheader("Coverage trend — linear regression over time")
    st.markdown(
        "Fitting a line through total articles per year tells us whether "
        "coverage has been increasing or decreasing on average."
    )
    regression_plot(
        x=df["year"], y=df["total_articles"],
        x_label="Year", y_label="Total articles",
        title="Total Guardian coverage of Ethiopia over time",
        year_labels=df["year"],
    )

    # Key stats
    col1, col2, col3 = st.columns(3)
    with col1:
        peak_year = df.loc[df["total_articles"].idxmax(), "year"]
        peak_n    = int(df["total_articles"].max())
        st.metric("Peak coverage year", peak_year, f"{peak_n} articles")
    with col2:
        st.metric("Total articles (all years)", int(df["total_articles"].sum()))
    with col3:
        st.metric("Total focused articles", int(df["focused_articles"].sum()))

# ── Q2: Sentiment over time ───────────────────────────────────────────────────
with tabs[2]:
    st.header("Q2. How has sentiment toward Ethiopia changed over time?")
    st.markdown(
        "**Average sentiment score** is calculated as: Positive=+1, Neutral=0, Negative=−1. "
        "A score above zero means more positive articles than negative in that year."
    )

    # Sentiment score line chart
    fig2a = go.Figure()
    fig2a.add_trace(go.Scatter(
        x=df["year"], y=df["average_sentiment_score"],
        mode="lines+markers",
        line=dict(color="#1A6B5A", width=2),
        marker=dict(size=7),
        name="Average sentiment",
        hovertemplate="<b>%{x}</b><br>Avg sentiment: %{y:.3f}<extra></extra>",
    ))
    fig2a.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
    fig2a.update_layout(
        title="Average sentiment score toward Ethiopia per year",
        xaxis_title="Year", yaxis_title="Average sentiment score (−1 to +1)",
        height=380, xaxis=dict(dtick=1, tickangle=45),
    )
    st.plotly_chart(fig2a, use_container_width=True)

    # Stacked area: positive / neutral / negative counts
    st.subheader("Breakdown: positive, neutral, and negative articles per year")
    fig2b = go.Figure()
    for col, label, colour in [
        ("positive_articles", "Positive", "#43A047"),
        ("neutral_articles",  "Neutral",  "#FFA726"),
        ("negative_articles", "Negative", "#E53935"),
    ]:
        if col in df.columns:
            fig2b.add_trace(go.Bar(
                name=label, x=df["year"], y=df[col].fillna(0),
                marker_color=colour,
                hovertemplate=f"<b>%{{x}}</b><br>{label}: %{{y}}<extra></extra>",
            ))
    fig2b.update_layout(
        barmode="stack",
        title="Sentiment breakdown per year",
        xaxis_title="Year", yaxis_title="Number of articles",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=380, xaxis=dict(dtick=1, tickangle=45),
    )
    st.plotly_chart(fig2b, use_container_width=True)

    # Sentiment trend line
    st.subheader("Sentiment trend — linear regression over time")
    regression_plot(
        x=df["year"], y=df["average_sentiment_score"],
        x_label="Year", y_label="Average sentiment score",
        title="Has sentiment toward Ethiopia changed over time?",
        year_labels=df["year"],
    )

    # Key stats
    col1, col2, col3 = st.columns(3)
    with col1:
        most_pos = df.loc[df["average_sentiment_score"].idxmax()]
        st.metric("Most positive year", int(most_pos["year"]),
                  f"score {most_pos['average_sentiment_score']:.3f}")
    with col2:
        most_neg = df.loc[df["average_sentiment_score"].idxmin()]
        st.metric("Most negative year", int(most_neg["year"]),
                  f"score {most_neg['average_sentiment_score']:.3f}")
    with col3:
        overall = df["average_sentiment_score"].mean()
        st.metric("Overall average sentiment", f"{overall:.3f}")

# ── Q3: Topics ────────────────────────────────────────────────────────────────
with tabs[3]:
    st.header("Q3. Which topics dominate Ethiopia-related coverage?")
    st.markdown(
        "Topic rates are calculated from **focused articles only** — "
        "articles where Ethiopia is the main subject. "
        "This avoids counting articles where Ethiopia is mentioned only briefly."
    )

    if not topic_count_cols:
        st.info("No topic data found. Check that label_ethiopia.py ran successfully.")
    else:
        # Overall topic totals (bar chart)
        topic_totals = (
            df[topic_count_cols].sum()
            .reset_index()
        )
        topic_totals.columns = ["topic", "count"]
        topic_totals["topic"] = topic_totals["topic"].apply(
            lambda x: pretty_topic(x, "_count")
        )
        topic_totals = topic_totals.sort_values("count", ascending=True)

        fig3a = go.Figure(go.Bar(
            x=topic_totals["count"], y=topic_totals["topic"],
            orientation="h", marker_color="#1A6B5A",
            hovertemplate="<b>%{y}</b><br>Articles: %{x}<extra></extra>",
        ))
        fig3a.update_layout(
            title="Total focused articles per topic (1995–2021)",
            xaxis_title="Number of focused articles",
            height=400,
        )
        st.plotly_chart(fig3a, use_container_width=True)

        # Topic rates over time (line chart — pick topic)
        st.subheader("Topic rate over time")
        st.markdown("Select a topic to see how its share of coverage has changed year by year.")

        topic_display = {pretty_topic(c, "_rate"): c for c in topic_rate_cols}
        selected_topic_label = st.selectbox(
            "Select topic", list(topic_display.keys()), key="q3_topic"
        )
        selected_topic_col = topic_display[selected_topic_label]

        fig3b = go.Figure(go.Scatter(
            x=df["year"], y=df[selected_topic_col].fillna(0),
            mode="lines+markers",
            line=dict(color="#1A6B5A", width=2), marker=dict(size=7),
            hovertemplate="<b>%{x}</b><br>Rate: %{y:.1%}<extra></extra>",
        ))
        fig3b.update_layout(
            title=f"{selected_topic_label} — share of focused articles per year",
            xaxis_title="Year", yaxis_title="Topic rate",
            height=350, xaxis=dict(dtick=1, tickangle=45),
        )
        st.plotly_chart(fig3b, use_container_width=True)

        # Heatmap: topic × year
        st.subheader("Topic heatmap — all topics across all years")
        heatmap_data = df.set_index("year")[topic_rate_cols].T
        heatmap_data.index = [pretty_topic(c, "_rate") for c in heatmap_data.index]

        fig3c, ax = plt.subplots(figsize=(14, 5))
        import matplotlib.cm as cm
        im = ax.imshow(heatmap_data.values.astype(float),
                       aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_xticks(range(len(heatmap_data.columns)))
        ax.set_xticklabels(heatmap_data.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(heatmap_data.index)))
        ax.set_yticklabels(heatmap_data.index, fontsize=9)
        plt.colorbar(im, ax=ax, label="Topic rate")
        ax.set_title("Topic distribution per year (focused articles only)")
        plt.tight_layout()
        st.pyplot(fig3c)
        plt.close()

# ── Q4: Focused coverage → next-year tourism ─────────────────────────────────
with tabs[4]:
    st.header("Q4. Do years with more focused coverage correspond to higher or lower tourist arrivals the following year?")
    st.markdown("""
We use a **one-year lag** — coverage in year **t** is compared with tourism in year **t+1**.
This is stronger than same-year comparison because coverage comes *before* the tourism change,
making it harder to argue that tourism caused the coverage.

Each data point is one year. The fitted line and its slope tell us the direction of the relationship.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Focused articles → next-year tourism")
        regression_plot(
            x=df["focused_articles"],
            y=df["tourist_arrivals_next_year"],
            x_label="Focused articles (year t)",
            y_label="Tourist arrivals (year t+1)",
            title="Does focused coverage predict next-year tourism?",
            year_labels=df["year"],
        )
    with col2:
        st.subheader("Total articles → next-year tourism")
        regression_plot(
            x=df["total_articles"],
            y=df["tourist_arrivals_next_year"],
            x_label="Total articles (year t)",
            y_label="Tourist arrivals (year t+1)",
            title="Does total coverage predict next-year tourism?",
            year_labels=df["year"],
        )

    # Also show tourism over time for context
    st.subheader("Tourist arrivals over time — for context")
    fig4b = go.Figure()
    fig4b.add_trace(go.Scatter(
        x=df["year"], y=df["tourist_arrivals"],
        mode="lines+markers",
        line=dict(color="#E65100", width=2), marker=dict(size=7),
        hovertemplate="<b>%{x}</b><br>Tourist arrivals: %{y:,.0f}<extra></extra>",
    ))
    fig4b.update_layout(
        title="Ethiopia international tourist arrivals 1995–2021",
        xaxis_title="Year", yaxis_title="Tourist arrivals",
        height=350, xaxis=dict(dtick=1, tickangle=45),
    )
    st.plotly_chart(fig4b, use_container_width=True)

# ── Q5: Negative coverage → next-year tourism ────────────────────────────────
with tabs[5]:
    st.header("Q5. Do years with more negative coverage correspond to lower tourism the following year?")
    st.markdown("""
A negative slope here would mean that years when Guardian coverage was more negative
were followed by *fewer* tourists the next year.

We test both the **negative rate** (proportion of negative articles)
and the **count of negative articles**.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Negative rate → next-year tourism")
        regression_plot(
            x=df["negative_rate"],
            y=df["tourist_arrivals_next_year"],
            x_label="Negative article rate (year t)",
            y_label="Tourist arrivals (year t+1)",
            title="Does more negative coverage predict fewer tourists?",
            year_labels=df["year"],
        )
    with col2:
        st.subheader("Negative article count → next-year tourism")
        regression_plot(
            x=df["negative_articles"],
            y=df["tourist_arrivals_next_year"],
            x_label="Negative articles (year t)",
            y_label="Tourist arrivals (year t+1)",
            title="Negative article count vs next-year tourism",
            year_labels=df["year"],
        )

    # Same-year for comparison
    st.markdown("---")
    st.subheader("Same-year comparison (for reference)")
    st.caption("This is less rigorous than the lagged version above — shown only for comparison.")
    regression_plot(
        x=df["negative_rate"],
        y=df["tourist_arrivals"],
        x_label="Negative article rate (year t)",
        y_label="Tourist arrivals (year t)",
        title="Same-year: negative rate vs tourist arrivals",
        year_labels=df["year"],
    )

# ── Q6: Positive coverage → next-year tourism ────────────────────────────────
with tabs[6]:
    st.header("Q6. Do years with more positive coverage correspond to higher tourism the following year?")
    st.markdown("""
A positive slope here would mean that years when Guardian coverage was more positive
were followed by *more* tourists the next year.

We test both the **positive rate** (proportion of positive articles)
and the **count of positive articles**.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Positive rate → next-year tourism")
        regression_plot(
            x=df["positive_rate"],
            y=df["tourist_arrivals_next_year"],
            x_label="Positive article rate (year t)",
            y_label="Tourist arrivals (year t+1)",
            title="Does more positive coverage predict more tourists?",
            year_labels=df["year"],
        )
    with col2:
        st.subheader("Positive article count → next-year tourism")
        regression_plot(
            x=df["positive_articles"],
            y=df["tourist_arrivals_next_year"],
            x_label="Positive articles (year t)",
            y_label="Tourist arrivals (year t+1)",
            title="Positive article count vs next-year tourism",
            year_labels=df["year"],
        )

    # Same-year for comparison
    st.markdown("---")
    st.subheader("Same-year comparison (for reference)")
    st.caption("Shown only for comparison with the lagged version above.")
    regression_plot(
        x=df["positive_rate"],
        y=df["tourist_arrivals"],
        x_label="Positive article rate (year t)",
        y_label="Tourist arrivals (year t)",
        title="Same-year: positive rate vs tourist arrivals",
        year_labels=df["year"],
    )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Data: Guardian API · International tourist arrivals (Our World in Data). "
    "All regression results show association, not causation."
)
