# -*- coding: utf-8 -*-
"""
streamlit_ethiopia.py
=====================
Dashboard: The Guardian Coverage of Ethiopia 1996-2021

Research questions:
  1. How has The Guardian coverage of Ethiopia changed over time?
  2. How has the sentiment of The Guardian articles on Ethiopia changed over time?
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
import streamlit.components.v1 as components
from statsmodels.formula.api import ols

# ============================================================
# Page setup
# ============================================================

st.set_page_config(
    page_title="The Guardian Coverage of Ethiopia",
    layout="wide",
)

st.title("The association of The Guardian Article Coverage on Ethiopia's Tourist arrivals (1996-2021)")
st.markdown(
    "How has The Guardian media coverage, sentiment, and topics about Ethiopia "
    "changed over time, and are they associated with tourist arrivals?"
    "Repo: https://github.com/senagal/SenaAbdisa_Ethiopia_FinalProject"
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

    # Fit line - same approach as the Kujenga course
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
        f"Slope m = **{m:.4f}** · R² = **{r2:.3f}** · p-value = **{pval:.4f}** - "
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
    "Conclusion",
])

# ── Pipeline ──────────────────────────────────────────────────────────────────
with tabs[0]:
    st.header("About This Project")

    st.markdown("""
### What is the Project's Objective?

This project started from a simple question: *"How are we being portrayed in international media? And does it have any tangible effect on our economy?"*

The main objective is to observe and analyse media coverage of Ethiopia in digital article sites and check if it has any correlation with the number of tourists we get.

Due to legal policies against obtaining information from most sites, this project is limited to **The Guardian** and all article-related information is analysed from data obtained through their official API. The tourist arrival data is obtained from Kaggle, by Mohamadreza Momeni, available [here](https://www.kaggle.com/datasets/imtkaggleteam/tourism?select=26-+international-arrivals-for-personal-vs-business-and-professional-reasons.csv).
    """)

    st.markdown("---")
    st.subheader("Research Questions")
    st.markdown("""
1. How has the quantity of The Guardian's coverage of Ethiopia changed over time?
2. How has the sentiment of their articles on Ethiopia changed over time?
3. Which topics dominate Ethiopia related coverage?
4. Do years with more or less focused coverage correlate with higher or lower tourist arrivals the following year?
5. Do years with more negative coverage correspond to lower tourism the following year?
6. Do years with more positive coverage correspond to higher tourism the following year?
    """)

    st.markdown("---")
    st.subheader("Pipeline overview")
    st.markdown("Here is how the four scripts connect and what each one produces. Run them in this order:")

    pipeline_svg = """
<svg width="100%" viewBox="0 0 680 520" role="img" xmlns="http://www.w3.org/2000/svg"
     style="max-width:560px;display:block;margin:0 auto;">
  <title>Ethiopia project pipeline flow</title>
  <defs>
    <marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>

  <!-- Step 1 -->
  <rect x="215" y="20" width="250" height="56" rx="8" fill="#9FE1CB" stroke="#0F6E56" stroke-width="0.5"/>
  <text x="340" y="42" text-anchor="middle" dominant-baseline="central" font-size="14" font-weight="500" fill="#085041">fetch_ethiopia.py</text>
  <text x="340" y="60" text-anchor="middle" dominant-baseline="central" font-size="12" fill="#0F6E56">Calls The Guardian API</text>

  <line x1="340" y1="76" x2="340" y2="108" stroke="#888780" stroke-width="1.5" marker-end="url(#arr)"/>

  <!-- File 1 -->
  <rect x="240" y="110" width="200" height="40" rx="6" fill="#F1EFE8" stroke="#888780" stroke-width="0.5"/>
  <text x="340" y="130" text-anchor="middle" dominant-baseline="central" font-size="12" fill="#444441">ethiopia_raw.json</text>
  <text x="452" y="124" font-size="11" fill="#888780">~9,000 articles</text>
  <text x="452" y="139" font-size="11" fill="#888780">1996 to 2021</text>

  <line x1="340" y1="150" x2="340" y2="182" stroke="#888780" stroke-width="1.5" marker-end="url(#arr)"/>

  <!-- Step 2 -->
  <rect x="215" y="184" width="250" height="56" rx="8" fill="#9FE1CB" stroke="#0F6E56" stroke-width="0.5"/>
  <text x="340" y="206" text-anchor="middle" dominant-baseline="central" font-size="14" font-weight="500" fill="#085041">label_ethiopia.py</text>
  <text x="340" y="224" text-anchor="middle" dominant-baseline="central" font-size="12" fill="#0F6E56">Labels each article with GPT</text>

  <line x1="340" y1="240" x2="340" y2="272" stroke="#888780" stroke-width="1.5" marker-end="url(#arr)"/>

  <!-- File 2 -->
  <rect x="225" y="274" width="230" height="40" rx="6" fill="#F1EFE8" stroke="#888780" stroke-width="0.5"/>
  <text x="340" y="294" text-anchor="middle" dominant-baseline="central" font-size="12" fill="#444441">ethiopia_labelled.csv</text>
  <text x="467" y="288" font-size="11" fill="#888780">topic, sentiment,</text>
  <text x="467" y="303" font-size="11" fill="#888780">is_focus per article</text>

  <line x1="340" y1="314" x2="340" y2="346" stroke="#888780" stroke-width="1.5" marker-end="url(#arr)"/>

  <!-- Tourism CSV input from left -->
  <rect x="28" y="354" width="168" height="44" rx="6" fill="#FAC775" stroke="#854F0B" stroke-width="0.5"/>
  <text x="112" y="371" text-anchor="middle" dominant-baseline="central" font-size="13" font-weight="500" fill="#633806">tourism CSV</text>
  <text x="112" y="389" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#854F0B">downloaded from Kaggle</text>
  <line x1="196" y1="376" x2="213" y2="376" stroke="#888780" stroke-width="1.5" marker-end="url(#arr)"/>

  <!-- Step 3 -->
  <rect x="215" y="348" width="250" height="56" rx="8" fill="#9FE1CB" stroke="#0F6E56" stroke-width="0.5"/>
  <text x="340" y="370" text-anchor="middle" dominant-baseline="central" font-size="14" font-weight="500" fill="#085041">prepare_ethiopia.py</text>
  <text x="340" y="388" text-anchor="middle" dominant-baseline="central" font-size="12" fill="#0F6E56">Merges articles + tourism data</text>

  <line x1="340" y1="404" x2="340" y2="436" stroke="#888780" stroke-width="1.5" marker-end="url(#arr)"/>

  <!-- File 3 -->
  <rect x="225" y="438" width="230" height="40" rx="6" fill="#F1EFE8" stroke="#888780" stroke-width="0.5"/>
  <text x="340" y="458" text-anchor="middle" dominant-baseline="central" font-size="12" fill="#444441">ethiopia_analysis.csv</text>
  <text x="467" y="452" font-size="11" fill="#888780">one row per year,</text>
  <text x="467" y="467" font-size="11" fill="#888780">ready to analyse</text>

  <line x1="340" y1="478" x2="340" y2="500" stroke="#888780" stroke-width="1.5" marker-end="url(#arr)"/>
  <text x="340" y="514" text-anchor="middle" font-size="12" fill="#0F6E56" font-weight="500">streamlit run streamlit.py</text>
</svg>
"""
    components.html(pipeline_svg, height=540)

    st.markdown("---")
    st.subheader("How the Data Was Sourced and Cleaned")

    st.markdown("#### The Guardian Articles")
    st.markdown("""
Articles were fetched from The Guardian's [Open Platform API](https://open-platform.theguardian.com/), which provides free access to their full article archive. To use the fetch script yourself:

1. Register for a free API key at [open-platform.theguardian.com](https://open-platform.theguardian.com/)
2. Add it to a `.env` file in the project root: `GUARDIAN_API_KEY=your_key_here`
3. Run `python fetch_ethiopia.py`

The script queries for any article containing the words "Ethiopia", "Ethiopian", or "Ethiopians". To avoid hitting rate limits, it fetches in **30-day chunks** from 1996 to 2021, saving each chunk to `data/progress_ethiopia/` as it goes. This means if the script is interrupted, you can restart it and it will pick up from where it left off without re-fetching already-completed chunks.

Each article is saved with its headline, summary, body text, author, publication date, section, word count, and tags. Duplicate articles (same URL appearing in multiple chunks) are removed before the final file is written to `data/ethiopia_raw.json`.
    """)

    st.markdown("#### Article Labelling and Cleaning")
    st.markdown("""
Raw articles were then passed through `label_ethiopia.py`, which sends each article to gpt-5-mini for classification. Each article receives three labels:

- **is_focus** : whether Ethiopia is the primary subject of the article, or just mentioned in passing
- **topic** : the single bestmfitting topic from: Politics & Governance, Economy & Business, Sport & Football, Tourism & Culture, Health & Development, Conflict & Security, Environment & Climate, or Other
- **sentiment** : the overall sentiment of the article on Ethiopia: Positive, Neutral, or Negative

Articles are processed in batches of 10 and labelled year by year, with each year's results saved to `data/progress_labels_ethiopia/` so the script can be safely stopped and resumed without re-calling the API. The final labelled dataset is saved to `data/ethiopia_labelled.csv`.
    """)

    st.markdown("#### Tourist Arrivals Data")
    st.markdown("""
The tourism dataset was sourced from Kaggle: [Tourism Dataset by Mohamadreza Momeni](https://www.kaggle.com/datasets/imtkaggleteam/tourism?select=26-+international-arrivals-for-personal-vs-business-and-professional-reasons.csv).

To obtain it:
1. Create a free Kaggle account at [kaggle.com](https://kaggle.com)
2. Navigate to the dataset linked above
3. Download the file `26- international-arrivals-for-personal-vs-business-and-professional-reasons.csv`
4. Place it in the `data/` folder of this project

**Important note on the arrivals figure:** The dataset reports tourist arrivals split into personal and business/professional reasons. The total arrivals figure used in this project is the **combined total** of both personal and business visitors, as both contribute to the overall tourism footprint of the country.

During cleaning (`prepare_ethiopia.py`), the full dataset, which covers many countries, is filtered to **Ethiopia only**, and further narrowed to the years **1996 to 2021** to match The Guardian article data. Any rows with missing arrival counts are dropped. The cleaned tourism data is then merged with the article summary data (one row per year) to produce the final analysis file at `data/ethiopia_analysis.csv`.
    """)

    st.markdown("---")
    st.caption(f"Loaded {len(df)} years of data ({year_min} to {year_max}). All regression results show association, not causation.")

# ── Q1: Coverage over time ────────────────────────────────────────────────────
with tabs[1]:
    st.header("Q1. How has The Guardian coverage of Ethiopia changed over time?")
    st.markdown(
        "Each bar shows the number of The Guardian articles mentioning Ethiopia in that year, "
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
            "Mention only articles: <b>%{y}</b><br>"
            "Total articles: %{customdata[0]}<extra></extra>"
        ),
    ))

    fig.update_layout(
        barmode="stack",
        title="The Guardian articles mentioning/covering Ethiopia per year",
        xaxis_title="Year",
        yaxis_title="Number of articles",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=420,
        xaxis=dict(dtick=1, tickangle=45),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Coverage trend line
    st.subheader("Coverage trend: linear regression over time")

    coverage_choice = st.radio(
        "Show trend for:",
        ["Focused articles (Ethiopia is main subject)", "Total articles (all mentions)"],
        horizontal=True,
        key="q1_coverage_choice",
    )

    if coverage_choice.startswith("Focused"):
        st.markdown(
            "Fitting a line through **focused** articles per year , articles where Ethiopia "
            "is the primary subject , tells us whether in-depth coverage has been increasing "
            "or decreasing on average. As you can see the slope is going upwards, meaning their articles focusing on Ethiopia is increasing with roughly 2 more focused articles being written every year(m~2). And since the P value is positive and less than 0.05, this positive association is statistically significant, meaning it's unlikely to be due to random chance."
            "One thing t"
        )
        regression_plot(
            x=df["year"], y=df["focused_articles"],
            x_label="Year", y_label="Focused articles",
            title="Focused The Guardian coverage of Ethiopia over time",
            year_labels=df["year"],
        )
        peak_year = df.loc[df["focused_articles"].idxmax(), "year"]
        peak_n    = int(df["focused_articles"].max())
        peak_label = "Peak focused coverage year"
    else:
        st.markdown(
            "Fitting a line through **total** articles per year , including passing mentions and focused articles , "
            "tells us whether overall coverage volume has been increasing or decreasing on average. Which can indicate if Ethiopia has at least been consistently including in the news, even if not always as the main subject. As you can see the slope is going upwards, meaning their articles mentioning or focusing on Ethiopia is increasing with roughly 16 more articles per year(m~16). And since the P value is positive and less than 0.05, this positive association is statistically significant, meaning it's unlikely to be due to random chance."
        )
        regression_plot(
            x=df["year"], y=df["total_articles"],
            x_label="Year", y_label="Total articles",
            title="Total The Guardian coverage of Ethiopia over time",
            year_labels=df["year"],
        )
        peak_year = df.loc[df["total_articles"].idxmax(), "year"]
        peak_n    = int(df["total_articles"].max())
        peak_label = "Peak total coverage year"

    # Key stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(peak_label, peak_year, f"{peak_n} articles")
    with col2:
        st.metric("Total articles (all years)", int(df["total_articles"].sum()))
    with col3:
        st.metric("Total focused articles", int(df["focused_articles"].sum()))
# ── Q2: Sentiment over time ───────────────────────────────────────────────────
with tabs[2]:
    st.header("Q2. How has sentiment of articles on Ethiopia changed over time?")
    st.markdown(
        "**Average sentiment score** is calculated as: Positive=+1, Neutral=0, Negative=−1. "
        "A score above zero means more positive articles than negative in that year."
        "As you can see below the average sentiment is consistently below zero meaning there are more negative aricles than positive ones. But if you check the bar chart below, you can see there are more neutral articles being written rather than negative or positive ones."
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
        title="Average sentiment score of Articles on Ethiopia per year",
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
    st.subheader("Sentiment trend: linear regression over time")
    st.markdown(
        "As you can see, the slope is nearly "
        "flat (m ≈ −0.0002) and the R² is essentially zero, meaning year alone explains almost "
        "none of the variation in sentiment. The p-value of 0.91 is far above 0.05, so this "
        "the trend is **not statistically significant** and we cannot conclude that sentiment "
        "of articles on Ethiopia in The Guardian has meaningfully changed over the period studied."
    )
    regression_plot(
        x=df["year"], y=df["average_sentiment_score"],
        x_label="Year", y_label="Average sentiment score",
        title="Has sentiment of articles on Ethiopia changed over time?",
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
    st.header("Q3. Which topics dominate Ethiopia related coverage?")
    st.markdown(
        "NB: Topic rates is the number of **articles on a certain topic** divided by the number of **focused articles** that year , "
        "articles where Ethiopia is the main subject. "
        "Articles where Ethiopia is mentioned only briefly are not included."
    )

    if not topic_count_cols:
        st.info("No topic data found. Check that label_ethiopia.py ran successfully.")
    else:
        # Overall topic totals (bar chart) with multi-year filter
        available_years = sorted(df["year"].dropna().astype(int).tolist())
        selected_years = st.multiselect(
            "Filter by year (select one or more, or leave empty to show all years):",
            options=available_years,
            default=[],
            key="q3_year_multiselect",
        )

        if not selected_years:
            filtered_df = df
            bar_title = "Total focused articles per topic (all years)"
        else:
            filtered_df = df[df["year"].isin(selected_years)]
            if len(selected_years) == 1:
                bar_title = f"Focused articles per topic in {selected_years[0]}"
            else:
                year_range = f"{min(selected_years)}-{max(selected_years)}" if selected_years == list(range(min(selected_years), max(selected_years)+1)) else ", ".join(str(y) for y in sorted(selected_years))
                bar_title = f"Focused articles per topic ({year_range})"

        topic_totals = filtered_df[topic_count_cols].sum().reset_index()
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
            title=bar_title,
            xaxis_title="Number of focused articles",
            height=400,
        )
        st.plotly_chart(fig3a, use_container_width=True)

        # Topic rates over time (line chart - pick topic)
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
            title=f"{selected_topic_label} - share of focused articles per year",
            xaxis_title="Year", yaxis_title="Topic rate",
            height=350, xaxis=dict(dtick=1, tickangle=45),
        )
        st.plotly_chart(fig3b, use_container_width=True)

        # Heatmap: topic × year
        st.subheader("Topic heatmap - all topics across all years")
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
Finally for the major question, "Does number of articles correlate with numer of tourist arrivals?" we will plot number of articles(focused/total) versus number of tourist arrivals and do a linear regression.
Considering that most people do not visit a country right after they read about  it, rather they take time to explore further before doing we will be using a **one-year lag** so coverage in year **t** is compared with tourism in year **t+1**.

Each data point is one year. The fitted line and its slope tell us the direction of the relationship. Read below the plots for the interpretation of the results.
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
    st.markdown("""
For the regression results using focused articles vs number of tourist arrivals, we have a slope of roughly 7859 with a p-value os 0.0011 which means that for an increase of 1 article focusing on Ethiopia, we see an increase of about 7859 tourist arrivals the following year. And since the p-value is less than 0.05, this positive association is statistically significant, meaning it's unlikely to be due to random chance.
Same goes for total articles, we have a slope of roughly 1233 with a p-value well less than 0.05 which means that for an increase of 1 article mentioning Ethiopia, we see an increase of about 1233 tourist arrivals the following year. 
But does this mean that the more they write about us, the more tourists we get? Not necessarily. We can only say that there is a positive association between the two, but we cannot conclude causation from this analysis alone. There could be other factors at play, such as global events, economic conditions, or changes in travel trends that influence both media coverage and tourism independently. 
Also our source of articles is only The Guardian, which does not fully represent global media coverage. So while the association is interesting and suggests a potential link, we would need more comprehensive data and analysis to draw stronger conclusions about causality.
This is just a starting point for a project I have wanted to do for a while, and scaling it up to cover more articles from different sources and more factors would be a great next step to explore these relationships further.""")
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
    st.subheader("Tourist arrivals over time - for context")
    fig4b = go.Figure()
    fig4b.add_trace(go.Scatter(
        x=df["year"], y=df["tourist_arrivals"],
        mode="lines+markers",
        line=dict(color="#E65100", width=2), marker=dict(size=7),
        hovertemplate="<b>%{x}</b><br>Tourist arrivals: %{y:,.0f}<extra></extra>",
    ))
    fig4b.update_layout(
        title="Ethiopia international tourist arrivals 1996-2021",
        xaxis_title="Year", yaxis_title="Tourist arrivals",
        height=350, xaxis=dict(dtick=1, tickangle=45),
    )
    st.plotly_chart(fig4b, use_container_width=True)

# ── Q5: Negative coverage → next-year tourism ────────────────────────────────
with tabs[5]:
    st.header("Q5. Do years with more negative coverage correspond to lower tourism the following year?")
    st.markdown("""
A negative slope here would mean that years when The Guardian coverage was more negative
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
    st.caption("This is less rigorous than the lagged version above - shown only for comparison.")
    regression_plot(
        x=df["negative_rate"],
        y=df["tourist_arrivals"],
        x_label="Negative article rate (year t)",
        y_label="Tourist arrivals (year t)",
        title="Same-year: negative rate vs tourist arrivals",
        year_labels=df["year"],
    )

    st.markdown("---")
    st.subheader("Q5 Findings")
    st.markdown("""
Based on the number of negative articles vs number of tourists, the result seems to be opposite of what you would expect. The slope is roughly 7029 which would indicate that for an increase of 1 negative article there is an additional 7029 tourists visiting next year. But one point we have to consider is years with more negative coverage also tend to be years with more total coverage overall (as seen in Q4), and more coverage of any kind correlates with more tourism so it could be confounding the result.

So to check even further see the negative rate vs number of tourists graph, where negative rate is the number of negatively written articles divided by the total number of articles that year. As you can see, the slope of that graph is negative meaning the more the share of negative articles increases the fewer the number of visitors next year. But the p-value is well over 0.05, meaning the relationship is not statistically significant enough.

In short: **negative coverage alone does not appear to deter tourism** in this dataset, at least not in a way that is statistically separable from overall coverage volume.
    """)

# ── Q6: Positive coverage → next-year tourism ────────────────────────────────
with tabs[6]:
    st.header("Q6. Do years with more positive coverage correspond to higher tourism the following year?")
    st.markdown("""
A positive slope here would mean that years when The Guardian coverage was more positive
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

    st.markdown("---")
    st.subheader("Q6 Findings")
    st.markdown("""
The positive rate shows a weak and generally non-significant association with number of tourists in the following year when we see the positive rate vs number of tourist arrival graph. But the number of positive articles vs number of tourist arrivals shows a more significant association, with an increase of 1 positive article being correlated with an increase of roughly 16,168 tourist arrivals. But again this is likely driven by the correlation between volume of coverage and tourism seen in Q4.

This suggests that **it is the volume of coverage rather than its positivity that is associated with higher tourism**, at least within the range and source of data available here.
    """)

# ── Conclusion ────────────────────────────────────────────────────────────────
with tabs[7]:
    st.header("Final Conclusions")
    st.markdown("""
This project set out to answer whether The Guardian's media coverage of Ethiopia is associated with how many tourists the country receives. Here is a summary of what was found:

| Question | Finding |
|---|---|
| Q1: Has coverage grown? | Yes, significantly. Both total and focused articles mostly trend upward over 1996-2021. |
| Q2: Has sentiment changed? | No. Sentiment is mostly mildly and consistently negative throughout, with no significant trend. But when observing the number of articles, most of them are written in a neutral tone. |
| Q3: Which topics dominate? | Politics & Governance and Conflict & Security dominate, with Health & Development following. |
| Q4: Does coverage volume predict tourism? | More coverage correlates with more arrivals the following year, but we can't fully determine causation due to the limitations of our dataset. |
| Q5: Does negative coverage deter tourism? | Not detectably. Negative rate alone does not show a significant negative effect based on our data. |
| Q6: Does positive coverage attract tourism? | Again not detectably beyond the effect of coverage volume overall. |

**The most important finding** is that overall coverage volume, regardless of sentiment, has the strongest correlation with next-year tourist arrivals in this dataset. This may reflect a simple awareness effect: the more Ethiopia appears in The Guardian, the more it is on the radar of potential visitors.

---
**Limitations to keep in mind:**
- All results show association, not causation. We cannot conclude The Guardian's articles caused the tourism arrivals.
- Data is limited to one outlet (The Guardian), which is only one site and does not fully represent global media at all.
- Simple linear regression does not control for confounding variables such as economic conditions, regional stability, or global travel trends.
- Tourism data combines personal and business arrivals, which may respond differently to media coverage.

---
**Next steps** that would strengthen this analysis:
- Expand to multiple news sources across different countries and languages.
- Include additional variables (GDP, flight routes, regional conflict indicators) in a multivariate model.
- Use more precise sentiment labelling criteria in the GPT prompt.
- Separate leisure from business tourism in the arrivals data.

This is a starting point for a project I have wanted to pursue for a while. The association between coverage and tourism is real and worth exploring further with richer data.
    """)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Data: The Guardian API · International tourist arrivals (Our World in Data). "
    "All regression results show association, not causation."
)
