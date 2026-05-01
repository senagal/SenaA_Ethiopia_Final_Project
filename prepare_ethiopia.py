"""
prepare_ethiopia.py
===================
Combines the labelled articles with tourist arrivals data
into one summary file — one row per year.

Output: data/ethiopia_analysis.csv

Requirements:
    pip install pandas numpy

Expects:
    data/ethiopia_labelled.csv   — from label_ethiopia.py
    data/<tourism file>.csv      — set TOURISM_FILE below
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── Set your tourism file name here ──────────────────────────────────────────
TOURISM_FILE = Path("data/26- international-arrivals-for-personal-vs-business-and-professional-reasons.csv")

# Column names in the tourism file — change if yours are different
COUNTRY_COL  = "Entity"
YEAR_COL     = "Year"
ARRIVALS_COL = "Tourist arrival"
# ─────────────────────────────────────────────────────────────────────────────

ARTICLES_CSV = Path("data/ethiopia_labelled.csv")
OUTPUT_CSV   = Path("data/ethiopia_analysis.csv")

# ── Step 1: Load articles ─────────────────────────────────────────────────────
print("Loading articles...")
articles = pd.read_csv(ARTICLES_CSV)
articles["year"]      = pd.to_numeric(articles["year"], errors="coerce")
articles["is_focus"]  = articles["is_focus"].astype(str).str.lower().isin(["true", "1", "yes"])
articles["sentiment"] = articles["sentiment"].astype(str).str.strip().str.capitalize()
articles["sentiment_score"] = articles["sentiment"].map(
    {"Positive": 1, "Neutral": 0, "Negative": -1}
)
articles = articles.dropna(subset=["year"])
articles["year"] = articles["year"].astype(int)
articles = articles[(articles["year"] >= 1995) & (articles["year"] <= 2021)]
print(f"  {len(articles)} articles loaded")

# ── Step 2: Summarise articles per year ───────────────────────────────────────
print("Summarising by year...")
summary = articles.groupby("year").agg(
    total_articles          = ("year",            "count"),
    focused_articles        = ("is_focus",         "sum"),
    average_sentiment_score = ("sentiment_score",  "mean"),
    positive_articles       = ("sentiment", lambda x: (x == "Positive").sum()),
    neutral_articles        = ("sentiment", lambda x: (x == "Neutral").sum()),
    negative_articles       = ("sentiment", lambda x: (x == "Negative").sum()),
).reset_index()

summary["focus_rate"]    = summary["focused_articles"]  / summary["total_articles"]
summary["positive_rate"] = summary["positive_articles"] / summary["total_articles"]
summary["neutral_rate"]  = summary["neutral_articles"]  / summary["total_articles"]
summary["negative_rate"] = summary["negative_articles"] / summary["total_articles"]

# ── Step 3: Count articles per topic (focused only) ───────────────────────────
topic_counts = (
    articles[articles["is_focus"]]
    .groupby(["year", "topic"])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)

# Clean column names: "Politics & Governance" -> "politics_and_governance_count"
import re
topic_counts.columns = [
    "year" if c == "year"
    else re.sub(r"[^a-z0-9]+", "_", c.lower().replace("&", "and")).strip("_") + "_count"
    for c in topic_counts.columns
]

summary = summary.merge(topic_counts, on="year", how="left")

# Fill any missing topic counts with 0
count_cols = [c for c in summary.columns if c.endswith("_count")]
summary[count_cols] = summary[count_cols].fillna(0).astype(int)

# Calculate topic rates (share of focused articles)
for col in count_cols:
    rate_col = col.replace("_count", "_rate")
    summary[rate_col] = np.where(
        summary["focused_articles"] > 0,
        summary[col] / summary["focused_articles"],
        0,
    )

# ── Step 4: Load tourism data ─────────────────────────────────────────────────
print("Loading tourism data...")
tourism = pd.read_csv(TOURISM_FILE)
tourism = tourism.rename(columns={
    COUNTRY_COL:  "country",
    YEAR_COL:     "year",
    ARRIVALS_COL: "tourist_arrivals",
})
tourism["tourist_arrivals"] = pd.to_numeric(
    tourism["tourist_arrivals"].astype(str).str.replace(",", "", regex=False),
    errors="coerce",
)
tourism["year"] = pd.to_numeric(tourism["year"], errors="coerce").astype("Int64")
tourism = tourism[
    (tourism["country"].str.strip() == "Ethiopia") &
    (tourism["year"] >= 1995) & (tourism["year"] <= 2021)
][["year", "tourist_arrivals"]].dropna()
tourism["year"] = tourism["year"].astype(int)
print(f"  {len(tourism)} years of tourism data loaded")

# ── Step 5: Merge and add lagged tourism ─────────────────────────────────────
print("Merging...")
df = summary.merge(tourism, on="year", how="left")
df = df.sort_values("year").reset_index(drop=True)

# Lagged tourism: tourist arrivals in year t+1
# Used in Q4, Q5, Q6 to test whether this year's coverage
# predicts NEXT year's tourism
df["tourist_arrivals_next_year"] = df["tourist_arrivals"].shift(-1)

df = df.replace([np.inf, -np.inf], np.nan)

# ── Step 6: Save ──────────────────────────────────────────────────────────────
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved {len(df)} rows to {OUTPUT_CSV}")
print(f"\nYears covered: {df['year'].min()} – {df['year'].max()}")
print(f"\nColumns: {list(df.columns)}")
print("\nNext step: streamlit run streamlit_ethiopia.py")
