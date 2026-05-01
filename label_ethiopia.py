"""
label_ethiopia.py
=================
Reads data/ethiopia_raw.json, sends each article to GPT to assign:
  - is_focus  : true/false — is Ethiopia the primary subject?
  - topic     : one of the topics (or Other)
  - sentiment : Positive / Neutral / Negative

Saves progress per year to data/progress_labels_ethiopia/
so the script can be stopped and restarted safely.

Final output: data/ethiopia_labelled.csv
(same columns as labelled_articles.csv from the 7-country project)

Usage:
    python label_ethiopia.py
"""

import json
import time
import os
import re
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR     = Path("data")
INPUT_FILE   = DATA_DIR / "ethiopia_raw.json"
OUTPUT_CSV   = DATA_DIR / "ethiopia_labelled.csv"
PROGRESS_DIR = DATA_DIR / "progress_labels_ethiopia"

PROGRESS_DIR.mkdir(exist_ok=True)

TOPICS = [
    "Politics & Governance",
    "Economy & Business",
    "Sport & Football",
    "Tourism & Culture",
    "Health & Development",
    "Conflict & Security",
    "Environment & Climate",
    "Other",
]

GPT_MODEL      = "gpt-4o-mini"
GPT_BATCH_SIZE = 10
GPT_RETRIES    = 3

# ── Prompt ────────────────────────────────────────────────────────────────────
def system_prompt():
    return f"""You are a news article classifier.

For each article about or mentioning Ethiopia, assign THREE labels:

1. is_focus — true if the article is PRIMARILY about Ethiopia, false if Ethiopia is only mentioned briefly or as context.

2. topic — the single best topic from this list:
   {", ".join(TOPICS)}

3. sentiment — the overall tone toward Ethiopia in the article:
   Positive, Neutral, or Negative

Input:  JSON array of objects with "id", "headline", "summary", and "body".
         Use all fields together to assign the labels.
Output: JSON array of objects with "id", "is_focus", "topic", "sentiment" ONLY.
Return ONLY the raw JSON array — no markdown, no explanation."""


# ── GPT batch call ────────────────────────────────────────────────────────────
def label_batch(batch):
    payload = json.dumps(
        [{"id":       r["id"],
          "headline": str(r.get("headline", "")),
          "summary":  str(r.get("summary", "") or ""),
          "body":     str(r.get("body_text", ""))[:2000]}
         for r in batch],
        ensure_ascii=False
    )

    for attempt in range(1, GPT_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt()},
                    {"role": "user",   "content": payload},
                ],
                temperature=0,
                max_tokens=4000,
            )
            raw = resp.choices[0].message.content.strip()

            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1].lstrip("json").strip()

            # Fix unquoted sentiment values GPT sometimes returns
            # e.g. "sentiment": Neutral  →  "sentiment": "Neutral"
            raw = re.sub(
                r':\s*(Positive|Negative|Neutral)([,\}])',
                r': "\1"\2',
                raw
            )

            return {str(r["id"]): r for r in json.loads(raw)}

        except Exception as e:
            if attempt < GPT_RETRIES:
                time.sleep(5)

    # Fallback on failure
    return {str(r["id"]): {"is_focus": False, "topic": "Other", "sentiment": "Neutral"}
            for r in batch}


# ── Label one year ────────────────────────────────────────────────────────────
def label_year(year, year_articles):
    cache_file = PROGRESS_DIR / f"{year}.json"

    # Load from cache if already done
    if cache_file.exists():
        with open(cache_file, encoding="utf-8") as f:
            cached = json.load(f)
        focused = sum(1 for a in cached if a.get("is_focus") is True)
        # Re-label if 0 focused and more than 10 articles — likely a failed run
        if focused == 0 and len(cached) >= 10:
            print(f"  {year}  {len(cached):>4} articles  (0 focused — re-labelling)")
            cache_file.unlink()
        else:
            print(f"  {year}  {len(cached):>4} articles  (cached, {focused} focused)")
            return cached

    print(f"  {year}  {len(year_articles):>4} articles  labelling...", flush=True)

    # Assign temp ids
    for i, a in enumerate(year_articles):
        a["_id"] = str(i)

    rows    = [{"id":       a["_id"],
                "headline": a.get("headline", ""),
                "summary":  str(a.get("summary", "") or ""),
                "body":     str(a.get("body_text", ""))[:2000]}
               for a in year_articles]
    batches = [rows[i:i + GPT_BATCH_SIZE] for i in range(0, len(rows), GPT_BATCH_SIZE)]

    labels = {}
    for idx, batch in enumerate(batches, 1):
        print(f"  {year}  batch {idx}/{len(batches)}", end="\r", flush=True)
        labels.update(label_batch(batch))
        time.sleep(0.3)

    topic_lower     = {t.lower(): t for t in TOPICS}
    valid_sentiment = {"positive", "negative", "neutral"}

    for a in year_articles:
        lbl = labels.get(a["_id"], {})

        raw_focus     = lbl.get("is_focus", False)
        a["is_focus"] = raw_focus if isinstance(raw_focus, bool) else str(raw_focus).lower() == "true"

        a["topic"]    = topic_lower.get(
            str(lbl.get("topic", "")).lower().strip(), "Other"
        )

        sent          = str(lbl.get("sentiment", "Neutral")).strip().capitalize()
        a["sentiment"]= sent if sent.lower() in valid_sentiment else "Neutral"

        a["country"]  = "Ethiopia"
        del a["_id"]

    # Save this year to cache
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(year_articles, f, ensure_ascii=False)

    focused = sum(1 for a in year_articles if a["is_focus"])
    print(f"  {year}  {len(year_articles):>4} articles · "
          f"{focused} focused · "
          f"top topic: {Counter(a['topic'] for a in year_articles).most_common(1)[0][0]}          ")

    return year_articles


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if not INPUT_FILE.exists():
        print(f"Input file not found: {INPUT_FILE}")
        print("Run fetch_ethiopia.py first.")
        return

    with open(INPUT_FILE, encoding="utf-8") as f:
        articles = json.load(f)

    print(f"Model: {GPT_MODEL} · Batch size: {GPT_BATCH_SIZE}")
    print(f"Loaded {len(articles)} articles\n")

    # Group by year
    by_year = {}
    for a in articles:
        year = a.get("published_at", "")[:4]
        by_year.setdefault(year, []).append(a)

    # Label year by year — saving progress after each year
    all_articles = []
    for year in sorted(by_year.keys()):
        all_articles.extend(label_year(year, by_year[year]))

    if not all_articles:
        print("No articles labelled.")
        return

    # Build final CSV — same columns as labelled_articles.csv
    df = pd.DataFrame(all_articles)

    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
    df["year"]         = df["published_at"].dt.year
    df["month"]        = df["published_at"].dt.to_period("M").astype(str)

    for col in ["headline", "summary", "author", "tags"]:
        if col in df.columns:
            df[col] = (df[col].astype(str)
                              .str.replace(r"[\r\n]+", " ", regex=True)
                              .str.strip()
                              .replace("nan", ""))

    df["wordcount"] = pd.to_numeric(
        df.get("wordcount", 0), errors="coerce"
    ).fillna(0).astype(int)

    # Same column order as labelled_articles.csv from the 7-country project
    cols = ["country", "is_focus", "topic", "sentiment",
            "published_at", "year", "month",
            "headline", "summary", "body_text",
            "author", "wordcount", "tags", "url", "section"]
    df = df[[c for c in cols if c in df.columns]].reset_index(drop=True)

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    print(f"\nSaved {len(df)} articles → {OUTPUT_CSV}")

    focused = df["is_focus"].sum()
    print(f"\nEthiopia  {len(df):>5} total  |  {focused:>4} focused")

    print(f"\nArticles per topic:")
    for topic, n in df["topic"].value_counts().items():
        print(f"  {topic:<30} {n}")

    print(f"\nSentiment breakdown:")
    print(df["sentiment"].value_counts().to_string())

    print(f"\nNext step: run prepare_ethiopia.py")


if __name__ == "__main__":
    main()
