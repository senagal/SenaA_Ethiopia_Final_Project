"""
fetch_ethiopia.py
=================
Fetches all Guardian articles mentioning Ethiopia from 1995 to 2021.
Saves progress so it can be stopped and restarted safely.

Output: data/ethiopia_raw.json

Requirements:
    pip install requests python-dotenv

Add your Guardian API key to a .env file:
    GUARDIAN_API_KEY=your_key_here

Get a free key at: https://open-platform.theguardian.com/
"""

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
import requests

load_dotenv()

API_KEY    = os.getenv("GUARDIAN_API_KEY")
FROM_DATE  = datetime(1995, 1, 1)
TO_DATE    = datetime(2021, 12, 31)
CHUNK_DAYS = 30   # fetch 30 days at a time

DATA_DIR     = Path("data")
PROGRESS_DIR = DATA_DIR / "progress_ethiopia"
OUTPUT_FILE  = DATA_DIR / "ethiopia_raw.json"

DATA_DIR.mkdir(exist_ok=True)
PROGRESS_DIR.mkdir(exist_ok=True)


def date_chunks():
    """Split 1995–2021 into 30-day windows."""
    chunks, end = [], TO_DATE
    while end > FROM_DATE:
        start = max(end - timedelta(days=CHUNK_DAYS), FROM_DATE)
        chunks.append((start, end))
        end = start - timedelta(days=1)
    chunks.reverse()
    return chunks


def fetch_chunk(start, end):
    """Fetch one 30-day window from the Guardian API."""
    params = {
        "q":           '"Ethiopia" OR "Ethiopian" OR "Ethiopians"',
        "from-date":   start.strftime("%Y-%m-%dT00:00:00Z"),
        "to-date":     end.strftime("%Y-%m-%dT23:59:59Z"),
        "show-fields": "headline,trailText,bodyText,byline,wordcount,sectionName",
        "show-tags":   "keyword",
        "page-size":   200,
        "api-key":     API_KEY,
    }

    try:
        response = requests.get(
            "https://content.guardianapis.com/search",
            params=params,
            timeout=30,
        )
        time.sleep(0.5)   # be polite to the API
        data = response.json()

        if data.get("response", {}).get("status") != "ok":
            print(f"  API error: {data.get('message', 'unknown')}")
            return []

        articles = []
        for a in data["response"].get("results", []):
            f = a.get("fields", {})
            articles.append({
                "url":          a.get("webUrl", ""),
                "published_at": a.get("webPublicationDate", ""),
                "section":      f.get("sectionName", ""),
                "headline":     f.get("headline", ""),
                "summary":      f.get("trailText", ""),
                "body_text":    f.get("bodyText", ""),
                "author":       f.get("byline", ""),
                "wordcount":    f.get("wordcount", 0),
                "tags":         ", ".join(
                    t.get("webTitle", "") for t in a.get("tags", [])
                ),
            })
        return articles

    except (requests.ConnectionError, requests.Timeout) as e:
        print(f"  Network error: {e} — skipping chunk")
        return []


def main():
    chunks = date_chunks()
    print(f"Fetching Ethiopia articles 1995–2021")
    print(f"Total chunks: {len(chunks)}  ({CHUNK_DAYS} days each)\n")

    all_articles = []

    for i, (start, end) in enumerate(chunks, 1):
        # Each chunk is saved to its own file so we can resume if interrupted
        cache_file = PROGRESS_DIR / f"{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.json"

        if cache_file.exists():
            with open(cache_file) as f:
                articles = json.load(f)
            print(f"  [{i:>3}/{len(chunks)}] {start.strftime('%Y-%m-%d')}  {len(articles):>4} articles  (cached)")
        else:
            print(f"  [{i:>3}/{len(chunks)}] {start.strftime('%Y-%m-%d')}  fetching...", end="\r")
            articles = fetch_chunk(start, end)
            with open(cache_file, "w") as f:
                json.dump(articles, f)
            print(f"  [{i:>3}/{len(chunks)}] {start.strftime('%Y-%m-%d')}  {len(articles):>4} articles  (saved)  ")

        all_articles.extend(articles)

    # Remove duplicate articles (same URL)
    seen, unique = set(), []
    for a in all_articles:
        if a["url"] not in seen:
            seen.add(a["url"])
            unique.append(a)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(unique, f)

    print(f"\nDone. {len(unique)} unique articles saved to {OUTPUT_FILE}")
    print("Next step: run label_ethiopia.py")


if __name__ == "__main__":
    main()
