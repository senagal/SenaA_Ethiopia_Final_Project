# Guardian Coverage of Ethiopia (1996–2021)

> **Does international media portrayal correlate with tourist arrivals?**

This project analyses how The Guardian has covered Ethiopia over 25 years and whether that coverage is associated with the number of tourists Ethiopia receives.

---

## Project Overview

Starting from the question *"How is Ethiopia being portrayed in international media, and does it have any tangible effect on our economy?"*, this project collects Guardian articles mentioning Ethiopia (1996–2021), labels them using GPT-4o-mini, and compares coverage patterns against tourist arrival data.

**Six research questions are explored:**

1. How has the quantity of The Guardian's coverage of Ethiopia changed over time?
2. How has the sentiment of their articles on Ethiopia changed over time?
3. Which topics dominate Ethiopia-related coverage?
4. Do years with more/less focused coverage correlate with higher or lower tourist arrivals the following year?
5. Do years with more/less negative coverage correspond to lower tourism the following year?
6. Do years with more/less positive coverage correspond to higher or lower tourism the following year?

---

## Pipeline

Run these four scripts in order. Each one produces a file that the next one reads.

```
fetch_ethiopia.py
      |
      v
ethiopia_raw.json          (~9,000 articles, 1996-2021)
      |
      v
label_ethiopia.py
      |
      v
ethiopia_labelled.csv      (topic, sentiment, is_focus per article)
      |
      v
prepare_ethiopia.py  <---  tourism CSV (from Kaggle)
      |
      v
ethiopia_analysis.csv      (one row per year, ready to analyse)
      |
      v
streamlit run streamlit.py
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up API keys

Create a `.env` file in the project root:

```
GUARDIAN_API_KEY=your_guardian_key_here
OPENAI_API_KEY=your_openai_key_here
```

- Get a free Guardian API key at [open-platform.theguardian.com](https://open-platform.theguardian.com/)
- Get an OpenAI API key at [platform.openai.com](https://platform.openai.com/)

### 3. Download the tourism data

1. Create a free Kaggle account at [kaggle.com](https://kaggle.com)
2. Download the file from [this dataset](https://www.kaggle.com/datasets/imtkaggleteam/tourism?select=26-+international-arrivals-for-personal-vs-business-and-professional-reasons.csv)
3. Place it in the `data/` folder as-is (do not rename it)

> **Note on arrivals figures:** The dataset splits arrivals into personal and business/professional reasons. This project uses the **combined total** of both as the overall tourist arrivals figure.

---

## Running the Pipeline

### Step 1 — Fetch articles

```bash
python fetch_ethiopia.py
```

Queries The Guardian API for any article containing "Ethiopia", "Ethiopian", or "Ethiopians". Fetches in 30-day chunks from 1996 to 2021 and saves progress to `data/progress_ethiopia/` so it can be safely stopped and restarted.

**Output:** `data/ethiopia_raw.json`

### Step 2 — Label articles

```bash
python label_ethiopia.py
```

Sends each article to GPT-4o-mini in batches of 10. Each article is labelled with:

- `is_focus` — is Ethiopia the primary subject, or just mentioned in passing?
- `topic` — one of: Politics & Governance, Economy & Business, Sport & Football, Tourism & Culture, Health & Development, Conflict & Security, Environment & Climate, or Other
- `sentiment` — the overall sentiment of the article on Ethiopia: Positive, Neutral, or Negative

Saves progress per year to `data/progress_labels_ethiopia/` so it can be stopped and resumed without re-calling the API.

**Output:** `data/ethiopia_labelled.csv`

### Step 3 — Prepare analysis data

```bash
python prepare_ethiopia.py
```

Aggregates labelled articles into one row per year, calculates sentiment rates, topic counts, and merges with the tourism CSV. Also adds a one-year lagged tourism column for the regression analysis.

**Output:** `data/ethiopia_analysis.csv`

### Step 4 — Run the dashboard

```bash
streamlit run streamlit.py
```

Opens an interactive dashboard with tabs for each research question, including regression plots and findings.

---

## Data Sources

| Data | Source |
|---|---|
| Guardian articles | [The Guardian Open Platform API](https://open-platform.theguardian.com/) |
| Tourist arrivals | [Kaggle — Tourism Dataset by Mohamadreza Momeni](https://www.kaggle.com/datasets/imtkaggleteam/tourism) |

---

## Project Files

| File | Purpose |
|---|---|
| `fetch_ethiopia.py` | Fetches articles from The Guardian API |
| `label_ethiopia.py` | Labels articles using GPT-4o-mini |
| `prepare_ethiopia.py` | Merges and aggregates data for analysis |
| `streamlit.py` | Interactive dashboard |
| `SenaAbdisa_Ethiopia_FinalProject.ipynb` | Jupyter notebook version of the analysis |
| `data/ethiopia_analysis.csv` | Final analysis-ready dataset (one row per year) |

---

## Notes

- All regression results show **association, not causation**.
- Article data is limited to The Guardian only and does not represent global media coverage.
- Tourism data covers personal and business arrivals combined.
