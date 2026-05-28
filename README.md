# ABSA Expert Chatbot — Assignment 3
**Aspect-Based Sentiment Analysis of News Articles**
Charles Darwin University — Bhandari, Patel, Rahaman Mohammed (2026)

---

## What this is

An **expert chatbot** implementing the assignment brief:
- Specialised domain knowledge on fake news detection, ABSA, and NLP
- Uses the ABSA system from the notebook as an API tool
- Handles casual conversation AND expert queries
- Built-in performance evaluation tab

---

## Setup & Run

```bash
# Install core dependencies
pip install -r requirements.txt

# Optional: URL fetching (Cell 44 of notebook)
pip install newspaper3k

# Optional: LSA summarisation (Cell 18 of notebook)
pip install sumy

# Run
streamlit run app.py
```

Opens at **http://localhost:8501**

---

## File structure

| File | Notebook cells it replicates |
|------|------------------------------|
| `absa_engine.py` | Cells 16, 18, 20, 22-23, 44, 46 — full ABSA pipeline |
| `chatbot_nlu.py` | Cells 47-48 — intent classifier + expert knowledge base |
| `app.py` | Streamlit UI — replaces Colab terminal chatbot |

---

## Features

### Chat tab
- Paste headline → full ABSA analysis (11 aspects, hybrid VADER scoring, fake/real)
- Type `url: https://...` → auto-fetches article (requires newspaper3k)
- Ask any question about the project methodology
- LIME-style word signal display
- Radar + bar chart visualisations
- Session history

### Evaluation tab
- Runs pipeline on 10 labelled QA headlines
- Reports accuracy, FAKE/REAL breakdown, per-category results
- Compares against notebook Model A (67.19%), B (66.86%), C (49%), Naive Bayes (64%)

---

## Deploy free (Streamlit Cloud)

1. Push folder to GitHub
2. Go to https://share.streamlit.io
3. Connect repo → select `app.py`
4. Get a public shareable URL for your marker
