"""
chatbot_nlu.py
Intent classifier + expert knowledge base.
All knowledge is grounded in Assignment3.ipynb content.
"""

import re
import random

# ── Intent patterns ───────────────────────────────────────────────────────────
INTENTS = {
    # Casual
    "greet":        [r"\b(hi|hello|hey|howdy|hiya|good morning|good afternoon|sup|yo)\b"],
    "how_are_you":  [r"\bhow are you\b", r"\bhow'?re (you|things)\b", r"\byou okay\b", r"\bwhat'?s up\b"],
    "who_are_you":  [r"\bwho are you\b", r"\bwhat are you\b", r"\byour name\b", r"\bintroduce\b"],
    "thanks":       [r"\b(thank|thanks|thank you|cheers|appreciated|thx|ty)\b"],
    "bye":          [r"\b(bye|goodbye|see you|cya|farewell|later|exit|quit)\b"],
    "help":         [r"\b(help|what can you do|commands|options|features|capabilities)\b"],

    # ABSA tool triggers
    "analyse": [
        r"\b(analys[ei]|check|test|scan|classify|evaluate|assess|predict|detect)\b.{0,30}\b(article|headline|news|text|sentence|this|it)\b",
        r"\b(is this|is it fake|is it real)\b",
        r"^(analyse|analyze|check|test|scan):?\s+\S.{10,}",
        r"\burl:\s*https?://",
        r"\bhttps?://\S+",
    ],

    # Domain knowledge
    "what_is_absa":     [r"\bwhat is absa\b", r"\babsa.{0,20}(mean|stand|explain|work)\b"],
    "what_is_vader":    [r"\bvader\b", r"\bhow.{0,10}vader\b", r"\bwhat is vader\b"],
    "what_is_bert":     [r"\bbert\b", r"\broberta\b", r"\btransformer\b"],
    "hybrid":           [r"\bhybrid\b", r"\bvader.{0,15}bert\b", r"\bbert.{0,15}vader\b", r"\bcombine.{0,20}sentiment\b"],
    "aspects":          [r"\b(aspect|categor|topic)s?\b", r"\b11 aspect\b", r"\blist.{0,10}aspect\b", r"\bwhich aspect\b"],
    "fake_real":        [r"\bfake.{0,10}real\b", r"\bhow.{0,10}classif\b", r"\bfake news detect\b"],
    "datasets":         [r"\bdatasets?\b", r"\bliar\b", r"\bfakenewsnet\b", r"\btraining data\b", r"\bhow.{0,10}train\b"],
    "models":           [r"\bmodel [abcABC]\b", r"\baccuracy\b", r"\blogistic regression\b", r"\btf.?idf\b", r"\bnaive bayes\b", r"\bsmote\b"],
    "lime":             [r"\blime\b", r"\bexplain(ability)?\b", r"\binterpret\b", r"\bwhich word\b"],
    "preprocessing":    [r"\bpreprocess\b", r"\bclean(ing)?\b", r"\blemm\b", r"\bstopword\b", r"\btokeniz\b"],
    "summarisation":    [r"\bsummar\b", r"\blsa\b", r"\b512 token\b", r"\barticle.{0,10}long\b"],
    "url_fetch":        [r"\burl\b", r"\bfetch\b", r"\bnewspaper\b", r"\blink\b", r"\bwebsite\b"],
    "subjectivity":     [r"\bsubjectiv\b", r"\bopinion\b", r"\bobjective\b"],
    "ethics":           [r"\bethic\b", r"\bbias\b", r"\brisk\b", r"\bfair\b", r"\bprivacy\b", r"\bfalse positive\b"],
    "limitations":      [r"\blimit\b", r"\bweakness\b", r"\bfuture work\b", r"\bimprove\b"],
    "project_overview": [r"\bproject\b", r"\bwhat.{0,15}(chatbot|system|app|built)\b", r"\boverview\b"],
    "contributions":    [r"\bwho (built|made|wrote|coded)\b", r"\bcontribut\b", r"\bteam\b", r"\bauthor\b"],
    "notebook":         [r"\bnotebook\b", r"\bcolab\b", r"\bjupyter\b", r"\bcode\b"],
}

# ── Casual replies ────────────────────────────────────────────────────────────
CASUAL = {
    "greet": [
        "Hey! I'm the ABSA Fake News Expert — paste a headline or article to analyse, or ask me anything about the project.",
        "Hello! I can check if news is fake, score sentiment per topic, or answer questions about the ABSA system.",
        "Hi there! Drop a news headline or full article and I'll run the full analysis.",
    ],
    "how_are_you": [
        "Running great! I've processed thousands of headlines. What would you like to check?",
        "All good — ready to analyse news. Paste a headline or article!",
        "Fully operational. Ask me to analyse something or explain the methodology.",
    ],
    "thanks": [
        "Happy to help! Any other articles to check?",
        "You're welcome. Paste another article anytime.",
        "No problem! Stay critical of the news you read.",
    ],
    "bye": [
        "Goodbye! Stay skeptical of what you read online.",
        "See you! Remember: high subjectivity + emotional framing = worth questioning.",
        "Bye! Keep thinking critically.",
    ],
}

# ── Expert knowledge base (grounded in notebook cells) ───────────────────────
KB = {
    "what_is_absa": """**Aspect-Based Sentiment Analysis (ABSA)** goes beyond simple positive/negative scoring.

Instead of saying *"this article is negative overall"*, ABSA asks:
- Negative about *what*? Government? Health? The economy?

**How it works in the notebook (Cell 22-23):**
1. Split the article into sentences
2. Check each sentence for aspect keywords (e.g. "vaccine" → health aspect)
3. Score that sentence's sentiment using the hybrid VADER+BERT approach
4. Average all matching sentence scores for that aspect

**Example:**
> *"The government handled the economy well but its healthcare policy failed"*
- government → **positive**
- economy → **positive**
- health → **negative**

This project applies ABSA across **11 aspect categories** to detect how news articles frame each topic emotionally.""",

    "what_is_vader": """**VADER** (Valence Aware Dictionary and sEntiment Reasoner) — Cell 5 of the notebook.

VADER is a **rule-based** tool that:
- Looks up each word in a pre-built sentiment dictionary
- Handles **punctuation** (GREAT!!! scores higher than great)
- Handles **capitalisation** and emoticons
- Outputs a **compound score** between -1.0 and +1.0

**Thresholds used in the notebook:**
- ≥ 0.05 → positive
- ≤ -0.05 → negative
- Between → neutral

**Why VADER is used first (Cell 23):**
It's extremely fast — no GPU needed. The notebook runs VADER on every sentence, then only calls BERT when VADER is uncertain (compound between -0.1 and 0.1).

**VADER's weakness:** It fails on sarcasm. The notebook demonstrates this (Cell 5):
> *"Oh great, ANOTHER government scandal. What a surprise."*
VADER scores this **positive** (sees "great") — BERT gets it right.""",

    "what_is_bert": """**BERT** — Bidirectional Encoder Representations from Transformers (Cells 4-5 of the notebook).

Model used: **`cardiffnlp/twitter-roberta-base-sentiment-latest`**
- RoBERTa-based (improved BERT architecture)
- Trained on Twitter data — similar informal style to news headlines
- Reads sentences **bidirectionally** — understands full context

**Why Twitter-trained?** The research group cardiffnlp found Twitter data matches the informal, headline-style language of news articles better than formal text.

**Model C in the notebook (Cell 32):** Also tested `hamzab/roberta-fake-news-classification` — a RoBERTa model fine-tuned specifically for fake news detection on a 200-article sample → **49% accuracy** (near random), showing domain specificity issues.

**BERT's 512 token limit** is why the notebook adds LSA summarisation (Cell 18) for long articles.""",

    "hybrid": """**Hybrid VADER + BERT Approach — Cell 23 of the notebook:**

```
For each sentence:
  1. Run VADER (fast, always)
  2. If |VADER compound| > 0.1:
       → VADER is confident → use VADER alone
  3. If |VADER compound| <= 0.1:
       → VADER uncertain → run BERT
       → Final = 70% BERT + 30% VADER
```

**Why the 0.1 threshold (not 0.3)?**
The notebook notes this was a deliberate tuning choice — a tighter threshold *significantly cuts BERT calls* while still catching the genuinely ambiguous sentences.

**Why 70/30 weighting?**
BERT understands context; VADER knows sentiment intensity markers (!!!, CAPS). The 70/30 split trusts BERT more while keeping VADER's intensity signals.""",

    "aspects": """**11 Aspect Categories — Cell 20 of the notebook:**

| # | Aspect | Example keywords |
|---|--------|-----------------|
| 1 | Government | president, congress, senate, federal, policy |
| 2 | Economy | inflation, GDP, unemployment, tax, market |
| 3 | Health | vaccine, pandemic, hospital, COVID, FDA |
| 4 | Immigration | border, refugee, deportation, asylum, ICE |
| 5 | Election | vote, ballot, campaign, electoral fraud |
| 6 | Media | journalist, propaganda, censorship, mainstream |
| 7 | Climate | emissions, global warming, fossil fuel, EPA |
| 8 | Crime | police, shooting, arrest, FBI, homicide |
| 9 | Education | school, university, tuition, curriculum |
| 10 | Military | army, war, NATO, troops, drone, Pentagon |
| 11 | Technology | AI, surveillance, 5G, cybersecurity, hack |

**Why 11?** The original notebook had 6 (government, economy, health, immigration, election, media). The improvement (Cell 19) added **climate, crime, education, military, technology** because fake news frequently covers these topics.""",

    "fake_real": """**Fake / Real Classification — Cells 29-38:**

**Three models compared:**

| Model | Approach | Accuracy |
|-------|----------|----------|
| A | TF-IDF only (Logistic Regression) | **67.19%** |
| B | TF-IDF + 11 ABSA sentiment scores | **66.86%** |
| C | RoBERTa fake news BERT (200 sample) | **49%** |

Also tested:
| Naive Bayes original | ~64% | Fake recall only 10% |
| Naive Bayes balanced | 64% | Fake recall 55% |
| Logistic Regression + SMOTE | similar to A | Fake recall 49% |

**Key finding:** Adding sentiment features **did not improve** classification. Factual accuracy and emotional tone are largely independent in news text.""",

    "datasets": """**Two datasets — Cells 7-14:**

**FakeNewsNet (Cells 7-8):**
- 422 full news articles
- 4 CSV files: BuzzFeed_fake, BuzzFeed_real, PolitiFact_fake, PolitiFact_real
- Labelled by professional fact-checkers

**LIAR (Cell 9 — HuggingFace `UKPLab/liar`):**
- 12,836 short political statements
- Converted to binary: pants-fire/false/barely-true → FAKE, rest → REAL

**Critical bug fixed (Cell 10-12):**
The original code applied the label conversion *twice*, flipping all labels to FAKE. Cell 12 removes the duplicate call.

**Combined (Cell 14):** 13,258 articles — 3,772 fake, 9,486 real.""",

    "models": """**All models from the notebook (Cells 29-38):**

**Model A — TF-IDF + Logistic Regression (Cell 30):**
- TF-IDF: max_features=15,000, ngram_range=(1,2), min_df=3, sublinear_tf=True
- class_weight='balanced' to handle 3:7 fake:real imbalance
- Accuracy: **67.19%**, ROC-AUC: **0.6806**

**Model B — TF-IDF + Sentiment + Logistic Regression:**
- Adds all 11 ABSA scores as extra features
- Accuracy: **66.86%**, ROC-AUC: **0.6778**

**Model C — RoBERTa (Cell 32):**
- `hamzab/roberta-fake-news-classification`
- Accuracy: **49%** — domain specificity problem

**Naive Bayes (Cells 33-36):** 64% accuracy, only 10% fake recall

**SMOTE oversampling (Cell 37):** Similar accuracy to Model A, fake recall ~49%""",

    "lime": """**LIME Explainability — Cells 41-42:**

LIME = Local Interpretable Model-agnostic Explanations.

Shows *which words* drove each prediction, making the model transparent.

**How it works (Cell 42):**
1. Takes one article prediction
2. Slightly changes the input (removes/replaces words)
3. Re-predicts each variation
4. Words whose removal most changes the prediction = most important

**Example results from the notebook:**
- Conspiracy headline → FAKE: *vaccine, mainstream, controlling* pushed prediction to FAKE
- Financial headline → REAL: *federal, reserve, percent, rate* pushed prediction to REAL""",

    "preprocessing": """**Text Cleaning Pipeline — Cell 16:**

```python
def clean_text(text):
    text = text.lower()                        # 1. Lowercase
    text = re.sub(r'http\\S+', '', text)       # 2. Remove URLs
    text = re.sub(r'[^a-z\\s]', ' ', text)    # 3. Remove punctuation
    text = re.sub(r'\\s+', ' ', text).strip() # 4. Remove extra spaces
    words = [lemmatizer.lemmatize(w)           # 5. Lemmatise
             for w in words
             if w not in stop_words            # 6. Remove stopwords
             and len(w) > 2]                   # 7. Remove short tokens
    return ' '.join(words)
```

**Title doubling (Cell 16):**
`combined = clean_title + ' ' + clean_title + ' ' + clean_text`
Headlines get 2x weight in TF-IDF because they are the most information-dense part.""",

    "summarisation": """**LSA Article Summarisation — Cell 18:**

For articles over **200 words**, the notebook compresses them to 8 sentences using LSA (Latent Semantic Analysis) before running ABSA.

**Why:**
- BERT has a **512 token limit**
- Long articles are slow to process
- LSA picks the most semantically important sentences using SVD

**Library:** `sumy` (LsaSummarizer). Falls back to first N sentences if sumy fails.""",

    "url_fetch": """**URL Article Fetching — Cell 44:**

```python
def fetch_article_from_url(url):
    article = Article(url)
    article.download()
    article.parse()
    return article.title, article.text
```

**Library:** `newspaper3k`

The user types `url: https://...` and the chatbot fetches the article automatically — no copy-pasting required.

**Limitations:** Paywalled sites, JavaScript-rendered pages, and some news sites block scraping.""",

    "subjectivity": """**Subjectivity Scoring (TextBlob):**

Range: **0.0** (completely objective/factual) → **1.0** (completely subjective/opinionated)

**Examples from the notebook:**
- *"Federal Reserve raises interest rates by 0.25%"* → subjectivity ≈ **0.30** (factual)
- *"EXPOSED: Government secretly controlling population..."* → subjectivity ≈ **0.64** (opinionated)

High subjectivity is a weak signal for fake news — it's one signal among several.""",

    "ethics": """**Ethical Considerations (report Section 6):**

- **US-centric bias** — both datasets are US political news
- **Keyword gaps** — 11 aspects miss housing, accessibility, indigenous rights
- **False positives** — mislabelling genuine journalism risks suppressing minority voices
- **Transparency** — LIME provides partial explainability but human review is still needed
- **Privacy** — user-submitted content needs data protection compliance

**The system should assist human reviewers, not replace them.**""",

    "limitations": """**Limitations & Future Work:**

- Keyword-based aspect extraction misses novel vocabulary
- ~67% accuracy — text alone is insufficient for reliable fake news detection
- Dataset is US-political-news specific
- No publisher/author metadata used

**Suggested improvements (Cell 51):**
- Use Named Entity Recognition instead of keyword lists
- Incorporate publisher metadata (strong predictor)
- Fine-tune BERT end-to-end on this specific dataset
- Multi-label classification for articles spanning multiple aspects""",

    "project_overview": """**Project Overview:**

Expert chatbot for the ABSA Fake News Detection system — Assignment 3, Charles Darwin University (2026).

**What the system does:**
1. Takes a news headline, article body, or URL
2. Runs the full ABSA pipeline (11 aspects, hybrid VADER+BERT)
3. Predicts FAKE or REAL with confidence score
4. Shows LIME-style word explanations
5. Answers domain expert questions

**Notebook structure (Assignment3.ipynb):**
- Cells 1-5: Install + setup VADER/BERT
- Cells 7-14: Load datasets, fix label bug
- Cells 16-23: Cleaning, summarisation, ABSA pipeline
- Cells 29-40: Train Models A, B, C + Naive Bayes + SMOTE
- Cells 41-42: LIME explainability
- Cells 44-48: URL fetching + chatbot""",

    "contributions": """**Team contributions:**

**Shekhar Bhandari:**
- Data loading and LIAR label bug fix
- Text cleaning pipeline
- TF-IDF feature engineering
- Model A & B training + evaluation metrics

**Princy Patel:**
- VADER sentiment implementation
- BERT hybrid approach
- Aspect category design
- Visualisations

**Hadi Rahaman Mohammed:**
- Article summarisation (LSA)
- Model C (RoBERTa fake news classifier)
- LIME explainability
- Chatbot interface
- Report writing""",

    "notebook": """**Notebook details (Assignment3.ipynb):**

Runs on **Google Colab** with GPU recommended for BERT.

**Key cells:**
- Cell 5: Setup VADER + load BERT (`cardiffnlp/twitter-roberta-base-sentiment-latest`)
- Cell 16: `clean_text()` function
- Cell 18: `summarise_article()` — LSA summarisation
- Cell 20: 11 aspect keyword lists
- Cell 23: `get_aspect_sentiment_fast()` — hybrid scoring
- Cell 30: Train Model A (TF-IDF only)
- Cell 32: Test Model C (RoBERTa)
- Cell 42: LIME explainer
- Cell 44: `fetch_article_from_url()` — newspaper3k
- Cell 46: `predict_article()` — main prediction function
- Cell 48: `chatbot()` — interactive terminal chatbot
- Cell 50: Save model pkl files""",

    "help": """**Here's what I can do:**

**Analyse news — paste anything:**
- A short headline: *"Government raises interest rates"*
- A full article (paste the whole text)
- A URL: `url: https://bbc.com/news/...`

**Answer expert questions:**
- What is ABSA / VADER / BERT?
- How does the hybrid approach work?
- What are the 11 aspect categories?
- How accurate are the models?
- What is LIME explainability?
- What datasets were used?
- What are the ethical risks?
- Who built this project?

**Explain results** — after an analysis, ask:
- *"Why was it predicted fake?"*
- *"What does subjectivity mean?"*

**Performance evaluation** — switch to the Evaluation tab""",
}


def classify_intent(message: str) -> str:
    msg = message.lower().strip()
    word_count = len(message.split())

    # Check explicit intent patterns first
    for intent, patterns in INTENTS.items():
        for p in patterns:
            if re.search(p, msg):
                return intent

    # ── Smart fallback ────────────────────────────────────────────────────────
    # Very short question → unknown
    if word_count < 5 and "?" in message:
        return "unknown"

    # Any substantial text (10+ words) that isn't a question → treat as article
    if word_count >= 10 and not msg.strip().endswith("?"):
        return "analyse"

    # Medium text (6-9 words) or ends without "?" — check for news vocabulary
    news_words = [
        "government", "police", "president", "market", "health", "climate",
        "election", "school", "military", "technology", "media", "crime",
        "vaccine", "border", "inflation", "fraud", "exposed", "shocking",
        "report", "says", "warns", "claims", "denies", "accused", "reveals",
        "breaking", "official", "study", "research", "according", "announce",
        "trump", "biden", "senate", "congress", "court", "minister", "prime",
        "economy", "tax", "arrest", "war", "virus", "pandemic", "shooting",
        "university", "hospital", "bank", "company", "stock", "rate", "price",
    ]
    if word_count >= 4 and any(w in msg for w in news_words):
        return "analyse"

    return "unknown"


def get_response(intent: str, message: str, last_result: dict | None) -> tuple[str, bool]:
    """
    Returns (response_text, should_analyse).
    should_analyse=True means the caller should run predict_article() on the message.
    """
    if intent in CASUAL:
        return random.choice(CASUAL[intent]), False

    if intent in KB:
        return KB[intent], False

    if intent == "analyse":
        return "", True

    # Context-aware follow-ups on last result
    if last_result and not last_result.get("error") and intent == "unknown":
        msg = message.lower()
        if any(w in msg for w in ["why", "explain", "reason", "how", "word", "lime", "aspect"]):
            r        = last_result
            asp_list = ", ".join(r["detected"].keys()) or "none detected"
            subj     = r["subjectivity"]
            sdesc    = "high (opinionated)" if subj > 0.5 else "moderate" if subj > 0.3 else "low (factual)"
            fw       = ", ".join(r["lime_fake"]) or "none"
            rw       = ", ".join(r["lime_real"]) or "none"
            return f"""**Explaining the last result:**

- **Prediction:** {r['prediction']} ({r['confidence']}% confidence)
- **Subjectivity:** {r['subjectivity']} — {sdesc}
- **Overall VADER score:** {r['overall_score']} ({r['overall_label']})
- **Aspects detected:** {asp_list}
- **Words pushing toward FAKE:** {fw}
- **Words pushing toward REAL:** {rw}

{"High subjectivity and negative framing on key aspects pushed the fake score above threshold." if r['prediction'] == 'FAKE' else "Low subjectivity and factual/neutral tone kept the fake score below threshold — predicted REAL."}

*Note: The classifier achieves ~67% accuracy — always apply human judgement.*""", False

    return """I'm not sure what you mean. Try:
- **Pasting a news headline or full article** to analyse
- Asking about ABSA, VADER, BERT, or the hybrid approach
- Typing **"help"** for a full list of topics""", False
