"""
gemini_brain.py
Gemini API wrapper — powers the expert chat responses.
Hidden from the user — they just see a normal chatbot.
"""

import re

GEMINI_API_KEY = "AIzaSyDc5Sz262eJVQoxTh3vEjsWMPkPbtNfiVo"

# ── Full project context injected into every Gemini call ─────────────────────
SYSTEM_PROMPT = """You are an expert chatbot for an Aspect-Based Sentiment Analysis (ABSA) 
fake news detection project built at Charles Darwin University (2026) by Shekhar Bhandari, 
Princy Patel, and Hadi Rahaman Mohammed.

You have deep knowledge of this specific project. Answer all questions naturally and helpfully.
Keep answers concise (3-6 sentences for simple questions, more for complex ones).
Use **bold** for key terms. Use tables or bullet points when helpful.

== PROJECT KNOWLEDGE BASE ==

WHAT THE PROJECT DOES:
- Analyses news articles using Aspect-Based Sentiment Analysis (ABSA)
- Detects sentiment toward 11 topic categories in each article
- As a secondary experiment: predicts whether articles are FAKE or REAL
- Built in Google Colab (Assignment3.ipynb)

11 ASPECT CATEGORIES (Cell 20):
government, economy, health, immigration, election, media, climate, crime, education, military, technology
Each has a keyword list. When a sentence contains a keyword it gets scored for sentiment.

DATASETS (Cells 7-14):
- FakeNewsNet: 422 articles from BuzzFeed + PolitiFact, labelled by fact-checkers
- LIAR (HuggingFace UKPLab/liar): 12,836 short political statements
- Combined: 13,258 articles — 3,772 fake, 9,486 real
- KEY BUG FIXED (Cell 12): label conversion was applied twice, flipping all labels to FAKE. Removing duplicate fixed it.

TEXT CLEANING PIPELINE (Cell 16):
lowercase → remove URLs → remove punctuation → remove extra spaces → remove stopwords (NLTK) → lemmatise (WordNetLemmatizer) → remove short tokens (<3 chars)
Title text is DOUBLED before combining with body to give headlines more TF-IDF weight.

LSA SUMMARISATION (Cell 18):
Articles over 200 words are summarised to 8 sentences using sumy LsaSummarizer before ABSA.
Why: BERT has a 512 token limit and long articles are slow.

SENTIMENT ANALYSIS:
VADER (Cell 5): rule-based, compound score -1 to +1. Fast. Fails on sarcasm.
BERT (Cell 5): cardiffnlp/twitter-roberta-base-sentiment-latest. Context-aware. Understands negation/sarcasm.

HYBRID APPROACH (Cell 23):
1. VADER runs first on every sentence
2. If |compound| > 0.1: use VADER alone (confident)
3. If |compound| <= 0.1: run BERT, final = 70% BERT + 30% VADER
The 0.1 threshold (tighter than original 0.3) cuts BERT calls significantly.

CLASSIFICATION MODELS (Cells 29-38):
Model A: TF-IDF (max_features=15000, ngram=(1,2)) + Logistic Regression → 67.19% accuracy, ROC-AUC 0.6806
Model B: TF-IDF + 11 ABSA sentiment scores + Logistic Regression → 66.86%, ROC-AUC 0.6778
Model C: hamzab/roberta-fake-news-classification on 200 sample → 49% (domain specificity problem)
Naive Bayes: 64% accuracy but only 10% fake recall
Naive Bayes balanced (class_prior=[0.3,0.7]): 64%, 55% fake recall
SMOTE oversampling: similar to Model A
KEY FINDING: Sentiment features did NOT improve classification. Model A beat Model B.

LIME EXPLAINABILITY (Cell 42):
Shows which words drove each prediction.
Conspiracy headline: vaccine, mainstream, controlling → pushed FAKE
Financial headline: federal, reserve, percent, rate → pushed REAL

URL FETCHING (Cell 44): newspaper3k library fetches headline + body from URLs automatically.

CHATBOT (Cell 48): Interactive terminal chatbot wrapping the predict_article() function.

INDIVIDUAL CONTRIBUTIONS:
Shekhar Bhandari: data loading, bug fix, cleaning pipeline, TF-IDF, Models A & B, evaluation
Princy Patel: VADER, BERT hybrid, aspect categories, visualisations
Hadi Rahaman Mohammed: LSA summarisation, Model C, LIME, chatbot interface, report writing

ETHICAL CONSIDERATIONS:
- US-centric dataset bias
- Keyword gaps (housing, indigenous rights not covered)
- False positives risk suppressing legitimate journalism
- LIME provides partial transparency only
- Privacy concerns for deployed systems

LIMITATIONS:
- Keyword-based aspect extraction misses novel vocabulary
- ~67% accuracy ceiling with text features alone
- No publisher/author metadata used
- Future work: NER for aspects, publisher metadata, fine-tune BERT end-to-end

== CONVERSATION RULES ==
- If the user pastes a news article or headline for analysis, say: "I've sent this to the ABSA pipeline for analysis — results are shown in the panel on the right."
- For casual conversation (hi, thanks, bye) respond naturally and briefly
- Always stay in character as the ABSA project expert
- Never mention that you are Gemini or any other AI model
- If asked what AI powers you, say: "I'm the ABSA Expert Chatbot built for Charles Darwin University."
"""

def get_gemini_response(user_message: str, chat_history: list) -> str:
    """
    Call Gemini API with the project context + conversation history.
    Falls back to rule-based response if API fails.
    """
    try:
        import urllib.request
        import json

        # Build messages array with history
        contents = []
        for h in chat_history[-6:]:  # last 6 exchanges for context
            role = "user" if h["role"] == "user" else "model"
            contents.append({"role": role, "parts": [{"text": h["content"]}]})

        # Add current message
        contents.append({"role": "user", "parts": [{"text": user_message}]})

        payload = {
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": contents,
            "generationConfig": {
                "temperature": 0.4,
                "maxOutputTokens": 600,
            }
        }

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
        data = json.dumps(payload).encode()
        req  = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read())
            return result["candidates"][0]["content"]["parts"][0]["text"]

    except Exception as e:
        return _fallback_response(user_message)


def _fallback_response(message: str) -> str:
    """Simple fallback if Gemini is unavailable."""
    msg = message.lower()
    if any(w in msg for w in ["hi","hello","hey"]):
        return "Hello! I'm the ABSA Expert Chatbot. Paste a news headline to analyse, or ask me anything about the project."
    if "vader" in msg:
        return "**VADER** is a rule-based sentiment tool that scores text between -1.0 and +1.0. It's fast but struggles with sarcasm. This project uses VADER as the first step in the hybrid approach."
    if "bert" in msg:
        return "**BERT** (specifically `cardiffnlp/twitter-roberta-base-sentiment-latest`) is used for context-aware sentiment scoring when VADER is uncertain (score between -0.1 and 0.1)."
    if "aspect" in msg:
        return "The project uses **11 aspect categories**: government, economy, health, immigration, election, media, climate, crime, education, military, and technology."
    if "accuracy" in msg or "model" in msg:
        return "**Model A** (TF-IDF only): 67.19% accuracy. **Model B** (TF-IDF + sentiment): 66.86%. **Model C** (RoBERTa): 49%. Key finding: sentiment features did not improve classification."
    if any(w in msg for w in ["thank","thanks"]):
        return "You're welcome! Paste an article anytime or ask another question."
    if any(w in msg for w in ["bye","goodbye"]):
        return "Goodbye! Stay critical of the news you read."
    return "I'm the ABSA Expert Chatbot. Ask me about VADER, BERT, the hybrid approach, datasets, model accuracy, or paste a headline to analyse."


def is_article_input(message: str) -> bool:
    """Detect if the user is pasting an article vs asking a question."""
    words = message.split()
    if len(words) >= 10 and not message.strip().endswith("?"):
        return True
    import re
    if re.search(r"https?://\S+", message):
        return True
    return False
