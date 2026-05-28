"""
app.py — ABSA Expert Chatbot  v3
Assignment 3 — Charles Darwin University (2026)
Bhandari · Patel · Rahaman Mohammed
"""

import re
import time
import streamlit as st
import plotly.graph_objects as go
from absa_engine import predict_article, QA_EVAL, aspects
from chatbot_nlu import classify_intent, get_response

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ABSA News Chatbot",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,400;0,500;0,600;0,700;1,400&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #f0f2f5; }

/* ── Sidebar ─────────────────────────────────────── */
[data-testid="stSidebar"] { background: #0f172a !important; border-right: 1px solid #1e293b; }
[data-testid="stSidebar"] * { color: #94a3b8 !important; font-size: 12.5px !important; }
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3 {
    color: #f1f5f9 !important; font-size: 13px !important; font-weight: 600 !important;
    letter-spacing: 0.05em; text-transform: uppercase;
}
[data-testid="stSidebar"] .stButton > button {
    background: #1e293b !important; border: 1px solid #334155 !important;
    color: #94a3b8 !important; border-radius: 7px !important;
    font-size: 12px !important; padding: 6px 10px !important;
    text-align: left !important; width: 100% !important;
    transition: all 0.15s !important; line-height: 1.4 !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #334155 !important; color: #f1f5f9 !important; border-color: #475569 !important;
}
[data-testid="stSidebar"] hr { border-color: #1e293b !important; margin: 8px 0 !important; }

/* ── Top header bar ──────────────────────────────── */
.top-header {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #1e40af 100%);
    border-radius: 16px; padding: 24px 28px; margin-bottom: 20px;
    display: flex; align-items: center; gap: 16px;
}
.top-header-icon { font-size: 2.2rem; }
.top-header-title { font-size: 1.5rem; font-weight: 700; color: #fff; margin: 0; letter-spacing: -0.3px; }
.top-header-sub { font-size: 13px; color: #a5b4fc; margin-top: 2px; }
.top-header-badge {
    margin-left: auto; background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 20px; padding: 4px 14px;
    font-size: 12px; color: #c7d2fe; white-space: nowrap;
}

/* ── Cards ───────────────────────────────────────── */
.card {
    background: #fff; border: 1px solid #e2e8f0; border-radius: 14px;
    padding: 18px 22px; margin-bottom: 14px;
}
.card-title {
    font-size: 11px; font-weight: 600; color: #94a3b8;
    text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 10px;
}

/* ── Metrics ─────────────────────────────────────── */
[data-testid="stMetric"] {
    background: #fff; border: 1px solid #e2e8f0; border-radius: 12px;
    padding: 14px 16px !important;
}
[data-testid="stMetricLabel"] {
    font-size: 10px !important; font-weight: 600 !important;
    text-transform: uppercase; letter-spacing: 0.08em; color: #94a3b8 !important;
}
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.35rem !important; font-weight: 500 !important; color: #0f172a !important;
}

/* ── Chat messages ───────────────────────────────── */
[data-testid="stChatMessage"] {
    background: #fff !important; border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important; margin-bottom: 8px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
}
[data-testid="stChatInput"] > div {
    background: #fff !important; border: 1.5px solid #cbd5e1 !important;
    border-radius: 12px !important; box-shadow: 0 1px 4px rgba(0,0,0,0.06) !important;
}
[data-testid="stChatInput"] textarea {
    font-family: 'Inter', sans-serif !important; font-size: 14px !important; color: #0f172a !important;
}

/* ── Tabs ────────────────────────────────────────── */
[data-baseweb="tab-list"] { background: #fff !important; border-radius: 10px !important;
    border: 1px solid #e2e8f0 !important; padding: 4px !important; gap: 2px !important; }
[data-baseweb="tab"] {
    font-weight: 500 !important; font-size: 13.5px !important;
    border-radius: 8px !important; padding: 8px 18px !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    background: #6366f1 !important; color: #fff !important;
}

/* ── Expander ────────────────────────────────────── */
[data-testid="stExpander"] {
    background: #fff !important; border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important; margin-bottom: 8px !important;
}

/* ── Verdict badges ──────────────────────────────── */
.verdict-fake {
    display: inline-flex; align-items: center; gap: 6px;
    background: #fef2f2; color: #dc2626; border: 1.5px solid #fca5a5;
    padding: 5px 18px; border-radius: 22px; font-weight: 700; font-size: 1rem;
}
.verdict-real {
    display: inline-flex; align-items: center; gap: 6px;
    background: #f0fdf4; color: #16a34a; border: 1.5px solid #86efac;
    padding: 5px 18px; border-radius: 22px; font-weight: 700; font-size: 1rem;
}
.badge-pos { display:inline-block;background:#f0fdf4;color:#16a34a;border:1px solid #86efac;padding:1px 9px;border-radius:10px;font-size:0.7rem;font-weight:600; }
.badge-neg { display:inline-block;background:#fef2f2;color:#dc2626;border:1px solid #fca5a5;padding:1px 9px;border-radius:10px;font-size:0.7rem;font-weight:600; }
.badge-neu { display:inline-block;background:#f8fafc;color:#64748b;border:1px solid #cbd5e1;padding:1px 9px;border-radius:10px;font-size:0.7rem;font-weight:600; }
.badge-info { display:inline-block;background:#eff6ff;color:#1d4ed8;border:1px solid #bfdbfe;padding:1px 9px;border-radius:10px;font-size:0.7rem;font-weight:600; }

/* ── Aspect rows ─────────────────────────────────── */
.asp-row { display:flex;align-items:center;gap:10px;padding:6px 0;border-bottom:1px solid #f1f5f9; }
.asp-name { font-size:12px;font-weight:600;color:#334155;text-transform:capitalize;width:90px;flex-shrink:0; }
.asp-bar-bg { flex:1;background:#f1f5f9;border-radius:3px;height:5px;overflow:hidden; }
.asp-bar-fill { height:100%;border-radius:3px;transition:width 0.5s; }
.asp-val { font-family:'JetBrains Mono',monospace;font-size:10.5px;color:#94a3b8;width:48px;text-align:right; }

/* ── LIME box ────────────────────────────────────── */
.lime-box { background:#fafafa;border:1px solid #e2e8f0;border-radius:8px;padding:10px 14px;margin-top:8px; }
.lime-fake-word { display:inline-block;background:#fef2f2;color:#dc2626;border:1px solid #fecaca;padding:1px 7px;border-radius:6px;margin:2px 2px;font-size:11px; }
.lime-real-word { display:inline-block;background:#f0fdf4;color:#16a34a;border:1px solid #bbf7d0;padding:1px 7px;border-radius:6px;margin:2px 2px;font-size:11px; }

/* ── Info / banner ───────────────────────────────── */
.info-banner { background:#eff6ff;border:1px solid #bfdbfe;border-radius:10px;padding:11px 16px;font-size:13px;color:#1e40af;line-height:1.6;margin-bottom:14px; }
.warn-banner { background:#fffbeb;border:1px solid #fde68a;border-radius:10px;padding:11px 16px;font-size:13px;color:#92400e;line-height:1.6;margin-bottom:14px; }

/* ── Empty state ─────────────────────────────────── */
.empty-state { background:#fff;border:1px solid #e2e8f0;border-radius:14px;padding:56px 24px;text-align:center; }

/* ── URL input hint ──────────────────────────────── */
.url-hint { font-size:11px;color:#94a3b8;margin-top:4px; }

/* ── Eval result row ─────────────────────────────── */
.eval-row { border-radius:10px;padding:11px 14px;margin-bottom:7px;border-left:4px solid; }
.eval-pass { background:#f0fdf4;border-color:#22c55e; }
.eval-fail { background:#fef2f2;border-color:#ef4444; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in {
    "messages": [], "last_result": None,
    "history": [], "eval_results": [], "eval_done": False,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Helpers ───────────────────────────────────────────────────────────────────
def bar_color(s):
    return "#22c55e" if s >= 0.05 else "#ef4444" if s <= -0.05 else "#94a3b8"

def sent_cls(s):
    return "pos" if s >= 0.05 else "neg" if s <= -0.05 else "neu"

def extract_url(text):
    m = re.search(r"https?://\S+", text)
    return m.group(0) if m else None

def fetch_url(url):
    """Fetch article from URL using newspaper3k."""
    try:
        from newspaper import Article
        a = Article(url)
        a.download()
        a.parse()
        return (a.title or "").strip(), (a.text or "").strip()
    except Exception as e:
        return None, str(e)

def render_result_panel(r):
    badge_cls = "fake" if r["prediction"] == "FAKE" else "real"
    emoji     = "🚨" if r["prediction"] == "FAKE" else "✅"

    # Verdict card
    st.markdown(f"""
<div class="card">
  <div class="card-title">Prediction</div>
  <div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap">
    <span class="verdict-{badge_cls}">{emoji} {r['prediction']}</span>
    <span style="font-size:13px;color:#64748b;font-family:'JetBrains Mono',monospace">{r['confidence']}% confidence</span>
  </div>
  {"<div style='margin-top:10px;font-size:12px;color:#64748b;border-top:1px solid #f1f5f9;padding-top:8px'><b>Headline:</b> " + r['headline'][:100] + ("…" if len(r['headline']) > 100 else "") + "</div>" if r.get("headline") else ""}
</div>""", unsafe_allow_html=True)

    # Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Sentiment",    r["overall_label"].capitalize())
    c2.metric("VADER score",  r["overall_score"])
    c3.metric("Subjectivity", r["subjectivity"])

    # Charts + breakdown
    detected = r["detected"]
    if not detected:
        st.info("No recognised aspect keywords found in this text.")
        return

    t1, t2 = st.tabs(["🕸 Radar", "📊 Bar chart"])
    labels = list(detected.keys())
    scores = list(detected.values())

    with t1:
        norm = [(s + 1) / 2 for s in scores]
        fig  = go.Figure(go.Scatterpolar(
            r=norm + [norm[0]], theta=labels + [labels[0]],
            fill="toself", fillcolor="rgba(99,102,241,0.1)",
            line=dict(color="#6366f1", width=2.5),
            marker=dict(size=6, color="#6366f1"),
        ))
        fig.update_layout(
            polar=dict(
                bgcolor="#fafafa",
                radialaxis=dict(visible=True, range=[0,1], tickfont=dict(color="#94a3b8", size=9),
                                gridcolor="#e2e8f0", linecolor="#e2e8f0"),
                angularaxis=dict(tickfont=dict(color="#334155", size=10, family="Inter"),
                                 gridcolor="#e2e8f0", linecolor="#e2e8f0"),
            ),
            paper_bgcolor="#ffffff", margin=dict(l=30,r=30,t=20,b=20),
            height=270, showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        fig2 = go.Figure(go.Bar(
            x=[l.capitalize() for l in labels], y=scores,
            marker_color=[bar_color(s) for s in scores], marker_line_width=0,
            text=[f"{s:+.2f}" for s in scores], textposition="outside",
            textfont=dict(size=10),
        ))
        fig2.update_layout(
            paper_bgcolor="#fff", plot_bgcolor="#fafafa",
            xaxis=dict(tickfont=dict(size=10, color="#334155"), gridcolor="#f1f5f9"),
            yaxis=dict(range=[-1.2, 1.2], gridcolor="#e2e8f0", zeroline=True,
                       zerolinecolor="#cbd5e1", zerolinewidth=1.5),
            margin=dict(l=10,r=10,t=20,b=10), height=210, showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Aspect breakdown
    st.markdown('<div class="card-title" style="margin-top:4px">Aspect breakdown</div>', unsafe_allow_html=True)
    tm = {"positive":"pos","negative":"neg","neutral":"neu"}
    for asp, score in detected.items():
        pct   = int((score + 1) / 2 * 100)
        label = "positive" if score >= 0.05 else "negative" if score <= -0.05 else "neutral"
        st.markdown(f"""
<div class="asp-row">
  <span class="asp-name">{asp}</span>
  <div class="asp-bar-bg"><div class="asp-bar-fill" style="width:{pct}%;background:{bar_color(score)}"></div></div>
  <span class="badge-{tm[label]}">{label}</span>
  <span class="asp-val">{score:+.3f}</span>
</div>""", unsafe_allow_html=True)

    # LIME signals
    fw = r.get("lime_fake", [])
    rw = r.get("lime_real", [])
    fp = " ".join(f'<span class="lime-fake-word">{w}</span>' for w in fw) if fw else '<span style="color:#94a3b8;font-size:11px">none detected</span>'
    rp = " ".join(f'<span class="lime-real-word">{w}</span>' for w in rw) if rw else '<span style="color:#94a3b8;font-size:11px">none detected</span>'
    st.markdown(f"""
<div class="lime-box">
  <div style="font-size:11px;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px">LIME — word signals</div>
  <div style="margin-bottom:4px"><span style="font-size:11px;color:#dc2626;font-weight:500">→ FAKE: </span>{fp}</div>
  <div><span style="font-size:11px;color:#16a34a;font-weight:500">→ REAL: </span>{rp}</div>
</div>""", unsafe_allow_html=True)


def make_chat_reply(r):
    asp = ", ".join(r["detected"].keys()) or "none detected"
    e   = "🚨" if r["prediction"] == "FAKE" else "✅"
    return f"""{e} **{r['prediction']}** — {r['confidence']}% confidence

| | |
|---|---|
| Overall sentiment | {r['overall_label'].capitalize()} (VADER: {r['overall_score']}) |
| Subjectivity | {r['subjectivity']} |
| Aspects found | {asp} |

Full breakdown in the panel → Ask *"why {r['prediction'].lower()}?"* or paste another article."""


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
<div style="padding:12px 4px 8px">
  <div style="font-size:1.1rem;font-weight:700;color:#f1f5f9;letter-spacing:-0.3px">📰 ABSA Chatbot</div>
  <div style="font-size:11px;color:#475569;margin-top:2px">Charles Darwin University · 2026</div>
  <div style="font-size:11px;color:#475569">Bhandari · Patel · Rahaman Mohammed</div>
</div>
""", unsafe_allow_html=True)
    st.divider()

    st.markdown("### Try a headline")
    examples = {
        "🚨 5G conspiracy":    "EXPOSED: Government secretly controlling population through 5G towers and vaccines",
        "📈 Fed rate rise":    "Federal Reserve raises interest rates by 0.25 percent following inflation data",
        "🗳 Election fraud":   "SHOCKING: Democrats rigging election through mail-in ballot fraud across swing states",
        "🌍 Climate warning":  "Climate scientists warn record carbon emissions risk catastrophic warming as governments stall",
        "💊 Pharma cover-up":  "Big Pharma hiding cancer cure to protect vaccine profits — doctors speak out",
        "🏫 Enrolment drop":   "Universities report sharp decline in student enrolment for third consecutive year",
        "🔫 Crime report":     "Police arrest three suspects following armed robbery at downtown bank branch",
        "🪖 Military aid":     "Pentagon announces new military aid package for Ukraine amid ongoing conflict",
    }
    for label, text in examples.items():
        if st.button(label, use_container_width=True):
            st.session_state["_prefill"] = text

    st.divider()
    st.markdown("### Ask the expert")
    questions = [
        "What is ABSA?",
        "How does VADER work?",
        "How does BERT work?",
        "How does the hybrid approach work?",
        "What are the 11 aspects?",
        "How is fake/real classified?",
        "What were the model accuracies?",
        "What is LIME explainability?",
        "What datasets were used?",
        "What are the ethical risks?",
        "Who built this project?",
    ]
    for q in questions:
        if st.button(q, use_container_width=True):
            st.session_state["_prefill"] = q

    st.divider()
    c1, c2 = st.columns(2)
    c1.metric("Messages", len(st.session_state.messages))
    c2.metric("Analysed",  len(st.session_state.history))
    if st.button("🗑 Clear chat", use_container_width=True):
        st.session_state.messages   = []
        st.session_state.last_result = None
        st.session_state.history    = []
        st.rerun()


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_chat, tab_eval = st.tabs(["💬  Expert Chatbot", "📊  Performance Evaluation"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CHAT
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    col_chat, col_panel = st.columns([3, 2], gap="large")

    # ── Left: chat ────────────────────────────────────────────────────────────
    with col_chat:
        # Header
        st.markdown("""
<div class="top-header">
  <div class="top-header-icon">🧠</div>
  <div>
    <div class="top-header-title">ABSA Expert Chatbot</div>
    <div class="top-header-sub">Fake news detection · Aspect-based sentiment analysis</div>
  </div>
  <div class="top-header-badge">CDU 2026</div>
</div>""", unsafe_allow_html=True)

        st.markdown("""
<div class="info-banner">
  Paste a <b>headline</b>, full <b>article text</b>, or a <b>URL</b> to analyse.
  Or ask any question about the methodology, models, or datasets.
</div>""", unsafe_allow_html=True)

        # Welcome message
        if not st.session_state.messages:
            with st.chat_message("assistant"):
                st.markdown("""Hi! I'm the **ABSA Fake News Expert** from Charles Darwin University (2026).

**What I can do:**
- 🔍 Analyse any **headline, full article, or URL** → fake/real prediction + 11-aspect sentiment breakdown
- 💡 Answer **expert questions** about ABSA, VADER, BERT, LIME, datasets, models
- 📊 Run **performance evaluation** on preset or custom articles

Try pasting a headline below, or pick one from the sidebar. Type **"help"** for everything I can do.""")

        # Render message history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Prefill from sidebar
        prefill    = st.session_state.pop("_prefill", None)
        user_input = st.chat_input("Paste headline, article, URL — or ask a question…")
        if prefill and not user_input:
            user_input = prefill

        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            intent = classify_intent(user_input)
            resp, do_analyse = get_response(intent, user_input, st.session_state.last_result)

            with st.chat_message("assistant"):
                # ── URL detection ─────────────────────────────────────
                url = extract_url(user_input)

                if url or do_analyse or (not resp and len(user_input.split()) >= 5):
                    with st.spinner("Analysing…"):
                        if url:
                            st.info(f"🔗 Fetching article from URL…")
                            title, body = fetch_url(url)
                            if title or body:
                                result = predict_article(headline=title, body=body)
                                if result:
                                    result["headline"] = title or url[:80]
                            else:
                                result = {"error": f"Could not fetch URL. Try pasting the article text directly.\n\nError: {body}"}
                        else:
                            result = predict_article(headline=user_input)

                    if result and "error" not in result:
                        st.session_state.last_result = result
                        st.session_state.history.append({"text": user_input[:80], "result": result})
                        reply = make_chat_reply(result)
                    else:
                        err = result.get("error", "Could not analyse. Please try again.") if result else "Analysis failed."
                        reply = f"⚠️ {err}"

                    st.markdown(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                else:
                    st.markdown(resp)
                    st.session_state.messages.append({"role": "assistant", "content": resp})

    # ── Right: results panel ──────────────────────────────────────────────────
    with col_panel:
        st.markdown("""<div style="font-size:11px;font-weight:600;color:#94a3b8;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:12px">Analysis Results</div>""", unsafe_allow_html=True)

        if st.session_state.last_result and "error" not in st.session_state.last_result:
            render_result_panel(st.session_state.last_result)

            with st.expander("All 11 aspect scores"):
                for asp in aspects:
                    score = st.session_state.last_result["absa_scores"].get(asp, 0.0)
                    label = "positive" if score >= 0.05 else "negative" if score <= -0.05 else "neutral"
                    pct   = int((score + 1) / 2 * 100)
                    tm    = {"positive":"pos","negative":"neg","neutral":"neu"}
                    st.markdown(f"""
<div class="asp-row">
  <span class="asp-name">{asp}</span>
  <div class="asp-bar-bg"><div class="asp-bar-fill" style="width:{pct}%;background:{bar_color(score)}"></div></div>
  <span class="badge-{tm[label]}">{label}</span>
  <span class="asp-val">{score:+.3f}</span>
</div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
<div class="empty-state">
  <div style="font-size:2.5rem;margin-bottom:12px">📰</div>
  <div style="font-size:14px;font-weight:600;color:#64748b">Results appear here</div>
  <div style="font-size:12px;color:#94a3b8;margin-top:6px">Paste a headline or article in the chat</div>
</div>""", unsafe_allow_html=True)

    # History
    if len(st.session_state.history) > 1:
        with st.expander(f"📚 Session history — {len(st.session_state.history)} articles analysed"):
            for h in reversed(st.session_state.history):
                r   = h["result"]
                ico = "🚨" if r["prediction"] == "FAKE" else "✅"
                asp = ", ".join(r["detected"].keys()) or "—"
                st.markdown(
                    f"**{ico} {r['prediction']}** ({r['confidence']}%) &nbsp;·&nbsp; "
                    f"Subjectivity: `{r['subjectivity']}` &nbsp;·&nbsp; Aspects: *{asp}*  \n"
                    f"<span style='font-size:12px;color:#94a3b8'>{h['text']}</span>",
                    unsafe_allow_html=True,
                )
                st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PERFORMANCE EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
with tab_eval:
    # Header
    st.markdown("""
<div class="top-header" style="margin-bottom:20px">
  <div class="top-header-icon">📊</div>
  <div>
    <div class="top-header-title">Performance Evaluation</div>
    <div class="top-header-sub">Test the ABSA pipeline against labelled articles</div>
  </div>
</div>""", unsafe_allow_html=True)

    # ── Two evaluation modes ──────────────────────────────────────────────────
    eval_tab1, eval_tab2 = st.tabs(["📋  Preset QA Dataset (10 articles)", "✏️  Test Your Own Article"])

    # ── MODE 1: Preset QA dataset ─────────────────────────────────────────────
    with eval_tab1:
        st.markdown("""
<div class="info-banner">
Evaluates the pipeline against <b>10 pre-labelled headlines</b> covering all 11 aspect categories —
replicating the human testing methodology from the notebook (Cell 46-48).
Results are compared against Model A (67.19%), Model B (66.86%), and Model C (49%).
</div>""", unsafe_allow_html=True)

        col_btn, col_desc = st.columns([1, 2])
        with col_btn:
            run_preset = st.button("▶ Run preset evaluation", type="primary", use_container_width=True)
        with col_desc:
            st.markdown("""**10 headlines** · labelled FAKE / REAL  
**Metric:** Exact-match accuracy  
**Covers:** All 11 aspect categories""")

        if run_preset:
            with st.spinner("Running pipeline on 10 preset headlines…"):
                results = []
                bar = st.progress(0)
                for i, item in enumerate(QA_EVAL):
                    r      = predict_article(headline=item["q"])
                    passed = r is not None and r.get("prediction") == item["expected"]
                    results.append({
                        "question":   item["q"],
                        "expected":   item["expected"],
                        "predicted":  r["prediction"] if r else "ERROR",
                        "confidence": r["confidence"] if r else 0,
                        "category":   item["category"],
                        "passed":     passed,
                    })
                    bar.progress((i + 1) / len(QA_EVAL))
                bar.empty()
                st.session_state.eval_results = results
                st.session_state.eval_done    = True

        if st.session_state.eval_done and st.session_state.eval_results:
            results = st.session_state.eval_results
            n       = len(results)
            correct = sum(1 for r in results if r["passed"])
            acc     = round(correct / n * 100, 1)

            st.divider()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy",    f"{acc}%")
            m2.metric("Correct",     f"{correct}/{n}")
            m3.metric("FAKE correct",
                      f"{sum(1 for r in results if r['passed'] and r['expected']=='FAKE')}/"
                      f"{sum(1 for r in results if r['expected']=='FAKE')}")
            m4.metric("REAL correct",
                      f"{sum(1 for r in results if r['passed'] and r['expected']=='REAL')}/"
                      f"{sum(1 for r in results if r['expected']=='REAL')}")

            # Comparison chart
            st.markdown("#### Comparison with notebook models")
            fig = go.Figure(go.Bar(
                x=["Model A\n(TF-IDF)", "Model B\n(TF-IDF+Sent)", "Naive Bayes", "Model C\n(RoBERTa)", f"This chatbot\n(eval set)"],
                y=[67.19, 66.86, 64.0, 49.0, acc],
                marker_color=["#6366f1","#818cf8","#a5b4fc","#f87171","#22c55e" if acc >= 60 else "#f59e0b"],
                marker_line_width=0,
                text=[f"{v}%" for v in [67.19, 66.86, 64.0, 49.0, acc]],
                textposition="outside", textfont=dict(size=11),
            ))
            fig.update_layout(
                paper_bgcolor="#fff", plot_bgcolor="#fafafa",
                font=dict(family="Inter", color="#64748b"),
                xaxis=dict(tickfont=dict(size=11, color="#334155")),
                yaxis=dict(range=[0, 105], gridcolor="#e2e8f0", title="Accuracy (%)"),
                margin=dict(l=10,r=10,t=30,b=10), height=270,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Detailed results
            st.markdown("#### Detailed results")
            for r in results:
                icon  = "✅" if r["passed"] else "❌"
                eb    = "verdict-real" if r["expected"]  == "REAL" else "verdict-fake"
                pb    = "verdict-real" if r["predicted"] == "REAL" else "verdict-fake"
                cls   = "eval-pass" if r["passed"] else "eval-fail"
                st.markdown(f"""
<div class="eval-row {cls}">
  <div style="display:flex;gap:8px;align-items:flex-start">
    <span style="font-size:1rem;flex-shrink:0">{icon}</span>
    <div>
      <div style="font-size:13px;font-weight:500;color:#1e293b;margin-bottom:5px">{r['question']}</div>
      <div style="display:flex;flex-wrap:wrap;gap:8px;align-items:center">
        <span style="font-size:11px;color:#94a3b8">Expected:</span>
        <span class="badge-{'pos' if r['expected']=='REAL' else 'neg'}">{r['expected']}</span>
        <span style="font-size:11px;color:#94a3b8">Got:</span>
        <span class="badge-{'pos' if r['predicted']=='REAL' else 'neg'}">{r['predicted']}</span>
        <span class="badge-info">{r['confidence']}% conf</span>
        <span style="font-size:11px;background:#f1f5f9;padding:1px 8px;border-radius:6px;color:#64748b">{r['category']}</span>
      </div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

            # Interpretation
            st.divider()
            st.markdown(f"""
<div class="info-banner">
<b>Chatbot accuracy: {acc}%</b> — {correct}/{n} correct.<br>
{"Comparable to or better than the notebook's Model A (67.19%) and Model B (66.86%)." if acc >= 60 else "Reflects the notebook's finding: text features alone hit a ~67% ceiling. Publisher metadata and social context signals are needed to go further."}<br><br>
<b>Key finding from the notebook:</b> Adding ABSA sentiment scores <i>did not improve</i> classification
(Model B 66.86% &lt; Model A 67.19%). Emotional tone and factual accuracy are largely independent.
</div>""", unsafe_allow_html=True)

    # ── MODE 2: Custom article evaluation ─────────────────────────────────────
    with eval_tab2:
        st.markdown("""
<div class="info-banner">
Test <b>your own article or headline</b> — paste text or a URL below, set the expected label,
and see how the pipeline performs on it. Great for testing with articles you already know are fake or real.
</div>""", unsafe_allow_html=True)

        custom_text = st.text_area(
            "Paste headline, article body, or URL",
            placeholder="e.g. Federal Reserve raises interest rates by 0.25 percent…\nor paste a full article\nor https://bbc.com/news/...",
            height=120,
            label_visibility="collapsed",
        )

        col_label, col_run2 = st.columns([1, 1])
        with col_label:
            expected_label = st.selectbox("Expected label (what should the correct answer be?)",
                                          ["FAKE", "REAL"], key="custom_expected")
        with col_run2:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            run_custom = st.button("▶ Run analysis", type="primary", use_container_width=True, key="run_custom")

        if run_custom:
            if not custom_text.strip():
                st.warning("Please paste some text or a URL above.")
            else:
                with st.spinner("Analysing…"):
                    url = extract_url(custom_text)
                    if url:
                        st.info(f"🔗 Fetching from URL: `{url[:70]}…`")
                        title, body = fetch_url(url)
                        if title or body:
                            result = predict_article(headline=title, body=body)
                            if result:
                                result["headline"] = title or url[:80]
                            display_text = title or url[:80]
                        else:
                            st.error(f"Could not fetch URL. Paste the article text directly instead.\n\n{body}")
                            result = None
                            display_text = ""
                    else:
                        result = predict_article(headline=custom_text)
                        display_text = custom_text[:80]

                if result and "error" not in result:
                    predicted = result["prediction"]
                    passed    = predicted == expected_label
                    icon      = "✅ Correct!" if passed else "❌ Incorrect"
                    banner_cls = "info-banner" if passed else "warn-banner"

                    st.markdown(f"""
<div class="{banner_cls}" style="font-size:14px">
  <b>{icon}</b> — Predicted <b>{predicted}</b> ({result['confidence']}% confidence),
  expected <b>{expected_label}</b>
</div>""", unsafe_allow_html=True)

                    # Full result panel
                    render_result_panel(result)

                    # Log to session
                    st.session_state.last_result = result
                    st.session_state.history.append({"text": display_text, "result": result})
