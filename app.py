"""
app.py — ABSA Expert Chatbot  v5  (Premium UI)
Assignment 3 — Charles Darwin University (2026)
Bhandari · Patel · Rahaman Mohammed
"""

import re
import time
import streamlit as st
import plotly.graph_objects as go
from absa_engine import predict_article, QA_EVAL, aspects
from gemini_brain import get_gemini_response, is_article_input

st.set_page_config(
    page_title="ABSA News Chatbot",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── GLOBAL ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
#MainMenu, footer, [data-testid="stToolbar"], .stDeployButton { display:none !important; }
.stApp { background: #f5f7fa !important; }
.main .block-container { padding: 1.5rem 2rem 3rem !important; max-width: 100% !important; }

/* ── SIDEBAR ── */
[data-testid="stSidebar"] { background: #1a2236 !important; border-right: 1px solid #232d42 !important; }
[data-testid="stSidebar"] > div:first-child { background: #1a2236 !important; padding-top: 0 !important; }
[data-testid="stSidebar"] * { font-family: 'Inter', sans-serif !important; }
[data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] div { color: #4a6080 !important; font-size: 12px !important; background: transparent !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #8ba3c7 !important; font-size: 9.5px !important; font-weight: 600 !important; text-transform: uppercase !important; letter-spacing: 0.1em !important; }
[data-testid="stSidebar"] .stButton > button { background: transparent !important; border: none !important; border-radius: 8px !important; color: #4a6080 !important; font-size: 12px !important; padding: 7px 10px !important; text-align: left !important; width: 100% !important; transition: all 0.15s !important; font-family: 'Inter', sans-serif !important; }
[data-testid="stSidebar"] .stButton > button:hover { background: #232d42 !important; color: #7eb8f7 !important; }
[data-testid="stSidebar"] hr { border: none !important; border-top: 1px solid #232d42 !important; margin: 7px 0 !important; }
[data-testid="stSidebar"] [data-testid="stMetric"] { background: #141c2e !important; border: 1px solid #232d42 !important; border-radius: 8px !important; padding: 10px 12px !important; }
[data-testid="stSidebar"] [data-testid="stMetricLabel"] { color: #2d4460 !important; font-size: 9px !important; text-transform: uppercase !important; letter-spacing: 0.1em !important; }
[data-testid="stSidebar"] [data-testid="stMetricValue"] { color: #5b8fc7 !important; font-size: 1.3rem !important; font-family: 'JetBrains Mono', monospace !important; }

/* ── TABS ── */
[data-baseweb="tab-list"] { background: #ffffff !important; border-radius: 12px !important; border: 1px solid #e2e8f0 !important; padding: 4px !important; gap: 3px !important; box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important; }
[data-baseweb="tab"] { font-family: 'Inter', sans-serif !important; font-size: 13px !important; font-weight: 500 !important; border-radius: 9px !important; color: #64748b !important; padding: 8px 22px !important; transition: all 0.2s !important; background: transparent !important; }
[aria-selected="true"][data-baseweb="tab"] { background: #1e40af !important; color: #ffffff !important; }

/* ── CHAT MESSAGES ── */
[data-testid="stChatMessage"] { background: #ffffff !important; border: 1px solid #e8edf5 !important; border-radius: 16px !important; margin-bottom: 10px !important; box-shadow: 0 1px 4px rgba(0,0,0,0.05) !important; }
[data-testid="stChatMessage"] p { color: #334155 !important; font-size: 13.5px !important; line-height: 1.65 !important; }
[data-testid="stChatMessage"] strong { color: #1e40af !important; }
[data-testid="stChatMessage"] code { color: #1e40af !important; background: #eff6ff !important; border-radius: 4px !important; padding: 1px 5px !important; font-family: 'JetBrains Mono', monospace !important; }
[data-testid="stChatMessage"] li { color: #475569 !important; }
[data-testid="stChatMessage"] td, [data-testid="stChatMessage"] th { color: #475569 !important; border-color: #e2e8f0 !important; }

/* ── CHAT INPUT ── */
[data-testid="stChatInput"] > div { background: #ffffff !important; border: 1.5px solid #cbd5e1 !important; border-radius: 14px !important; box-shadow: 0 1px 4px rgba(0,0,0,0.06) !important; }
[data-testid="stChatInput"] > div:focus-within { border-color: #3b82f6 !important; }
[data-testid="stChatInput"] textarea { color: #1e293b !important; font-family: 'Inter', sans-serif !important; font-size: 13.5px !important; background: transparent !important; }
[data-testid="stChatInput"] textarea::placeholder { color: #94a3b8 !important; }

/* ── METRICS ── */
[data-testid="stMetric"] { background: #ffffff !important; border: 1px solid #e2e8f0 !important; border-radius: 12px !important; padding: 14px 16px !important; box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important; }
[data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size: 9.5px !important; text-transform: uppercase !important; letter-spacing: 0.1em !important; font-weight: 600 !important; }
[data-testid="stMetricValue"] { color: #1e293b !important; font-family: 'JetBrains Mono', monospace !important; font-size: 1.35rem !important; font-weight: 600 !important; }

/* ── EXPANDER ── */
[data-testid="stExpander"] { background: #ffffff !important; border: 1px solid #e2e8f0 !important; border-radius: 12px !important; margin-bottom: 8px !important; box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important; }
[data-testid="stExpander"] summary { color: #64748b !important; font-size: 12.5px !important; font-weight: 500 !important; }
[data-testid="stExpander"] p, [data-testid="stExpander"] span { color: #64748b !important; }

/* ── SELECTBOX ── */
[data-testid="stSelectbox"] > div > div { background: #ffffff !important; border: 1px solid #e2e8f0 !important; border-radius: 10px !important; color: #334155 !important; }

/* ── TEXTAREA ── */
textarea { background: #ffffff !important; border: 1px solid #e2e8f0 !important; border-radius: 10px !important; color: #334155 !important; font-family: 'Inter', sans-serif !important; }
textarea::placeholder { color: #94a3b8 !important; }

/* ── BUTTONS ── */
.stButton > button { background: #1e40af !important; border: none !important; color: #ffffff !important; border-radius: 10px !important; font-family: 'Inter', sans-serif !important; font-weight: 500 !important; font-size: 13px !important; transition: all 0.2s !important; }
.stButton > button:hover { background: #1d4ed8 !important; }

/* ── PROGRESS BAR ── */
[data-testid="stProgressBar"] > div { background: #e2e8f0 !important; border-radius: 4px !important; }
[data-testid="stProgressBar"] > div > div { background: #3b82f6 !important; border-radius: 4px !important; }

/* ── ALERTS ── */
[data-testid="stAlert"] { border-radius: 10px !important; }

/* ── CUSTOM COMPONENTS ── */
.page-header {
    background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 60%, #2563eb 100%);
    border-radius: 16px;
    padding: 20px 26px;
    display: flex; align-items: center; gap: 14px;
    margin-bottom: 16px;
}
.ph-icon { width: 40px; height: 40px; background: rgba(255,255,255,0.15); border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 20px; flex-shrink:0; }
.ph-title { font-size: 1.1rem; font-weight: 700; color: #ffffff; letter-spacing: -0.3px; }
.ph-sub { font-size: 12px; color: #93c5fd; margin-top: 2px; }
.ph-pill { margin-left: auto; background: rgba(255,255,255,0.15); border: 1px solid rgba(255,255,255,0.25); border-radius: 20px; padding: 4px 13px; font-size: 11px; color: #bfdbfe; display: flex; align-items: center; gap: 6px; white-space:nowrap; }
.ph-dot { width: 7px; height: 7px; border-radius: 50%; background: #4ade80; }

.info-strip { background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 10px; padding: 10px 14px; font-size: 12.5px; color: #1e40af; margin-bottom: 14px; line-height: 1.6; }
.info-strip b { color: #1d4ed8; }

.verdict-card { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 16px; padding: 18px 20px; margin-bottom: 14px; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }
.vc-label { font-size: 9.5px; font-weight: 600; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 10px; }
.badge-fake { display: inline-flex; align-items: center; gap: 6px; background: #fef2f2; color: #dc2626; border: 1.5px solid #fca5a5; padding: 5px 18px; border-radius: 22px; font-weight: 700; font-size: 13px; }
.badge-real { display: inline-flex; align-items: center; gap: 6px; background: #f0fdf4; color: #16a34a; border: 1.5px solid #86efac; padding: 5px 18px; border-radius: 22px; font-weight: 700; font-size: 13px; }
.conf-text { font-size: 12.5px; color: #94a3b8; font-family: 'JetBrains Mono', monospace; }
.hl-text { font-size: 11.5px; color: #94a3b8; margin-top: 10px; padding-top: 10px; border-top: 1px solid #f1f5f9; }

.asp-row { display:flex; align-items:center; gap:10px; padding:6px 0; border-bottom:1px solid #f8fafc; }
.asp-name { font-size:11.5px; color:#475569; text-transform:capitalize; width:85px; flex-shrink:0; font-weight:500; }
.asp-bar-bg { flex:1; background:#f1f5f9; border-radius:4px; height:6px; overflow:hidden; }
.asp-fill { height:100%; border-radius:4px; }
.asp-bp { background:#22c55e; }
.asp-bn { background:#ef4444; }
.asp-bnu { background:#94a3b8; }
.asp-badge { font-size:10px; padding:2px 9px; border-radius:10px; font-weight:600; letter-spacing:0.02em; }
.bpos { background:#f0fdf4; color:#16a34a; border:1px solid #86efac; }
.bneg { background:#fef2f2; color:#dc2626; border:1px solid #fca5a5; }
.bneu { background:#f8fafc; color:#64748b; border:1px solid #e2e8f0; }
.asp-val { font-family:'JetBrains Mono',monospace; font-size:10.5px; color:#94a3b8; width:50px; text-align:right; }

.lime-box { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 14px; margin-top: 8px; }
.lime-title { font-size:9.5px; font-weight:600; color:#94a3b8; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px; }
.lime-row { display:flex; align-items:flex-start; gap:8px; margin-bottom:5px; }
.lime-dir { font-size:11.5px; color:#94a3b8; width:40px; flex-shrink:0; font-weight:500; }
.lw { display:inline-block; padding:2px 8px; border-radius:6px; margin:1px 2px; font-size:11px; font-weight:500; }
.lf { background:#fef2f2; color:#dc2626; border:1px solid #fca5a5; }
.lr { background:#f0fdf4; color:#16a34a; border:1px solid #86efac; }

.empty-panel { background:#ffffff; border:1px solid #e2e8f0; border-radius:16px; padding:52px 24px; text-align:center; box-shadow:0 1px 3px rgba(0,0,0,0.04); }
.empty-icon { font-size:2.2rem; margin-bottom:12px; opacity:0.3; }
.empty-title { font-size:14px; font-weight:600; color:#94a3b8; }
.empty-sub { font-size:12px; color:#cbd5e1; margin-top:5px; }

.eval-pass { background:#f0fdf4; border:1px solid #bbf7d0; border-radius:10px; padding:12px 14px; margin-bottom:7px; }
.eval-fail { background:#fef2f2; border:1px solid #fecaca; border-radius:10px; padding:12px 14px; margin-bottom:7px; }
.eval-q { font-size:13px; font-weight:500; color:#1e293b; margin-bottom:7px; }

.hist-row { background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:10px 14px; margin-bottom:6px; display:flex; align-items:center; gap:10px; }
.hist-badge { font-size:10.5px; padding:2px 10px; border-radius:10px; font-weight:600; flex-shrink:0; }
.hist-text { font-size:12.5px; color:#475569; flex:1; }
.hist-conf { font-size:10.5px; font-family:'JetBrains Mono',monospace; color:#94a3b8; }
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
def bar_color_cls(s):
    return "asp-bp" if s >= 0.05 else "asp-bn" if s <= -0.05 else "asp-bnu"

def badge_cls(s):
    return "bpos" if s >= 0.05 else "bneg" if s <= -0.05 else "bneu"

def badge_label(s):
    return "pos" if s >= 0.05 else "neg" if s <= -0.05 else "neu"

def extract_url(text):
    m = re.search(r"https?://\S+", text)
    return m.group(0) if m else None

def fetch_url(url):
    try:
        from newspaper import Article
        a = Article(url)
        a.download(); a.parse()
        return (a.title or "").strip(), (a.text or "").strip()
    except Exception as e:
        return None, str(e)

def render_result_panel(r):
    badge_cls_v = "fake" if r["prediction"] == "FAKE" else "real"
    emoji = "🚨" if r["prediction"] == "FAKE" else "✅"

    hl = r.get("headline", "")
    hl_html = f'<div class="hl-text">"{hl[:90]}{"…" if len(hl)>90 else ""}"</div>' if hl else ""

    st.markdown(f"""
<div class="verdict-card">
  <div class="vc-label">Prediction</div>
  <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap">
    <span class="badge-{badge_cls_v}">{emoji} {r['prediction']}</span>
    <span class="conf-text">{r['confidence']}% confidence</span>
  </div>
  {hl_html}
</div>""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Sentiment", r["overall_label"].capitalize())
    c2.metric("VADER", r["overall_score"])
    c3.metric("Subjectivity", r["subjectivity"])

    detected = r["detected"]
    if not detected:
        st.markdown('<div style="font-size:12px;color:#1e3050;padding:10px 0">No aspect keywords detected.</div>', unsafe_allow_html=True)
        return

    # Charts
    t1, t2 = st.tabs(["Radar", "Bar"])
    labels = list(detected.keys())
    scores = list(detected.values())

    chart_colors = {
        "paper_bgcolor": "#f5f7fa",
        "plot_bgcolor":  "#ffffff",
        "grid":          "#e2e8f0",
        "tick":          "#94a3b8",
        "line":          "#3b82f6",
        "fill":          "rgba(59,130,246,0.08)",
    }

    with t1:
        norm = [(s + 1) / 2 for s in scores]
        fig  = go.Figure(go.Scatterpolar(
            r=norm + [norm[0]], theta=labels + [labels[0]],
            fill="toself",
            fillcolor=chart_colors["fill"],
            line=dict(color=chart_colors["line"], width=2),
            marker=dict(size=5, color=chart_colors["line"]),
        ))
        fig.update_layout(
            polar=dict(
                bgcolor=chart_colors["plot_bgcolor"],
                radialaxis=dict(visible=True, range=[0,1],
                    tickfont=dict(color=chart_colors["tick"], size=8),
                    gridcolor=chart_colors["grid"], linecolor=chart_colors["grid"]),
                angularaxis=dict(
                    tickfont=dict(color=chart_colors["tick"], size=9, family="DM Sans"),
                    gridcolor=chart_colors["grid"], linecolor=chart_colors["grid"]),
            ),
            paper_bgcolor=chart_colors["paper_bgcolor"],
            margin=dict(l=28,r=28,t=18,b=18), height=240, showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        bar_colors = ["#1a6e3a" if s >= 0.05 else "#6e1a1a" if s <= -0.05 else "#1a3a6e" for s in scores]
        fig2 = go.Figure(go.Bar(
            x=[l.capitalize() for l in labels], y=scores,
            marker_color=bar_colors, marker_line_width=0,
            text=[f"{s:+.2f}" for s in scores],
            textposition="outside",
            textfont=dict(size=9, color=chart_colors["tick"]),
        ))
        fig2.update_layout(
            paper_bgcolor=chart_colors["paper_bgcolor"],
            plot_bgcolor=chart_colors["plot_bgcolor"],
            font=dict(family="DM Sans", color=chart_colors["tick"]),
            xaxis=dict(tickfont=dict(size=9, color=chart_colors["tick"]), gridcolor=chart_colors["grid"]),
            yaxis=dict(range=[-1.25,1.25], gridcolor=chart_colors["grid"],
                       zeroline=True, zerolinecolor="#1a3050", zerolinewidth=1),
            margin=dict(l=8,r=8,t=18,b=8), height=190, showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Aspect breakdown
    st.markdown('<div style="font-size:9px;font-weight:600;color:#1e3050;text-transform:uppercase;letter-spacing:0.1em;margin:4px 0 8px">Aspect breakdown</div>', unsafe_allow_html=True)
    for asp, score in detected.items():
        pct = int((score + 1) / 2 * 100)
        bc  = bar_color_cls(score)
        bl  = badge_label(score)
        bcl = badge_cls(score)
        st.markdown(f"""
<div class="asp-row">
  <span class="asp-name">{asp}</span>
  <div class="asp-bar-bg"><div class="asp-fill {bc}" style="width:{pct}%"></div></div>
  <span class="asp-badge {bcl}">{bl}</span>
  <span class="asp-val">{score:+.3f}</span>
</div>""", unsafe_allow_html=True)

    # LIME signals
    fw = r.get("lime_fake", [])
    rw = r.get("lime_real", [])
    fp = " ".join(f'<span class="lw lf">{w}</span>' for w in fw) if fw else '<span style="color:#1e3050;font-size:10px">none</span>'
    rp = " ".join(f'<span class="lw lr">{w}</span>' for w in rw) if rw else '<span style="color:#1e3050;font-size:10px">none</span>'
    st.markdown(f"""
<div class="lime-box">
  <div class="lime-title">LIME word signals</div>
  <div class="lime-row"><span class="lime-dir">→ fake</span><div>{fp}</div></div>
  <div class="lime-row"><span class="lime-dir">→ real</span><div>{rp}</div></div>
</div>""", unsafe_allow_html=True)

def make_chat_reply(r):
    asp = ", ".join(r["detected"].keys()) or "none detected"
    e   = "🚨" if r["prediction"] == "FAKE" else "✅"
    return f"""{e} **{r['prediction']}** — {r['confidence']}% confidence

**Sentiment:** {r['overall_label'].capitalize()} · **VADER:** {r['overall_score']} · **Subjectivity:** {r['subjectivity']}

**Aspects:** {asp}

Full breakdown in the panel → Ask *"why {r['prediction'].lower()}?"* or paste another article."""

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
<div style="padding:18px 16px 14px;border-bottom:1px solid #0f1f3d;margin-bottom:4px">
  <div style="font-size:15px;font-weight:600;color:#3d6fa8;letter-spacing:-0.3px">📰 ABSA Chatbot</div>
  <div style="font-size:10px;color:#1e3050;margin-top:3px">Charles Darwin University · 2026</div>
  <div style="font-size:10px;color:#162538;margin-top:1px">Bhandari · Patel · Rahaman Mohammed</div>
</div>""", unsafe_allow_html=True)

    st.markdown("### Try a headline")
    examples = {
        "🚨  5G conspiracy":    "EXPOSED: Government secretly controlling population through 5G towers and vaccines",
        "📈  Fed rate rise":    "Federal Reserve raises interest rates by 0.25 percent following inflation data",
        "🗳  Election fraud":   "SHOCKING: Democrats rigging election through mail-in ballot fraud across swing states",
        "🌍  Climate warning":  "Climate scientists warn record carbon emissions risk catastrophic warming as governments stall",
        "💊  Pharma cover-up":  "Big Pharma hiding cancer cure to protect vaccine profits — doctors speak out",
        "🏫  Enrolment drop":   "Universities report sharp decline in student enrolment for third consecutive year",
        "🔫  Crime report":     "Police arrest three suspects following armed robbery at downtown bank branch",
        "🪖  Military aid":     "Pentagon announces new military aid package for Ukraine amid ongoing conflict",
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

    if st.session_state.history:
        fakes = sum(1 for h in st.session_state.history if h["result"]["prediction"] == "FAKE")
        reals = len(st.session_state.history) - fakes
        c1, c2 = st.columns(2)
        c1.metric("🚨 Fake", fakes)
        c2.metric("✅ Real", reals)

    if st.button("🗑 Clear chat", use_container_width=True):
        st.session_state.messages    = []
        st.session_state.last_result = None
        st.session_state.history     = []
        st.rerun()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_chat, tab_eval = st.tabs(["💬   Expert Chatbot", "📊   Performance Evaluation"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CHAT
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    col_chat, col_panel = st.columns([3, 2], gap="large")

    with col_chat:
        st.markdown("""
<div class="page-header">
  <div class="ph-icon">🧠</div>
  <div>
    <div class="ph-title">ABSA Expert Chatbot</div>
    <div class="ph-sub">Fake news detection · Aspect-based sentiment analysis · AI-powered</div>
  </div>
  <div class="ph-pill"><div class="ph-dot"></div> Online</div>
</div>
""", unsafe_allow_html=True)

        st.markdown("""
<div class="info-strip" style="margin-top:14px">
  Paste a <b>headline</b>, full <b>article</b>, or a <b>URL</b> to analyse —
  or ask any expert question about the ABSA project.
</div>""", unsafe_allow_html=True)

        if not st.session_state.messages:
            with st.chat_message("assistant"):
                st.markdown("""Hi! I'm the **ABSA Fake News Expert** from Charles Darwin University (2026).

**What I can do:**
- 🔍 Analyse any **headline, article, or URL** → fake/real + 11-aspect sentiment breakdown
- 💡 Answer **expert questions** about ABSA, VADER, BERT, LIME, datasets, models
- 📊 **Evaluate performance** on preset or custom articles

Try pasting a headline or pick one from the sidebar.""")

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        prefill    = st.session_state.pop("_prefill", None)
        user_input = st.chat_input("Paste headline, article, URL — or ask anything…")
        if prefill and not user_input:
            user_input = prefill

        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                url = extract_url(user_input)

                if url or is_article_input(user_input):
                    with st.spinner("Running ABSA pipeline…"):
                        if url:
                            title, body = fetch_url(url)
                            if title or body:
                                result = predict_article(headline=(title or ""), body=(body or ""))
                                if result:
                                    result["headline"] = title or url[:80]
                            else:
                                result = {"error": f"Could not fetch URL — try pasting the text directly.\n\n{body}"}
                        else:
                            result = predict_article(headline=user_input)

                    if result and "error" not in result:
                        st.session_state.last_result = result
                        st.session_state.history.append({"text": user_input[:80], "result": result})
                        reply = make_chat_reply(result)
                    else:
                        reply = f"⚠️ {result.get('error', 'Could not analyse.') if result else 'Analysis failed.'}"
                else:
                    with st.spinner("Thinking…"):
                        reply = get_gemini_response(user_input, st.session_state.messages[:-1])

                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})

    with col_panel:
        st.markdown('<div style="font-size:9px;font-weight:600;color:#1e3050;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:12px">Analysis Results</div>', unsafe_allow_html=True)

        if st.session_state.last_result and "error" not in st.session_state.last_result:
            render_result_panel(st.session_state.last_result)

            with st.expander("All 11 aspect scores"):
                for asp in aspects:
                    score = st.session_state.last_result["absa_scores"].get(asp, 0.0)
                    pct   = int((score + 1) / 2 * 100)
                    bc    = bar_color_cls(score)
                    bl    = badge_label(score)
                    bcl   = badge_cls(score)
                    st.markdown(f"""
<div class="asp-row">
  <span class="asp-name">{asp}</span>
  <div class="asp-bar-bg"><div class="asp-fill {bc}" style="width:{pct}%"></div></div>
  <span class="asp-badge {bcl}">{bl}</span>
  <span class="asp-val">{score:+.3f}</span>
</div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
<div class="empty-panel">
  <div class="empty-icon">📰</div>
  <div class="empty-title">Results appear here</div>
  <div class="empty-sub">Paste a headline or article in the chat</div>
</div>""", unsafe_allow_html=True)

    # History
    if len(st.session_state.history) > 0:
        st.markdown('<div style="font-size:9px;font-weight:600;color:#1e3050;text-transform:uppercase;letter-spacing:0.1em;margin:16px 0 8px">Session history</div>', unsafe_allow_html=True)
        for h in reversed(st.session_state.history[-6:]):
            r   = h["result"]
            ico = "🚨" if r["prediction"] == "FAKE" else "✅"
            cls = "bneg" if r["prediction"] == "FAKE" else "bpos"
            asp = ", ".join(list(r["detected"].keys())[:3])
            if len(r["detected"]) > 3:
                asp += "…"
            st.markdown(f"""
<div class="hist-row">
  <span class="hist-badge {cls}">{ico} {r['prediction']}</span>
  <span class="hist-text">{h['text'][:55]}{"…" if len(h['text'])>55 else ""}</span>
  <span class="hist-conf">{r['confidence']}%</span>
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.markdown("""
<div class="page-header" style="margin-bottom:18px">
  <div class="ph-icon">📊</div>
  <div>
    <div class="ph-title">Performance Evaluation</div>
    <div class="ph-sub">Test the ABSA pipeline against labelled articles</div>
  </div>
</div>""", unsafe_allow_html=True)

    eval_tab1, eval_tab2 = st.tabs(["📋   Preset QA Dataset", "✏️   Test Your Own Article"])

    with eval_tab1:
        st.markdown("""
<div class="info-strip">
  Evaluates the pipeline against <b>10 pre-labelled headlines</b> covering all 11 aspect categories.
  Results compared against Model A (67.19%), Model B (66.86%), and Model C (49%) from the notebook.
</div>""", unsafe_allow_html=True)

        col_b, col_d = st.columns([1, 2])
        with col_b:
            run_preset = st.button("▶ Run evaluation", use_container_width=True)
        with col_d:
            st.markdown('<div style="font-size:12px;color:#1e3050;padding-top:8px">10 labelled headlines · exact-match accuracy · all 11 aspects</div>', unsafe_allow_html=True)

        if run_preset:
            with st.spinner("Running pipeline on 10 headlines…"):
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
            m1.metric("Accuracy",     f"{acc}%")
            m2.metric("Correct",      f"{correct}/{n}")
            m3.metric("FAKE correct", f"{sum(1 for r in results if r['passed'] and r['expected']=='FAKE')}/{sum(1 for r in results if r['expected']=='FAKE')}")
            m4.metric("REAL correct", f"{sum(1 for r in results if r['passed'] and r['expected']=='REAL')}/{sum(1 for r in results if r['expected']=='REAL')}")

            st.markdown('<div style="font-size:9px;font-weight:600;color:#1e3050;text-transform:uppercase;letter-spacing:0.1em;margin:14px 0 8px">Comparison with notebook models</div>', unsafe_allow_html=True)

            fig = go.Figure(go.Bar(
                x=["Model A\n(TF-IDF)", "Model B\n(TF-IDF+Sent)", "Naive Bayes", "Model C\n(RoBERTa)", f"This chatbot\n({acc}%)"],
                y=[67.19, 66.86, 64.0, 49.0, acc],
                marker_color=["#3b82f6","#3b82f6","#3b82f6","#ef4444","#1a6e3a" if acc >= 60 else "#6e3a1a"],
                marker_line_width=0,
                text=[f"{v}%" for v in [67.19, 66.86, 64.0, 49.0, acc]],
                textposition="outside",
                textfont=dict(size=10, color="#64748b"),
            ))
            fig.update_layout(
                paper_bgcolor="#f5f7fa", plot_bgcolor="#ffffff",
                font=dict(family="Inter", color="#94a3b8"),
                xaxis=dict(tickfont=dict(size=10, color="#94a3b8"), gridcolor="#e2e8f0"),
                yaxis=dict(range=[0,105], gridcolor="#e2e8f0", title="Accuracy (%)"),
                margin=dict(l=10,r=10,t=30,b=10), height=260,
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown('<div style="font-size:9px;font-weight:600;color:#1e3050;text-transform:uppercase;letter-spacing:0.1em;margin:6px 0 10px">Detailed results</div>', unsafe_allow_html=True)
            for r in results:
                icon = "✅" if r["passed"] else "❌"
                cls  = "eval-pass" if r["passed"] else "eval-fail"
                eb   = "bpos" if r["expected"]  == "REAL" else "bneg"
                pb   = "bpos" if r["predicted"] == "REAL" else "bneg"
                st.markdown(f"""
<div class="{cls}">
  <div class="eval-q">{icon} {r['question']}</div>
  <div style="display:flex;flex-wrap:wrap;gap:7px;align-items:center">
    <span style="font-size:10px;color:#1e3050">Expected:</span>
    <span class="asp-badge {eb}">{r['expected']}</span>
    <span style="font-size:10px;color:#1e3050">Got:</span>
    <span class="asp-badge {pb}">{r['predicted']}</span>
    <span style="font-size:10px;font-family:'DM Mono',monospace;color:#1e3050">{r['confidence']}% conf</span>
    <span style="font-size:10px;background:#0a1830;padding:1px 7px;border-radius:6px;color:#2d4460">{r['category']}</span>
  </div>
</div>""", unsafe_allow_html=True)

    with eval_tab2:
        st.markdown("""
<div class="info-strip">
  Test <b>your own article</b> — paste text or a URL, set the expected label, and see if the pipeline gets it right.
</div>""", unsafe_allow_html=True)

        custom_text = st.text_area(
            "Article", height=110,
            placeholder="Paste headline, full article, or https://...",
            label_visibility="collapsed",
        )
        col_l, col_r = st.columns([1,1])
        with col_l:
            expected_label = st.selectbox("Expected label", ["FAKE","REAL"], key="custom_expected")
        with col_r:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            run_custom = st.button("▶ Analyse", use_container_width=True, key="run_custom")

        if run_custom:
            if not custom_text.strip():
                st.warning("Please paste some text or a URL.")
            else:
                with st.spinner("Analysing…"):
                    url = extract_url(custom_text)
                    if url:
                        title, body = fetch_url(url)
                        result = predict_article(headline=(title or ""), body=(body or "")) if (title or body) else {"error": body}
                        if result and "error" not in result:
                            result["headline"] = title or url[:80]
                    else:
                        result = predict_article(headline=custom_text)

                if result and "error" not in result:
                    predicted = result["prediction"]
                    passed    = predicted == expected_label
                    icon      = "✅ Correct!" if passed else "❌ Incorrect"
                    strip_cls = "info-strip" if passed else "info-strip" 
                    color     = "#1a6e3a" if passed else "#6e1a1a"
                    border    = "#153d22" if passed else "#3d1515"
                    st.markdown(f"""
<div style="background:{'#0a1f12' if passed else '#1f0a0a'};border:1px solid {border};border-radius:10px;padding:11px 16px;font-size:13px;color:{'#4ade80' if passed else '#f87171'};margin-bottom:14px">
  <b>{icon}</b> — Predicted <b>{predicted}</b> ({result['confidence']}% confidence), expected <b>{expected_label}</b>
</div>""", unsafe_allow_html=True)
                    render_result_panel(result)
                    st.session_state.last_result = result
                    st.session_state.history.append({"text": custom_text[:80], "result": result})
                else:
                    st.error(result.get("error", "Analysis failed.") if result else "Analysis failed.")
