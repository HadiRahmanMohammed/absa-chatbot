"""
app.py — ABSA Expert Chatbot v6 — Premium UI
Assignment 3 — Charles Darwin University (2026)
Bhandari · Patel · Rahaman Mohammed
"""

import re
import streamlit as st
import plotly.graph_objects as go
from absa_engine import predict_article, QA_EVAL, aspects
from gemini_brain import get_gemini_response, is_article_input

st.set_page_config(
    page_title="ABSA Expert Chatbot",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ─── GLOBAL ─────────────────────────────────────── */
*, html, body { box-sizing: border-box; }
html, body, .stApp, [class*="css"], section, .main,
.block-container, [data-testid="stAppViewContainer"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background: #050d1a !important;
    color: #7ba3cc !important;
}
.block-container { padding: 1.5rem 2rem 3rem !important; max-width: 100% !important; }
#MainMenu, footer, header, .stDeployButton,
[data-testid="stToolbar"], .stApp > header { display: none !important; visibility: hidden !important; }

/* ─── SIDEBAR ────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #030912 !important;
    border-right: 1px solid #0a1628 !important;
    min-width: 220px !important;
    max-width: 220px !important;
}
[data-testid="stSidebar"] > div { background: #030912 !important; padding-top: 0 !important; }
[data-testid="stSidebar"] * {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background: transparent !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div { color: #2a4060 !important; font-size: 12px !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #1a3050 !important; font-size: 9px !important; font-weight: 600 !important;
    text-transform: uppercase !important; letter-spacing: 0.12em !important;
}
[data-testid="stSidebar"] hr { border: none !important; border-top: 1px solid #0a1628 !important; margin: 6px 0 !important; }
[data-testid="stSidebar"] .stButton > button {
    background: transparent !important; border: none !important;
    color: #2a4060 !important; font-size: 12px !important;
    padding: 6px 10px !important; text-align: left !important;
    width: 100% !important; border-radius: 7px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    transition: all 0.15s !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #0a1628 !important; color: #4a80b0 !important;
}
[data-testid="stSidebar"] [data-testid="stMetric"] {
    background: #07101c !important; border: 1px solid #0a1628 !important;
    border-radius: 8px !important; padding: 10px 12px !important;
}
[data-testid="stSidebar"] [data-testid="stMetricLabel"] { color: #1a3050 !important; font-size: 9px !important; text-transform: uppercase !important; letter-spacing: 0.1em !important; }
[data-testid="stSidebar"] [data-testid="stMetricValue"] { color: #2d5a8a !important; font-size: 1.4rem !important; font-family: 'JetBrains Mono', monospace !important; }

/* ─── TABS ───────────────────────────────────────── */
[data-baseweb="tab-list"] {
    background: #07101c !important; border-radius: 12px !important;
    border: 1px solid #0a1628 !important; padding: 4px !important; gap: 3px !important;
}
[data-baseweb="tab"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 13px !important; font-weight: 500 !important;
    border-radius: 9px !important; color: #2a4060 !important;
    padding: 8px 20px !important; transition: all 0.2s !important;
    background: transparent !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    background: #0d2040 !important; color: #5a9ad4 !important;
}

/* ─── CHAT MESSAGES ──────────────────────────────── */
[data-testid="stChatMessage"] {
    background: #07101c !important; border: 1px solid #0a1628 !important;
    border-radius: 14px !important; margin-bottom: 8px !important;
    padding: 4px !important;
}
[data-testid="stChatMessage"] p { color: #5a8ab4 !important; font-size: 13.5px !important; line-height: 1.65 !important; }
[data-testid="stChatMessage"] strong { color: #7aaccf !important; }
[data-testid="stChatMessage"] code { color: #4a8ab4 !important; background: #0a1628 !important; border-radius: 4px !important; padding: 1px 5px !important; font-family: 'JetBrains Mono', monospace !important; }
[data-testid="stChatMessage"] li { color: #4a7aa4 !important; }
[data-testid="stChatMessage"] td, [data-testid="stChatMessage"] th { color: #4a7aa4 !important; border-color: #0a1628 !important; }

/* ─── CHAT INPUT ─────────────────────────────────── */
[data-testid="stChatInput"] > div {
    background: #07101c !important; border: 1.5px solid #0d2040 !important;
    border-radius: 14px !important;
}
[data-testid="stChatInput"] textarea {
    color: #5a8ab4 !important; font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 13.5px !important; background: transparent !important;
    caret-color: #2a6abf !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #152535 !important; }
[data-testid="stChatInput"] button { background: #0d2040 !important; border-radius: 8px !important; }

/* ─── METRICS ────────────────────────────────────── */
[data-testid="stMetric"] {
    background: #07101c !important; border: 1px solid #0a1628 !important;
    border-radius: 12px !important; padding: 14px 16px !important;
}
[data-testid="stMetricLabel"] { color: #1a3050 !important; font-size: 9px !important; text-transform: uppercase !important; letter-spacing: 0.1em !important; font-weight: 600 !important; }
[data-testid="stMetricValue"] { color: #5a9ad4 !important; font-family: 'JetBrains Mono', monospace !important; font-size: 1.35rem !important; font-weight: 500 !important; }

/* ─── EXPANDER ───────────────────────────────────── */
[data-testid="stExpander"] { background: #07101c !important; border: 1px solid #0a1628 !important; border-radius: 12px !important; margin-bottom: 8px !important; }
[data-testid="stExpander"] summary { color: #2a4060 !important; font-size: 12px !important; }
[data-testid="stExpander"] p, [data-testid="stExpander"] span { color: #2a4060 !important; font-size: 12px !important; }

/* ─── BUTTONS ────────────────────────────────────── */
.stButton > button {
    background: #0d2040 !important; border: 1px solid #163060 !important;
    color: #4a8ab4 !important; border-radius: 10px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 500 !important; transition: all 0.2s !important;
    font-size: 13px !important;
}
.stButton > button:hover { background: #112a50 !important; color: #7aaccf !important; border-color: #1e4070 !important; }

/* ─── SELECT / TEXTAREA ──────────────────────────── */
[data-testid="stSelectbox"] > div > div {
    background: #07101c !important; border: 1px solid #0a1628 !important;
    border-radius: 10px !important; color: #4a7aa4 !important;
}
textarea {
    background: #07101c !important; border: 1px solid #0a1628 !important;
    border-radius: 10px !important; color: #5a8ab4 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
textarea::placeholder { color: #101e30 !important; }

/* ─── PROGRESS BAR ───────────────────────────────── */
[data-testid="stProgressBar"] > div { background: #07101c !important; border-radius: 4px !important; }
[data-testid="stProgressBar"] > div > div { background: #1a4a8a !important; border-radius: 4px !important; }

/* ─── ALERTS ─────────────────────────────────────── */
[data-testid="stAlert"] { background: #07101c !important; border: 1px solid #0a1628 !important; border-radius: 10px !important; }
[data-testid="stAlert"] p { color: #3a6a9a !important; }

/* ─── CUSTOM COMPONENTS ──────────────────────────── */

/* Page header */
.ph { background: #07101c; border: 1px solid #0a1628; border-radius: 16px; padding: 18px 24px; display: flex; align-items: center; gap: 14px; margin-bottom: 16px; }
.ph-icon { width: 40px; height: 40px; background: #0a1628; border: 1px solid #0d2040; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 20px; flex-shrink: 0; }
.ph-title { font-size: 1.05rem; font-weight: 600; color: #5a9ad4; letter-spacing: -0.2px; }
.ph-sub { font-size: 11px; color: #1a3050; margin-top: 2px; }
.ph-pill { margin-left: auto; background: #0a1628; border: 1px solid #0d2040; border-radius: 20px; padding: 4px 13px; font-size: 11px; color: #1e4060; display: flex; align-items: center; gap: 6px; white-space: nowrap; }
.ph-dot { width: 6px; height: 6px; border-radius: 50%; background: #1a4a7a; animation: pulse-dot 2s infinite; }
@keyframes pulse-dot { 0%,100%{opacity:0.5} 50%{opacity:1} }

/* Info strip */
.strip { background: #060f1c; border: 1px solid #0a1628; border-radius: 10px; padding: 10px 14px; font-size: 12.5px; color: #1e3a5a; margin-bottom: 14px; line-height: 1.6; }
.strip b { color: #2a5070; }

/* Verdict */
.vcard { background: #07101c; border: 1px solid #0a1628; border-radius: 14px; padding: 16px 18px; margin-bottom: 12px; }
.vcard-lbl { font-size: 9px; font-weight: 600; color: #101e30; text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 10px; }
.badge-fake { display: inline-flex; align-items: center; gap: 6px; background: #1a0808; color: #e05a5a; border: 1px solid #2d1010; padding: 5px 16px; border-radius: 20px; font-weight: 600; font-size: 13px; }
.badge-real { display: inline-flex; align-items: center; gap: 6px; background: #081a0e; color: #4ab870; border: 1px solid #0f2d18; padding: 5px 16px; border-radius: 20px; font-weight: 600; font-size: 13px; }
.vconf { font-size: 12px; color: #152535; font-family: 'JetBrains Mono', monospace; }
.vhl { font-size: 11px; color: #0f1e2d; margin-top: 8px; padding-top: 8px; border-top: 1px solid #0a1628; }

/* Aspect rows */
.ar { display:flex; align-items:center; gap:8px; padding:5px 0; border-bottom:1px solid #060f1c; }
.an { font-size:11px; color:#1e3a5a; text-transform:capitalize; width:82px; flex-shrink:0; font-weight:500; }
.abg { flex:1; background:#060f1c; border-radius:3px; height:4px; overflow:hidden; }
.af { height:100%; border-radius:3px; }
.af-p { background:#1a5a30; }
.af-n { background:#5a1a1a; }
.af-u { background:#0d2040; }
.abadge { font-size:9px; padding:1px 7px; border-radius:8px; font-weight:600; letter-spacing:0.02em; }
.bp { background:#081a0e; color:#4ab870; border:1px solid #0f2d18; }
.bn { background:#1a0808; color:#e05a5a; border:1px solid #2d1010; }
.bnu { background:#07101c; color:#2a5070; border:1px solid #0a1628; }
.av { font-family:'JetBrains Mono',monospace; font-size:10px; color:#0f1e2d; width:46px; text-align:right; }

/* LIME box */
.limebox { background:#040c18; border:1px solid #0a1628; border-radius:10px; padding:10px 14px; margin-top:6px; }
.lime-lbl { font-size:9px; font-weight:600; color:#0f1e2d; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px; }
.lrow { display:flex; align-items:flex-start; gap:6px; margin-bottom:5px; }
.ldir { font-size:11px; color:#0f1e2d; width:36px; flex-shrink:0; font-weight:500; }
.lw { display:inline-block; padding:1px 7px; border-radius:6px; margin:1px 2px; font-size:10px; font-weight:500; }
.lf { background:#1a0808; color:#e05a5a; border:1px solid #2d1010; }
.lr { background:#081a0e; color:#4ab870; border:1px solid #0f2d18; }

/* Empty state */
.empty { background:#07101c; border:1px solid #0a1628; border-radius:14px; padding:52px 24px; text-align:center; }
.empty-ico { font-size:2rem; margin-bottom:10px; opacity:0.2; }
.empty-t { font-size:13px; font-weight:500; color:#0f1e2d; }
.empty-s { font-size:11px; color:#0a1628; margin-top:4px; }

/* History rows */
.hrow { background:#040c18; border:1px solid #0a1628; border-radius:10px; padding:9px 14px; margin-bottom:5px; display:flex; align-items:center; gap:10px; }
.hbadge { font-size:10px; padding:2px 9px; border-radius:10px; font-weight:600; flex-shrink:0; }
.htxt { font-size:12px; color:#1e3a5a; flex:1; }
.hconf { font-size:10px; font-family:'JetBrains Mono',monospace; color:#0f1e2d; }

/* Eval rows */
.epass { background:#060f0a; border:1px solid #0f2d18; border-radius:10px; padding:11px 14px; margin-bottom:6px; }
.efail { background:#0f0606; border:1px solid #2d1010; border-radius:10px; padding:11px 14px; margin-bottom:6px; }
.eq { font-size:12.5px; font-weight:500; color:#2a5070; margin-bottom:6px; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in {"messages":[],"last_result":None,"history":[],"eval_results":[],"eval_done":False}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Helpers ───────────────────────────────────────────────────────────────────
def bc(s): return "#1a5a30" if s>=0.05 else "#5a1a1a" if s<=-0.05 else "#0d2040"
def bc_cls(s): return "af-p" if s>=0.05 else "af-n" if s<=-0.05 else "af-u"
def badge_cls(s): return "bp" if s>=0.05 else "bn" if s<=-0.05 else "bnu"
def badge_lbl(s): return "pos" if s>=0.05 else "neg" if s<=-0.05 else "neu"
def extract_url(t): m=re.search(r"https?://\S+",t); return m.group(0) if m else None

def fetch_url(url):
    try:
        from newspaper import Article
        a=Article(url); a.download(); a.parse()
        return (a.title or "").strip(), (a.text or "").strip()
    except Exception as e:
        return None, str(e)

def chart_layout(h=250):
    return dict(
        polar=dict(
            bgcolor="#07101c",
            radialaxis=dict(visible=True,range=[0,1],tickfont=dict(color="#0f1e2d",size=8),gridcolor="#0a1628",linecolor="#0a1628"),
            angularaxis=dict(tickfont=dict(color="#1e3a5a",size=9,family="Plus Jakarta Sans"),gridcolor="#0a1628",linecolor="#0a1628"),
        ),
        paper_bgcolor="#050d1a", margin=dict(l=28,r=28,t=18,b=18), height=h, showlegend=False,
    )

def render_panel(r):
    bc_v = "fake" if r["prediction"]=="FAKE" else "real"
    emoji = "🚨" if r["prediction"]=="FAKE" else "✅"
    hl = r.get("headline","")
    hl_html = f'<div class="vhl">"{hl[:95]}{"…" if len(hl)>95 else ""}"</div>' if hl else ""

    st.markdown(f"""
<div class="vcard">
  <div class="vcard-lbl">Prediction</div>
  <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap">
    <span class="badge-{bc_v}">{emoji} {r['prediction']}</span>
    <span class="vconf">{r['confidence']}% confidence</span>
  </div>
  {hl_html}
</div>""", unsafe_allow_html=True)

    c1,c2,c3 = st.columns(3)
    c1.metric("Sentiment", r["overall_label"].capitalize())
    c2.metric("VADER",     r["overall_score"])
    c3.metric("Subjectivity", r["subjectivity"])

    det = r["detected"]
    if not det:
        st.markdown('<div style="font-size:12px;color:#0f1e2d;padding:10px 0">No aspect keywords detected.</div>', unsafe_allow_html=True)
        return

    t1,t2 = st.tabs(["Radar","Bar"])
    labels = list(det.keys()); scores = list(det.values())

    with t1:
        norm = [(s+1)/2 for s in scores]
        fig = go.Figure(go.Scatterpolar(
            r=norm+[norm[0]], theta=labels+[labels[0]],
            fill="toself", fillcolor="rgba(26,74,138,0.12)",
            line=dict(color="#1a4a8a",width=2), marker=dict(size=5,color="#1a4a8a"),
        ))
        fig.update_layout(**chart_layout(240))
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        fig2 = go.Figure(go.Bar(
            x=[l.capitalize() for l in labels], y=scores,
            marker_color=[bc(s) for s in scores], marker_line_width=0,
            text=[f"{s:+.2f}" for s in scores], textposition="outside",
            textfont=dict(size=9,color="#1e3a5a"),
        ))
        fig2.update_layout(
            paper_bgcolor="#050d1a", plot_bgcolor="#07101c",
            xaxis=dict(tickfont=dict(size=9,color="#1e3a5a"),gridcolor="#0a1628"),
            yaxis=dict(range=[-1.25,1.25],gridcolor="#0a1628",zeroline=True,zerolinecolor="#0d2040",zerolinewidth=1),
            margin=dict(l=8,r=8,t=18,b=8), height=195, showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div style="font-size:9px;font-weight:600;color:#0f1e2d;text-transform:uppercase;letter-spacing:0.12em;margin:4px 0 8px">Aspect breakdown</div>', unsafe_allow_html=True)
    for asp,score in det.items():
        pct = int((score+1)/2*100)
        st.markdown(f"""
<div class="ar">
  <span class="an">{asp}</span>
  <div class="abg"><div class="af {bc_cls(score)}" style="width:{pct}%"></div></div>
  <span class="abadge {badge_cls(score)}">{badge_lbl(score)}</span>
  <span class="av">{score:+.3f}</span>
</div>""", unsafe_allow_html=True)

    fw = r.get("lime_fake",[]); rw = r.get("lime_real",[])
    fp = " ".join(f'<span class="lw lf">{w}</span>' for w in fw) if fw else '<span style="color:#0f1e2d;font-size:10px">none</span>'
    rp = " ".join(f'<span class="lw lr">{w}</span>' for w in rw) if rw else '<span style="color:#0f1e2d;font-size:10px">none</span>'
    st.markdown(f"""
<div class="limebox">
  <div class="lime-lbl">LIME word signals</div>
  <div class="lrow"><span class="ldir">→ fake</span><div>{fp}</div></div>
  <div class="lrow"><span class="ldir">→ real</span><div>{rp}</div></div>
</div>""", unsafe_allow_html=True)

def chat_reply(r):
    asp = ", ".join(r["detected"].keys()) or "none"
    e = "🚨" if r["prediction"]=="FAKE" else "✅"
    return f"""{e} **{r['prediction']}** — {r['confidence']}% confidence

**Sentiment:** {r['overall_label'].capitalize()} · **VADER:** {r['overall_score']} · **Subjectivity:** {r['subjectivity']}

**Aspects:** {asp}

Full breakdown in the results panel → ask *"why {r['prediction'].lower()}?"* for explanation."""

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
<div style="padding:16px 14px 12px;border-bottom:1px solid #0a1628;margin-bottom:4px">
  <div style="font-size:14px;font-weight:600;color:#2a5070;letter-spacing:-0.2px">📰 ABSA Chatbot</div>
  <div style="font-size:10px;color:#0f1e2d;margin-top:3px">Charles Darwin University · 2026</div>
  <div style="font-size:10px;color:#0a1820;margin-top:1px">Bhandari · Patel · Rahaman Mohammed</div>
</div>""", unsafe_allow_html=True)

    st.markdown("### Try a headline")
    for label,text in {
        "🚨  5G conspiracy":   "EXPOSED: Government secretly controlling population through 5G towers and vaccines",
        "📈  Fed rate rise":   "Federal Reserve raises interest rates by 0.25 percent following inflation data",
        "🗳  Election fraud":  "SHOCKING: Democrats rigging election through mail-in ballot fraud across swing states",
        "🌍  Climate warning": "Climate scientists warn record carbon emissions risk catastrophic warming as governments stall",
        "💊  Pharma cover-up": "Big Pharma hiding cancer cure to protect vaccine profits — doctors speak out",
        "🏫  Enrolment drop":  "Universities report sharp decline in student enrolment for third consecutive year",
        "🔫  Crime report":    "Police arrest three suspects following armed robbery at downtown bank branch",
        "🪖  Military aid":    "Pentagon announces new military aid package for Ukraine amid ongoing conflict",
    }.items():
        if st.button(label, use_container_width=True): st.session_state["_pre"] = text

    st.divider()
    st.markdown("### Ask the expert")
    for q in ["What is ABSA?","How does VADER work?","How does BERT work?",
               "How does the hybrid approach work?","What are the 11 aspects?",
               "How is fake/real classified?","What were the model accuracies?",
               "What is LIME explainability?","What datasets were used?",
               "What are the ethical risks?","Who built this project?"]:
        if st.button(q, use_container_width=True): st.session_state["_pre"] = q

    st.divider()
    c1,c2 = st.columns(2)
    c1.metric("Messages", len(st.session_state.messages))
    c2.metric("Analysed",  len(st.session_state.history))
    if st.session_state.history:
        fakes = sum(1 for h in st.session_state.history if h["result"]["prediction"]=="FAKE")
        c1,c2 = st.columns(2)
        c1.metric("🚨 Fake", fakes)
        c2.metric("✅ Real", len(st.session_state.history)-fakes)
    if st.button("🗑  Clear chat", use_container_width=True):
        st.session_state.messages=[]; st.session_state.last_result=None; st.session_state.history=[]; st.rerun()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_chat, tab_eval = st.tabs(["💬   Expert Chatbot","📊   Performance Evaluation"])

# ══════════════════════════════════════════════════════════════════════════════
# CHAT TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    col_chat, col_panel = st.columns([3,2], gap="large")

    with col_chat:
        st.markdown("""
<div class="ph">
  <div class="ph-icon">🧠</div>
  <div>
    <div class="ph-title">ABSA Expert Chatbot</div>
    <div class="ph-sub">Fake news detection · Aspect-based sentiment · AI-powered</div>
  </div>
  <div class="ph-pill"><div class="ph-dot"></div>Online</div>
</div>""", unsafe_allow_html=True)

        st.markdown("""<div class="strip">
  Paste a <b>headline</b>, full <b>article</b>, or a <b>URL</b> to analyse —
  or ask any expert question about the ABSA project.
</div>""", unsafe_allow_html=True)

        if not st.session_state.messages:
            with st.chat_message("assistant"):
                st.markdown("""Hi! I'm the **ABSA Fake News Expert** from Charles Darwin University (2026).

**What I can do:**
- 🔍 Analyse any **headline, article, or URL** — fake/real + 11-aspect sentiment
- 💡 Answer **expert questions** about ABSA, VADER, BERT, LIME, datasets, models
- 📊 **Evaluate performance** in the Evaluation tab

Try pasting a headline or pick one from the sidebar.""")

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        prefill = st.session_state.pop("_pre", None)
        user_input = st.chat_input("Paste headline, article, URL — or ask anything…")
        if prefill and not user_input: user_input = prefill

        if user_input:
            st.session_state.messages.append({"role":"user","content":user_input})
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
                                if result: result["headline"] = title or url[:80]
                            else:
                                result = {"error": f"Could not fetch URL — paste the text directly.\n\n{body}"}
                        else:
                            result = predict_article(headline=user_input)

                    if result and "error" not in result:
                        st.session_state.last_result = result
                        st.session_state.history.append({"text":user_input[:80],"result":result})
                        reply = chat_reply(result)
                    else:
                        reply = f"⚠️ {result.get('error','Could not analyse.') if result else 'Analysis failed.'}"
                else:
                    with st.spinner("Thinking…"):
                        reply = get_gemini_response(user_input, st.session_state.messages[:-1])

                st.markdown(reply)
                st.session_state.messages.append({"role":"assistant","content":reply})

    with col_panel:
        st.markdown('<div style="font-size:9px;font-weight:600;color:#0f1e2d;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:12px">Analysis Results</div>', unsafe_allow_html=True)

        if st.session_state.last_result and "error" not in st.session_state.last_result:
            render_panel(st.session_state.last_result)
            with st.expander("All 11 aspect scores"):
                for asp in aspects:
                    score = st.session_state.last_result["absa_scores"].get(asp,0.0)
                    pct = int((score+1)/2*100)
                    st.markdown(f"""
<div class="ar">
  <span class="an">{asp}</span>
  <div class="abg"><div class="af {bc_cls(score)}" style="width:{pct}%"></div></div>
  <span class="abadge {badge_cls(score)}">{badge_lbl(score)}</span>
  <span class="av">{score:+.3f}</span>
</div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="empty">
  <div class="empty-ico">📰</div>
  <div class="empty-t">Results appear here</div>
  <div class="empty-s">Paste a headline or article in the chat</div>
</div>""", unsafe_allow_html=True)

    if st.session_state.history:
        st.markdown('<div style="font-size:9px;font-weight:600;color:#0f1e2d;text-transform:uppercase;letter-spacing:0.12em;margin:16px 0 8px">Session history</div>', unsafe_allow_html=True)
        for h in reversed(st.session_state.history[-6:]):
            r = h["result"]
            cls = "bn" if r["prediction"]=="FAKE" else "bp"
            ico = "🚨" if r["prediction"]=="FAKE" else "✅"
            st.markdown(f"""
<div class="hrow">
  <span class="hbadge {cls}">{ico} {r['prediction']}</span>
  <span class="htxt">{h['text'][:55]}{"…" if len(h['text'])>55 else ""}</span>
  <span class="hconf">{r['confidence']}%</span>
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.markdown("""<div class="ph" style="margin-bottom:18px">
  <div class="ph-icon">📊</div>
  <div>
    <div class="ph-title">Performance Evaluation</div>
    <div class="ph-sub">Test the ABSA pipeline against labelled articles</div>
  </div>
</div>""", unsafe_allow_html=True)

    etab1, etab2 = st.tabs(["📋   Preset QA Dataset","✏️   Test Your Own Article"])

    with etab1:
        st.markdown("""<div class="strip">
  Evaluates against <b>10 pre-labelled headlines</b> covering all 11 aspect categories.
  Results compared against Model A (67.19%), Model B (66.86%), and Model C (49%).
</div>""", unsafe_allow_html=True)

        cb, cd = st.columns([1,2])
        with cb: run_p = st.button("▶  Run evaluation", use_container_width=True)
        with cd: st.markdown('<div style="font-size:12px;color:#0f1e2d;padding-top:8px">10 labelled headlines · exact-match accuracy · all 11 aspects</div>', unsafe_allow_html=True)

        if run_p:
            with st.spinner("Running…"):
                res=[]; bar=st.progress(0)
                for i,item in enumerate(QA_EVAL):
                    r=predict_article(headline=item["q"])
                    passed=r is not None and r.get("prediction")==item["expected"]
                    res.append({"question":item["q"],"expected":item["expected"],"predicted":r["prediction"] if r else "ERROR","confidence":r["confidence"] if r else 0,"category":item["category"],"passed":passed})
                    bar.progress((i+1)/len(QA_EVAL))
                bar.empty()
                st.session_state.eval_results=res; st.session_state.eval_done=True

        if st.session_state.eval_done and st.session_state.eval_results:
            res=st.session_state.eval_results; n=len(res)
            correct=sum(1 for r in res if r["passed"]); acc=round(correct/n*100,1)
            st.divider()
            m1,m2,m3,m4=st.columns(4)
            m1.metric("Accuracy",f"{acc}%"); m2.metric("Correct",f"{correct}/{n}")
            m3.metric("FAKE correct",f"{sum(1 for r in res if r['passed'] and r['expected']=='FAKE')}/{sum(1 for r in res if r['expected']=='FAKE')}")
            m4.metric("REAL correct",f"{sum(1 for r in res if r['passed'] and r['expected']=='REAL')}/{sum(1 for r in res if r['expected']=='REAL')}")

            st.markdown('<div style="font-size:9px;font-weight:600;color:#0f1e2d;text-transform:uppercase;letter-spacing:0.12em;margin:14px 0 8px">Comparison with notebook models</div>', unsafe_allow_html=True)
            fig=go.Figure(go.Bar(
                x=["Model A\n(TF-IDF)","Model B\n(TF-IDF+Sent)","Naive Bayes","Model C\n(RoBERTa)",f"This chatbot\n({acc}%)"],
                y=[67.19,66.86,64.0,49.0,acc],
                marker_color=["#0d2a50","#0d2a50","#0d2a50","#2d0a0a","#0a2d18" if acc>=60 else "#2d1a0a"],
                marker_line_width=0,
                text=[f"{v}%" for v in [67.19,66.86,64.0,49.0,acc]],
                textposition="outside", textfont=dict(size=10,color="#1e3a5a"),
            ))
            fig.update_layout(
                paper_bgcolor="#050d1a", plot_bgcolor="#07101c",
                font=dict(family="Plus Jakarta Sans",color="#1e3a5a"),
                xaxis=dict(tickfont=dict(size=10,color="#1e3a5a"),gridcolor="#0a1628"),
                yaxis=dict(range=[0,105],gridcolor="#0a1628",title="Accuracy (%)"),
                margin=dict(l=10,r=10,t=30,b=10),height=260,
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown('<div style="font-size:9px;font-weight:600;color:#0f1e2d;text-transform:uppercase;letter-spacing:0.12em;margin:6px 0 10px">Detailed results</div>', unsafe_allow_html=True)
            for r in res:
                icon="✅" if r["passed"] else "❌"
                cls="epass" if r["passed"] else "efail"
                eb="bp" if r["expected"]=="REAL" else "bn"
                pb="bp" if r["predicted"]=="REAL" else "bn"
                st.markdown(f"""
<div class="{cls}">
  <div class="eq">{icon} {r['question']}</div>
  <div style="display:flex;flex-wrap:wrap;gap:7px;align-items:center">
    <span style="font-size:10px;color:#0f1e2d">Expected:</span>
    <span class="abadge {eb}">{r['expected']}</span>
    <span style="font-size:10px;color:#0f1e2d">Got:</span>
    <span class="abadge {pb}">{r['predicted']}</span>
    <span style="font-size:10px;font-family:'JetBrains Mono',monospace;color:#0f1e2d">{r['confidence']}% conf</span>
    <span style="font-size:10px;background:#07101c;padding:1px 7px;border-radius:6px;color:#1e3a5a">{r['category']}</span>
  </div>
</div>""", unsafe_allow_html=True)

            st.markdown(f"""<div class="strip" style="margin-top:14px">
  <b>Chatbot accuracy: {acc}%</b> — {correct}/{n} correct.<br>
  {"Comparable to notebook Model A (67.19%) and Model B (66.86%)." if acc>=60 else "Reflects the known ~67% ceiling when using text features alone."}<br><br>
  <b>Key finding:</b> Adding ABSA sentiment features did <i>not</i> improve classification — Model B (66.86%) performed slightly worse than Model A (67.19%).
</div>""", unsafe_allow_html=True)

    with etab2:
        st.markdown("""<div class="strip">
  Test <b>your own article</b> — paste text or a URL, set the expected label, and see if the pipeline gets it right.
</div>""", unsafe_allow_html=True)
        custom = st.text_area("Article", placeholder="Paste headline, full article, or https://...", height=110, label_visibility="collapsed")
        cl,cr = st.columns([1,1])
        with cl: exp_lbl = st.selectbox("Expected label",["FAKE","REAL"],key="cust_exp")
        with cr: st.markdown("<div style='height:28px'></div>",unsafe_allow_html=True); run_c = st.button("▶  Analyse",use_container_width=True,key="run_c")
        if run_c:
            if not custom.strip():
                st.warning("Please paste some text or a URL.")
            else:
                with st.spinner("Analysing…"):
                    url = extract_url(custom)
                    if url:
                        title,body = fetch_url(url)
                        result = predict_article(headline=(title or ""),body=(body or "")) if (title or body) else {"error":body}
                        if result and "error" not in result: result["headline"]=title or url[:80]
                    else:
                        result = predict_article(headline=custom)
                if result and "error" not in result:
                    pred=result["prediction"]; passed=pred==exp_lbl
                    icon="✅ Correct!" if passed else "❌ Incorrect"
                    bg="#060f0a" if passed else "#0f0606"
                    bdr="#0f2d18" if passed else "#2d1010"
                    col="#4ab870" if passed else "#e05a5a"
                    st.markdown(f'<div style="background:{bg};border:1px solid {bdr};border-radius:10px;padding:11px 16px;font-size:13px;color:{col};margin-bottom:14px"><b>{icon}</b> — Predicted <b>{pred}</b> ({result["confidence"]}% confidence), expected <b>{exp_lbl}</b></div>', unsafe_allow_html=True)
                    render_panel(result)
                    st.session_state.last_result=result
                    st.session_state.history.append({"text":custom[:80],"result":result})
                else:
                    st.error(result.get("error","Analysis failed.") if result else "Analysis failed.")
