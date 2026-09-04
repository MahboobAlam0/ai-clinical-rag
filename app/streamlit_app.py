"""
app/streamlit_app.py
--------------------
Clinical RAG — Streamlit UI

Run:
  python -m streamlit run app/streamlit_app.py
"""

import os
import sys
import time
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv()

from src.rag_pipeline import ClinicalRAGPipeline

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Clinical RAG · PubMed Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Header banner ── */
.hero {
    background: linear-gradient(135deg, #0f2044 0%, #1a3a6b 50%, #0d5ea6 100%);
    border-radius: 16px;
    padding: 2.5rem 2rem 2rem 2rem;
    margin-bottom: 1.5rem;
    color: white;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "";
    position: absolute;
    top: -40px; right: -40px;
    width: 200px; height: 200px;
    background: rgba(255,255,255,0.04);
    border-radius: 50%;
}
.hero-title {
    font-size: 2.2rem;
    font-weight: 700;
    letter-spacing: -0.5px;
    margin: 0 0 0.4rem 0;
}
.hero-sub {
    font-size: 1rem;
    font-weight: 300;
    opacity: 0.82;
    margin: 0 0 1.4rem 0;
}
.hero-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
}
.pill {
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
    font-weight: 500;
    color: rgba(255,255,255,0.9);
}

/* ── Metric cards ── */
.metric-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.metric-card {
    flex: 1;
    background: white;
    border: 1px solid #e8ecf0;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.metric-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    color: #6b7280;
    margin-bottom: 0.3rem;
}
.metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #0f2044;
    line-height: 1;
}
.metric-sub {
    font-size: 0.75rem;
    color: #9ca3af;
    margin-top: 0.2rem;
}

/* ── Example question buttons ── */
.stButton > button {
    border-radius: 10px !important;
    border: 1.5px solid #dde3ec !important;
    background: white !important;
    color: #1a3a6b !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    padding: 0.55rem 1rem !important;
    transition: all 0.18s ease !important;
    text-align: left !important;
}
.stButton > button:hover {
    border-color: #0d5ea6 !important;
    background: #f0f6ff !important;
    color: #0d5ea6 !important;
    box-shadow: 0 2px 8px rgba(13,94,166,0.12) !important;
}

/* ── Primary search button ── */
div[data-testid="stButton"] button[kind="primary"] {
    background: linear-gradient(135deg, #0d5ea6, #1a3a6b) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.65rem 1rem !important;
    box-shadow: 0 4px 14px rgba(13,94,166,0.3) !important;
    transition: all 0.2s ease !important;
}
div[data-testid="stButton"] button[kind="primary"]:hover {
    box-shadow: 0 6px 20px rgba(13,94,166,0.4) !important;
    transform: translateY(-1px) !important;
}

/* ── Confidence badge ── */
.conf-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 14px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.88rem;
    letter-spacing: 0.3px;
}
.conf-HIGH   { background: #dcfce7; color: #15803d; border: 1.5px solid #86efac; }
.conf-MEDIUM { background: #fef9c3; color: #a16207; border: 1.5px solid #fde047; }
.conf-LOW    { background: #fee2e2; color: #b91c1c; border: 1.5px solid #fca5a5; }

/* ── Score bar ── */
.score-bar-wrap { display: flex; align-items: center; gap: 10px; margin: 4px 0; }
.score-bar-bg {
    flex: 1; height: 7px; background: #e5e7eb;
    border-radius: 6px; overflow: hidden;
}
.score-bar-fill { height: 100%; border-radius: 6px; }

/* ── Answer box ── */
.answer-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #0d5ea6;
    border-radius: 10px;
    padding: 1.4rem 1.6rem;
    font-size: 0.97rem;
    line-height: 1.75;
    color: #1e293b;
    margin: 1rem 0;
}

/* ── Section heading ── */
.section-heading {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #6b7280;
    margin: 1.6rem 0 0.8rem 0;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* ── Source card ── */
.source-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.7rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    transition: box-shadow 0.18s;
}
.source-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.09); }
.source-meta {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 0.6rem;
    flex-wrap: wrap;
}
.tag {
    font-size: 0.72rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 6px;
    background: #eff6ff;
    color: #1d4ed8;
    border: 1px solid #bfdbfe;
}
.tag-label {
    background: #f0fdf4;
    color: #166534;
    border-color: #bbf7d0;
}
.source-excerpt {
    font-size: 0.86rem;
    color: #4b5563;
    line-height: 1.6;
    border-top: 1px solid #f3f4f6;
    padding-top: 0.6rem;
    margin-top: 0.4rem;
}

/* ── History item ── */
.history-item {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.6rem;
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 1rem;
}
.history-q { font-size: 0.88rem; color: #1e293b; font-weight: 500; }
.history-meta { font-size: 0.75rem; color: #9ca3af; white-space: nowrap; }

/* ── Divider ── */
.custom-divider {
    border: none;
    border-top: 1px solid #e5e7eb;
    margin: 1.4rem 0;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] { background: #f8fafc; }
section[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }

/* ── Text area ── */
.stTextArea textarea {
    border-radius: 10px !important;
    border: 1.5px solid #dde3ec !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.94rem !important;
}
.stTextArea textarea:focus {
    border-color: #0d5ea6 !important;
    box-shadow: 0 0 0 3px rgba(13,94,166,0.1) !important;
}

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; max-width: 1100px; }
</style>
""", unsafe_allow_html=True)


# ── Pipeline ──────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_pipeline() -> ClinicalRAGPipeline:
    from src.ingest import build_index_if_needed, ensure_data
    from src.config import get_settings

    with st.status("🚀 Initialising knowledge base…", expanded=True) as status:
        st.write("📥 Checking PubMed dataset…")
        ensure_data()

        st.write("🔍 Building vector index (first run ~30 s)…")
        build_index_if_needed()

        st.write("🤖 Loading embedding model…")
        pipeline = ClinicalRAGPipeline()
        pipeline.retriever._load()

        status.update(label="✅ Knowledge base ready — 5,355 PubMed abstracts indexed",
                      state="complete", expanded=False)
    return pipeline

pipeline = _load_pipeline()

# ── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state["history"] = []
if "total_queries" not in st.session_state:
    st.session_state["total_queries"] = 0
if "cache_hits" not in st.session_state:
    st.session_state["cache_hits"] = 0
if "total_latency" not in st.session_state:
    st.session_state["total_latency"] = 0.0


# ── Helpers ───────────────────────────────────────────────────────────────────
CONF_ICON  = {"HIGH": "●", "MEDIUM": "●", "LOW": "●"}
SCORE_COLOR = {"HIGH": "#22c55e", "MEDIUM": "#eab308", "LOW": "#ef4444"}
CONF_LABEL  = {
    "HIGH":   "Strong Evidence",
    "MEDIUM": "Partial Evidence",
    "LOW":    "Weak Signal",
}

def _score_bar(score: float, color: str) -> str:
    pct = round(score * 100)
    return f"""
    <div class="score-bar-wrap">
        <div class="score-bar-bg">
            <div class="score-bar-fill" style="width:{pct}%;background:{color};"></div>
        </div>
        <span style="font-size:0.82rem;font-weight:600;color:#374151;min-width:42px">{score:.3f}</span>
    </div>"""

def _render_sources(sources: list) -> None:
    st.markdown('<div class="section-heading">📚 Retrieved Sources</div>', unsafe_allow_html=True)
    for src in sources:
        score   = src["score"]
        color   = "#22c55e" if score >= 0.70 else "#eab308" if score >= 0.50 else "#ef4444"
        st.markdown(f"""
        <div class="source-card">
            <div class="source-meta">
                <span class="tag">Source {src['index']}</span>
                <span class="tag">PubMed {src['pubid']}</span>
                <span class="tag tag-label">{src['label']}</span>
                {_score_bar(score, color)}
            </div>
            <div class="source-excerpt">{src['excerpt']}</div>
        </div>""", unsafe_allow_html=True)

def _render_result(entry: dict, streaming: bool = False) -> None:
    result   = entry["result"]
    conf     = result.confidence
    color    = SCORE_COLOR[conf]

    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin:1rem 0 0.5rem 0">
        <span class="conf-badge conf-{conf}">
            <span style="color:{color}">●</span> {conf} — {CONF_LABEL[conf]}
        </span>
        <span style="font-size:0.82rem;color:#6b7280">
            max cosine <b>{result.max_score}</b> · mean <b>{result.mean_score}</b>
            · {entry['latency_ms']} ms
        </span>
    </div>
    <p style="font-size:0.83rem;color:#6b7280;margin:0 0 1rem 0;font-style:italic">
        {result.confidence_note}
    </p>""", unsafe_allow_html=True)

    st.markdown('<div class="section-heading">📋 Answer</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="answer-box">{result.answer}</div>', unsafe_allow_html=True)

    _render_sources(result.sources)

    st.markdown("""
    <p style="font-size:0.75rem;color:#9ca3af;margin-top:1.2rem;padding:0.6rem 0.8rem;
              background:#f8fafc;border-radius:8px;border:1px solid #e5e7eb">
    ⚠️ Research use only. This system may contain errors and is not a substitute for
    professional clinical judgment or medical advice.
    </p>""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧬 Clinical RAG")
    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    top_k = st.slider("Sources to retrieve (top-k)", 1, 10, 5)

    st.markdown("**System**")
    st.markdown("""
    <div style="font-size:0.82rem;color:#4b5563;line-height:1.9">
    🤖 Llama-3.3-70B via Groq<br>
    🔍 Fine-tuned MiniLM-L6-v2<br>
    🗄️ Qdrant vector store<br>
    📄 5,355 PubMed chunks
    </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
    st.markdown("**Session Stats**")

    total = st.session_state["total_queries"]
    avg_lat = (
        round(st.session_state["total_latency"] / total)
        if total > 0 else 0
    )
    conf_counts = {}
    for h in st.session_state["history"]:
        c = h["result"].confidence
        conf_counts[c] = conf_counts.get(c, 0) + 1

    st.markdown(f"""
    <div style="font-size:0.82rem;color:#4b5563;line-height:2">
    🔢 Queries: <b>{total}</b><br>
    ⏱️ Avg latency: <b>{avg_lat} ms</b><br>
    🟢 HIGH: <b>{conf_counts.get('HIGH', 0)}</b>
    &nbsp;🟡 MED: <b>{conf_counts.get('MEDIUM', 0)}</b>
    &nbsp;🔴 LOW: <b>{conf_counts.get('LOW', 0)}</b>
    </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    if st.button("🗑️ Clear history", use_container_width=True):
        st.session_state["history"] = []
        st.session_state["total_queries"] = 0
        st.session_state["total_latency"] = 0.0
        st.rerun()


# ── Hero header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">🧬 Clinical RAG</div>
    <div class="hero-sub">
        Evidence-based answers from 5,355 PubMed abstracts · Uncertainty-aware · Source-cited
    </div>
    <div class="hero-pills">
        <span class="pill">Fine-tuned Embeddings · nDCG@10 +12%</span>
        <span class="pill">Qdrant Vector Store</span>
        <span class="pill">Llama-3.3-70B via Groq</span>
        <span class="pill">Streaming Responses</span>
        <span class="pill">Uncertainty Quantification</span>
    </div>
</div>""", unsafe_allow_html=True)


# ── Example questions ─────────────────────────────────────────────────────────
EXAMPLES = [
    "What are the effects of metformin on HbA1c in type 2 diabetes?",
    "Is there evidence for statins in primary prevention of cardiovascular disease?",
    "What does the literature say about beta-blockers in heart failure?",
    "How effective is CBT for major depressive disorder?",
]

st.markdown('<div class="section-heading">💡 Example Questions</div>', unsafe_allow_html=True)
cols = st.columns(2)
for i, ex in enumerate(EXAMPLES):
    if cols[i % 2].button(ex, use_container_width=True, key=f"ex_{i}"):
        st.session_state["question_input"] = ex
        st.rerun()

# ── Input ─────────────────────────────────────────────────────────────────────
question = st.text_area(
    "Ask a clinical question",
    height=90,
    placeholder="e.g. What is the evidence for aspirin in secondary prevention of stroke?",
    key="question_input",
    label_visibility="collapsed",
)

col_btn, col_hint = st.columns([2, 5])
search = col_btn.button("🔍 Search Literature", type="primary", use_container_width=True)
col_hint.markdown(
    '<p style="font-size:0.8rem;color:#9ca3af;padding-top:0.6rem">'
    f'top-k = {top_k} · PubMedQA corpus · answers grounded in retrieved context only'
    '</p>',
    unsafe_allow_html=True,
)


# ── Search & stream ───────────────────────────────────────────────────────────
if search:
    q = question.strip()
    if not q:
        st.warning("Please enter a question.")
    elif not os.environ.get("GROQ_API_KEY"):
        st.error("GROQ_API_KEY not set — add it to your .env file.")
    else:
        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
        st.markdown(
            f'<p style="font-size:0.9rem;color:#374151;font-weight:500;margin-bottom:1rem">'
            f'🔎 {q}</p>',
            unsafe_allow_html=True,
        )

        t0 = time.perf_counter()
        meta_slot    = st.empty()
        answer_slot  = st.empty()
        sources_slot = st.empty()
        error_slot   = st.empty()

        answer_text = ""
        meta        = None

        try:
            for event in pipeline.answer_stream(q, top_k=top_k):

                if event["type"] == "meta":
                    meta  = event
                    conf  = meta["confidence"]
                    color = SCORE_COLOR[conf]
                    meta_slot.markdown(f"""
                    <div style="display:flex;align-items:center;gap:12px;
                                flex-wrap:wrap;margin-bottom:0.6rem">
                        <span class="conf-badge conf-{conf}">
                            <span style="color:{color}">●</span>
                            {conf} — {CONF_LABEL[conf]}
                        </span>
                        <span style="font-size:0.82rem;color:#6b7280">
                            max cosine <b>{meta['max_score']}</b>
                            · mean <b>{meta['mean_score']}</b>
                        </span>
                    </div>
                    <p style="font-size:0.83rem;color:#6b7280;
                              font-style:italic;margin:0 0 0.8rem 0">
                        {meta['confidence_note']}
                    </p>""", unsafe_allow_html=True)

                elif event["type"] == "token":
                    answer_text += event["content"]
                    answer_slot.markdown(
                        f'<div class="answer-box">{answer_text}▌</div>',
                        unsafe_allow_html=True,
                    )

                elif event["type"] == "done":
                    answer_slot.markdown(
                        f'<div class="answer-box">{answer_text}</div>',
                        unsafe_allow_html=True,
                    )

                elif event["type"] == "error":
                    error_slot.error(event["detail"])

        except Exception as exc:
            st.error(f"Error: {exc}")
            st.stop()

        latency_ms = round((time.perf_counter() - t0) * 1000)

        # Render sources below the streamed answer
        if meta:
            with sources_slot.container():
                _render_sources(meta["sources"])
            st.markdown(
                f'<p style="font-size:0.75rem;color:#9ca3af;margin-top:0.6rem">'
                f'⏱️ {latency_ms} ms · top-k={top_k} · '
                f'{datetime.now().strftime("%H:%M:%S")}</p>',
                unsafe_allow_html=True,
            )

        # Build a synthetic RAGResponse for history storage
        from src.rag_pipeline import RAGResponse
        fake_result = RAGResponse(
            query=q,
            answer=answer_text,
            confidence=meta["confidence"] if meta else "LOW",
            confidence_note=meta["confidence_note"] if meta else "",
            max_score=meta["max_score"] if meta else 0.0,
            mean_score=meta["mean_score"] if meta else 0.0,
            sources=meta["sources"] if meta else [],
        )
        st.session_state["history"].insert(0, {
            "question": q,
            "result":   fake_result,
            "latency_ms": latency_ms,
            "timestamp":  datetime.now().strftime("%H:%M:%S"),
            "top_k":      top_k,
        })
        st.session_state["total_queries"] += 1
        st.session_state["total_latency"] += latency_ms


# ── History ───────────────────────────────────────────────────────────────────
history = st.session_state["history"]

# Skip index 0 — that's the result we just rendered above
past = history[1:] if search and history else history

if past:
    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
    st.markdown(
        f'<div class="section-heading">🕐 Query History ({len(past)})</div>',
        unsafe_allow_html=True,
    )
    for entry in past:
        conf  = entry["result"].confidence
        color = SCORE_COLOR[conf]
        with st.expander(
            f"{entry['timestamp']}  ·  {entry['question'][:80]}{'…' if len(entry['question'])>80 else ''}",
            expanded=False,
        ):
            _render_result(entry)
