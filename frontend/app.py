import streamlit as st
import sys
import os
import html
from pathlib import Path

# Add project root to sys.path so we can import modules
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.settings import Settings
from backend.orchestrator import Orchestrator
from backend.rag_engine import RAGEngine
from utils.logging import setup_logger

# Initialize logger
logger = setup_logger("frontend")

# Initialize session state for input fields
if "headline" not in st.session_state:
    st.session_state.headline = ""
if "article" not in st.session_state:
    st.session_state.article = ""
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None

# Set up Streamlit Page Configuration
st.set_page_config(
    page_title="VeriTruth: Agentic RAG Fake News Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .header-container {
        background: linear-gradient(135deg, #1E1B4B 0%, #0F172A 100%);
        padding: 2.5rem 1.5rem;
        border-radius: 16px;
        border: 1px solid #312E81;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4);
    }
    
    .header-title {
        color: #F8FAFC;
        font-family: 'Outfit', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .header-subtitle {
        color: #94A3B8;
        font-size: 1.1rem;
    }

    .output-card {
        background: #1E293B;
        border-radius: 14px;
        padding: 1.8rem;
        border: 1px solid #334155;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
    }

    .output-card-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.25rem;
        font-weight: 600;
        color: #F8FAFC;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .verdict-container {
        padding: 1.2rem;
        border-radius: 10px;
        font-weight: 700;
        text-align: center;
        font-size: 1.6rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        border-width: 2px;
        border-style: solid;
    }

    .verdict-true {
        background-color: rgba(16, 185, 129, 0.1);
        color: #10B981;
        border-color: #10B981;
    }

    .verdict-false {
        background-color: rgba(239, 68, 68, 0.1);
        color: #EF4444;
        border-color: #EF4444;
    }

    .verdict-misleading {
        background-color: rgba(245, 158, 11, 0.1);
        color: #F59E0B;
        border-color: #F59E0B;
    }

    .verdict-insufficient {
        background-color: rgba(148, 163, 184, 0.1);
        color: #94A3B8;
        border-color: #94A3B8;
    }

    .source-tag {
        display: inline-block;
        background: #2D3748;
        color: #63B3ED !important;
        text-decoration: none;
        padding: 0.35rem 0.7rem;
        border-radius: 6px;
        font-size: 0.8rem;
        margin-right: 0.5rem;
        margin-top: 0.5rem;
        border: 1px solid #4A5568;
        transition: background 0.2s ease;
    }
    
    .source-tag:hover {
        background: #4A5568;
    }
</style>
""", unsafe_allow_html=True)

# Main Title Header
st.markdown("""
<div class="header-container">
    <div class="header-title">🛡️ Agentic RAG Fake News Detection System</div>
    <div class="header-subtitle">Compare claim with evidence using Scanner, Researcher, and Fact Checker Agents</div>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.title("🛠️ Settings")

# Google Gemini API Key Input
api_key_placeholder = Settings.GEMINI_API_KEY
user_api_key = st.sidebar.text_input(
    "Google Gemini API Key",
    value=api_key_placeholder,
    type="password"
)

if user_api_key != Settings.GEMINI_API_KEY:
    Settings.set_api_key(user_api_key)
    logger.info("API Key dynamically updated in sidebar.")

if not Settings.validate():
    st.sidebar.warning("⚠️ API Key is missing. Please configure it to enable verification.")

# Model Select
model_selection = st.sidebar.selectbox("Gemini Model", ["gemini-2.5-flash", "gemini-2.5-pro"], index=0)
if model_selection != Settings.GEMINI_MODEL:
    Settings.GEMINI_MODEL = model_selection
    os.environ["GEMINI_MODEL"] = model_selection

st.sidebar.divider()
st.sidebar.subheader("📚 RAG Knowledge Base")
rag_engine = RAGEngine()
st.sidebar.metric("Ingested Facts Count", rag_engine.get_document_count())

# Ingestions
with st.sidebar.expander("Ingest Reference Facts"):
    kb_title = st.text_input("Source Title")
    kb_content = st.text_area("Factual Context")
    if st.button("Add to Knowledge Base", use_container_width=True):
        if not kb_title.strip() or not kb_content.strip():
            st.error("Fields cannot be empty.")
        else:
            chunks = [c.strip() for c in kb_content.split("\n\n") if len(c.strip()) > 10]
            metadatas = [{"title": kb_title, "source": "UI Setup"} for _ in chunks]
            if rag_engine.add_reference_documents(chunks, metadatas=metadatas):
                st.success("Facts Ingested Successfully!")
                st.rerun()

if st.sidebar.button("Clear Reference Facts", type="primary", use_container_width=True):
    if rag_engine.clear_kb():
        st.sidebar.success("Database cleared!")
        st.rerun()

# =====================================================================
# INPUT INTERFACE
# =====================================================================
st.subheader("🔍 News Verification Input")

# Binds input fields to session state
headline_input = st.text_input(
    "News Headline",
    value=st.session_state.headline,
    placeholder="Enter the news headline or core assertion statement..."
)

article_input = st.text_area(
    "Full News Article (optional)",
    value=st.session_state.article,
    height=180,
    placeholder="Paste the full body text here to supply contextual metadata..."
)

# Keep session state in sync
st.session_state.headline = headline_input
st.session_state.article = article_input

col_verify, col_clear, _ = st.columns([1.2, 1, 4.8])

with col_verify:
    verify_clicked = st.button("🛡️ Verify News", type="primary", use_container_width=True)

with col_clear:
    clear_clicked = st.button("🧹 Clear", use_container_width=True)

# Clear button action
if clear_clicked:
    st.session_state.headline = ""
    st.session_state.article = ""
    st.session_state.analysis_results = None
    st.rerun()

# =====================================================================
# VERIFICATION PROCESSING
# =====================================================================
if verify_clicked:
    if not Settings.validate():
        st.error("Verification stopped: Google Gemini API Key is missing. Input your key in the sidebar.")
    elif not headline_input.strip():
        st.warning("Headline input cannot be empty. Please specify the claim headline.")
    else:
        # Build composite verification context
        compound_text = f"Headline: {headline_input}"
        if article_input.strip():
            compound_text += f"\n\nFull Article:\n{article_input}"
            
        orchestrator = Orchestrator()
        
        # Loading spinner around the agent generator iterations
        with st.spinner("Connecting to multi-agent fact-checking network..."):
            pipeline_data = None
            status_indicators = st.empty()
            
            try:
                for update in orchestrator.run_pipeline(compound_text):
                    status = update["status"]
                    message = update["message"]
                    data = update["data"]
                    
                    status_indicators.info(f"⏳ **Active Step:** {message}")
                    
                    if status == "error":
                        st.error(f"Execution Error: {message}")
                        break
                    elif status == "completed":
                        pipeline_data = data
                        status_indicators.empty()
                        st.success("Fact-checking analysis completed successfully!")
                        
                if pipeline_data:
                    st.session_state.analysis_results = pipeline_data
                    
            except Exception as e:
                logger.exception("Orchestration pipeline collapsed.")
                st.error(f"Fatal execution crash: {e}")

# =====================================================================
# DISPLAY OUTPUT CARDS
# =====================================================================
if st.session_state.analysis_results:
    results = st.session_state.analysis_results
    scan_res = results.get("scan_results", {})
    fact_check_res = results.get("fact_check_results", {})
    evidence_res = results.get("evidence_by_claim", {})
    
    st.divider()
    st.subheader("📋 Verification Report Output")
    
    # Grid columns layout
    col_verdict, col_confidence = st.columns(2)
    
    # ----------------------------------------------------
    # CARD 1: VERIFICATION
    # ----------------------------------------------------
    with col_verdict:
        overall_verdict = fact_check_res.get("overall_verdict", "Insufficient Evidence")
        
        # Format styles matching classification output
        badge_style = "verdict-insufficient"
        if overall_verdict == "True":
            badge_style = "verdict-true"
        elif overall_verdict == "False":
            badge_style = "verdict-false"
        elif overall_verdict == "Misleading":
            badge_style = "verdict-misleading"
            
        st.markdown(f"""
        <div class="output-card">
            <div class="output-card-title">⚖️ Verification Status</div>
            <div class="verdict-container {badge_style}">
                {overall_verdict}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ----------------------------------------------------
    # CARD 2: CONFIDENCE
    # ----------------------------------------------------
    with col_confidence:
        verdicts_list = fact_check_res.get("claim_verdicts", [])
        if verdicts_list:
            avg_confidence = sum(v.get("confidence_score", 0.5) for v in verdicts_list) / len(verdicts_list)
        else:
            avg_confidence = 0.5
            
        st.markdown("""
        <div class="output-card">
            <div class="output-card-title">📊 Assessment Confidence</div>
            <div style="height: 10px;"></div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"**Confidence Level:** `{avg_confidence * 100:.0f}%` certainty in final classification.")
        st.progress(avg_confidence)

    # ----------------------------------------------------
    # CARD 3: EXPLANATION
    # ----------------------------------------------------
    st.markdown(f"""
    <div class="output-card">
        <div class="output-card-title">📖 Explanation Reasoning</div>
        <p style="color: #E2E8F0; font-size: 1.05rem; line-height: 1.6; margin: 0;">
            {fact_check_res.get('overall_reasoning', 'No step-by-step reasoning report generated.')}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ----------------------------------------------------
    # CARD 4: EVIDENCE
    # ----------------------------------------------------
    st.markdown("""
    <div class="output-card">
        <div class="output-card-title">📚 Supporting Evidence & Sources</div>
    </div>
    """, unsafe_allow_html=True)
    
    claims = scan_res.get("claims", [])
    claim_verdicts = fact_check_res.get("claim_verdicts", [])
    verdict_map = {v.get("claim_id"): v for v in claim_verdicts}
    
    for idx, c in enumerate(claims):
        cid = c.get("id")
        assertion = c.get("assertion")
        c_verdict_info = verdict_map.get(cid, {})
        c_evidence = evidence_res.get(cid, {})
        
        c_verdict = c_verdict_info.get("verdict", "Insufficient Evidence")
        c_badge_style = "verdict-insufficient"
        if c_verdict == "True":
            c_badge_style = "verdict-true"
        elif c_verdict == "False":
            c_badge_style = "verdict-false"
        elif c_verdict == "Misleading":
            c_badge_style = "verdict-misleading"
            
        with st.expander(f"Claim #{idx+1} [ {c_verdict} ] — {assertion[:90]}...", expanded=True):
            st.markdown(f"**Extracted Factual Assertion:** *\"{assertion}\"*")
            st.markdown(f"**Verdict:** <span class='verdict-badge {c_badge_style}'>{c_verdict}</span>", unsafe_allow_html=True)
            st.markdown(f"**Analysis:** {c_verdict_info.get('reasoning')}")
            
            st.write("")
            st.markdown("**Synthesized Research context:**")
            st.write(c_evidence.get("synthesized_evidence", "No evidence synthesized."))
            
            sources = c_evidence.get("sources", [])
            if sources:
                st.write("**Supporting Sources:**")
                links_html = ""
                for src in sources:
                    title = src.get("title", src.get("source", "Reference URL"))
                    url = src.get("url", "")
                    escaped_title = html.escape(title)
                    escaped_url = html.escape(url)
                    if url:
                        links_html += f'<a href="{escaped_url}" target="_blank" class="source-tag">🔗 {escaped_title}</a>'
                    else:
                        links_html += f'<span class="source-tag">📄 {escaped_title}</span>'
                st.markdown(links_html, unsafe_allow_html=True)
                st.write("")
