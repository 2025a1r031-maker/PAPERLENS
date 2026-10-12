"""
app.py
PaperLens — AI-Powered Academic Research Paper Analyzer
Main Application Entry Point & Professional Landing Dashboard.
"""

import os
import streamlit as st

# Streamlit Page Setup
st.set_page_config(
    page_title="PaperLens — Understand Research. Discover Connections.",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load CSS
css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 10px 0 20px 0;">
            <div style="font-size: 38px;">🔬</div>
            <h2 style="margin: 4px 0 0 0; color: #0f172a; font-weight: 800;">PaperLens</h2>
            <div style="font-size: 12px; color: #64748b; font-weight: 500;">v1.0.0 • Academic Edition</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🧭 Quick Navigation")
        st.page_link("app.py", label="Home Dashboard", icon="🏠")
        st.page_link("pages/1_🔬_Research_Analyzer.py", label="Research Analyzer", icon="🔬")
        st.page_link("pages/2_🔍_Paper_Discovery.py", label="Paper Discovery", icon="🔍")
        st.page_link("pages/3_ℹ️_About_PaperLens.py", label="About & Defense Notes", icon="ℹ️")

        st.markdown("---")
        st.markdown("### 🎓 Academic Evaluation")
        st.info(
            "Designed for university capstone demonstrations and academic evaluation. "
            "Runs 100% locally on CPU without paid API keys."
        )


def main():
    render_sidebar()

    # Hero Banner
    st.markdown("""
    <div class="brand-hero">
        <div class="brand-badge">⚡ AI-Powered Academic Analysis</div>
        <h1 class="brand-title">PaperLens</h1>
        <p class="brand-tagline">
            Understand Research. Discover Connections.
        </p>
        <div style="margin-top: 14px; font-size: 14.5px; color: #cbd5e1; max-width: 700px; line-height: 1.6;">
            A transparent, extractive semantic intelligence platform that turns dense academic PDFs
            into executive three-bullet summaries, extracts core scientific concepts, and maps knowledge
            relationships in an interactive graph.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Call to Action Buttons
    action_col1, action_col2, action_col3 = st.columns([2, 2, 3])
    with action_col1:
        st.page_link(
            "pages/1_🔬_Research_Analyzer.py",
            label="🚀 Launch Research Analyzer",
            icon="🔬",
            use_container_width=True,
        )
    with action_col2:
        st.page_link(
            "pages/2_🔍_Paper_Discovery.py",
            label="🔍 Search Academic Papers",
            icon="🔍",
            use_container_width=True,
        )
    with action_col3:
        st.page_link(
            "pages/3_ℹ️_About_PaperLens.py",
            label="📖 Architecture & Viva Notes",
            icon="ℹ️",
            use_container_width=True,
        )

    st.markdown("<br/>", unsafe_allow_html=True)

    # Core Features Showcase Cards
    st.markdown("### 🌟 Key Capabilities")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon-wrapper">🎯</div>
            <h4>Extractive 3-Bullet Summary</h4>
            <p>
                Ranks candidate sentences using Sentence-Transformers semantic centrality and
                archetypal objective probes to surface the Problem, Methodology, and Contributions
                with zero generative hallucination.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon-wrapper">🏷️</div>
            <h4>Key Concept Extraction</h4>
            <p>
                Identifies multi-word scientific phrases via KeyBERT-style semantic relevance
                and n-gram frequency. Inspect matching passages directly from the original manuscript.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon-wrapper">🕸️</div>
            <h4>Interactive Concept Graph</h4>
            <p>
                Generates a NetworkX knowledge graph visualized through PyVis physics.
                Features tunable similarity thresholds and degree sparsification to prevent visual hairballs.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # How It Works Section
    st.markdown("### 🔄 How PaperLens Works")

    step_col1, step_col2, step_col3, step_col4 = st.columns(4)

    with step_col1:
        st.markdown("""
        <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 18px; text-align: center;">
            <div style="font-size: 26px; margin-bottom: 8px;">📄</div>
            <strong style="color: #0f172a; font-size: 14.5px;">1. Ingest & Sanitize</strong>
            <p style="font-size: 12.5px; color: #64748b; margin-top: 6px; line-height: 1.5;">
                PyMuPDF extracts text, clears hyphenation, filters boilerplate, and detects scanned documents.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with step_col2:
        st.markdown("""
        <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 18px; text-align: center;">
            <div style="font-size: 26px; margin-bottom: 8px;">🧠</div>
            <strong style="color: #0f172a; font-size: 14.5px;">2. Dense Embeddings</strong>
            <p style="font-size: 12.5px; color: #64748b; margin-top: 6px; line-height: 1.5;">
                all-MiniLM-L6-v2 maps sentences to a 384-D vector space to calculate semantic centroid distance.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with step_col3:
        st.markdown("""
        <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 18px; text-align: center;">
            <div style="font-size: 26px; margin-bottom: 8px;">🕸️</div>
            <strong style="color: #0f172a; font-size: 14.5px;">3. Network Modeling</strong>
            <p style="font-size: 12.5px; color: #64748b; margin-top: 6px; line-height: 1.5;">
                NetworkX computes cosine graph topology with degree sparsification and hub centrality.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with step_col4:
        st.markdown("""
        <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 18px; text-align: center;">
            <div style="font-size: 26px; margin-bottom: 8px;">📊</div>
            <strong style="color: #0f172a; font-size: 14.5px;">4. Explore & Export</strong>
            <p style="font-size: 12.5px; color: #64748b; margin-top: 6px; line-height: 1.5;">
                Drag and inspect PyVis physics on screen; export data to TXT, MD, CSV, JSON, and standalone HTML.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # Direct Document Upload Area on Home Page
    st.markdown("### 📤 Quick Document Upload")
    st.write("Ready to analyze an academic paper? Upload it here to begin, or switch to the Research Analyzer:")

    quick_file = st.file_uploader(
        "Upload PDF Document:",
        type=["pdf"],
        key="home_quick_upload",
        help="Upload any research paper in PDF format (up to 50MB).",
    )

    if quick_file is not None:
        st.success(f"File '{quick_file.name}' received! Click below to process:")
        if st.button("🚀 Process Paper in Research Analyzer", type="primary"):
            st.session_state.home_uploaded_bytes = quick_file.read()
            st.session_state.home_uploaded_name = quick_file.name
            st.switch_page("pages/1_🔬_Research_Analyzer.py")

    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #94a3b8; font-size: 13px; padding: 12px 0;">
        PaperLens Academic Research Analyzer • Built with Python, Streamlit, PyMuPDF, Sentence-Transformers, and NetworkX.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
