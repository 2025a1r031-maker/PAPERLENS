"""
pages/2_🔍_Paper_Discovery.py
Optional Academic Paper Discovery powered by the CORE API (core.ac.uk).
Allows users to discover open-access research papers, view abstracts and citations,
and directly send papers into PaperLens for analysis.
"""

import os
import streamlit as st
import requests

from modules.core_discovery import (
    search_core_papers,
    get_core_api_key,
    CoreSearchResult,
)

st.set_page_config(
    page_title="PaperLens — Academic Paper Discovery",
    page_icon="🔍",
    layout="wide",
)

# Load CSS
css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("""
<div class="brand-hero">
    <div class="brand-badge">🔍 Open Access Repository</div>
    <h1 class="brand-title">Academic Paper Discovery</h1>
    <p class="brand-tagline">
        Discover open-access research articles indexed by CORE (core.ac.uk), inspect author abstracts,
        and explore scientific literature worldwide.
    </p>
</div>
""", unsafe_allow_html=True)

# API Key Handling & Guidance
env_key = get_core_api_key()

with st.sidebar:
    st.markdown("### 🔑 CORE API Configuration")
    user_api_key = st.text_input(
        "CORE API Key (Optional):",
        value=env_key or "",
        type="password",
        help="Free API key from https://core.ac.uk/services/api. If left blank, curated benchmark papers are displayed.",
    )

    if not user_api_key:
        st.info(
            "💡 **Offline / Free Mode Active:** You can search curated landmark research papers right away without an API key."
        )
    else:
        st.success("✅ Custom CORE API key detected.")

    st.markdown("---")
    st.markdown("### 📚 About CORE")
    st.write(
        "CORE aggregates open-access research from thousands of institutional repositories and journals globally."
    )

# Search Bar
search_col1, search_col2 = st.columns([4, 1])

with search_col1:
    search_query = st.text_input(
        "Search academic literature:",
        placeholder="e.g. Transformer Attention, Quantum Computing, Deep Residual Learning...",
    )

with search_col2:
    st.write("")
    st.write("")
    search_button = st.button("🔎 Search", type="primary", use_container_width=True)

# Search Execution
if search_query or search_button:
    with st.spinner("Querying open-access academic index..."):
        result: CoreSearchResult = search_core_papers(
            query=search_query or "Deep Learning",
            limit=10,
            api_key=user_api_key or env_key,
        )

    if result.is_mock_fallback:
        st.warning(result.error_message)

    if not result.is_success:
        st.error(result.error_message or "An error occurred while communicating with the CORE API.")
    else:
        st.markdown(f"### Found {result.total_hits} Research Papers for *\"{result.query}\"*")

        for paper in result.papers:
            with st.container():
                st.markdown(f"""
                <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                        <div>
                            <h3 style="margin: 0 0 6px 0; font-size: 18px; color: #0f172a;">{paper.title}</h3>
                            <div style="font-size: 13px; color: #64748b; margin-bottom: 10px;">
                                <span>✍️ {', '.join(paper.authors[:4])}{'...' if len(paper.authors) > 4 else ''}</span>
                                <span style="margin: 0 8px;">•</span>
                                <span>📅 {paper.year or 'Year N/A'}</span>
                                {f'<span style="margin: 0 8px;">•</span><span>🏛️ {paper.publisher}</span>' if paper.publisher else ''}
                            </div>
                        </div>
                    </div>
                    <div style="font-size: 13.5px; color: #334155; line-height: 1.6; margin-bottom: 12px; max-height: 120px; overflow-y: auto;">
                        {paper.abstract}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                act_col1, act_col2, act_col3 = st.columns([2, 2, 4])

                with act_col1:
                    if paper.download_url:
                        st.link_button(
                            "📥 Download Open-Access PDF",
                            paper.download_url,
                            use_container_width=True,
                        )
                    else:
                        st.button("🔒 Restricted Access", disabled=True, use_container_width=True)

                with act_col2:
                    if paper.source_url:
                        st.link_button(
                            "🔗 View Source Article",
                            paper.source_url,
                            use_container_width=True,
                        )

                with act_col3:
                    if paper.doi:
                        st.caption(f"DOI: `{paper.doi}`")

                st.markdown("<br/>", unsafe_allow_html=True)
else:
    st.info("💡 Enter a research topic above or click Search to explore sample academic papers.")
