"""
pages/3_ℹ️_About_PaperLens.py
Comprehensive project documentation, architecture, college project presentation notes,
algorithmic defense explanations, and technology stack breakdown for PaperLens.
"""

import os
import streamlit as st

st.set_page_config(
    page_title="PaperLens — Project Documentation & Defense Notes",
    page_icon="ℹ️",
    layout="wide",
)

# Load CSS
css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("""
<div class="brand-hero">
    <div class="brand-badge">🎓 Academic Project Guide</div>
    <h1 class="brand-title">About PaperLens</h1>
    <p class="brand-tagline">
        Architecture breakdown, algorithmic foundations, and academic defense guide
        for students presenting PaperLens to faculty evaluators.
    </p>
</div>
""", unsafe_allow_html=True)

doc_tab1, doc_tab2, doc_tab3, doc_tab4 = st.tabs([
    "📐 System Architecture",
    "🧠 Algorithmic Defense Guide",
    "🛠️ Tech Stack & Dependencies",
    "❓ Frequently Asked Questions",
])

with doc_tab1:
    st.markdown("### 📐 End-to-End System Architecture")
    st.markdown("""
    PaperLens operates on a modular 5-stage NLP pipeline:

    ```
    [Academic PDF Document]
               │
               ▼
    ┌───────────────────────────────────┐
    │ 1. Document Ingestion & Sanitizer │ (PyMuPDF / fitz)
    │    - Byte & header validation     │ - Hyphen de-breaking
    │    - Scanned/raster detection     │ - Page-level coordinate tracking
    └─────────────────┬─────────────────┘
                      │
                      ▼
    ┌───────────────────────────────────┐
    │ 2. Sentence & Section Engine      │ (Abbreviation-Aware Regex)
    │    - Protected abbreviations      │ - Section heuristic tagging
    │    - Informative noise filtering  │   (Abstract, Intro, Methods, Results)
    └─────────────────┬─────────────────┘
                      │
                      ▼
    ┌───────────────────────────────────┐
    │ 3. Semantic Embedding Pipeline    │ (Sentence-Transformers all-MiniLM-L6-v2)
    │    - 384-dimensional dense vectors│ - Cosine similarity matrix
    │    - Document Centroid Vector     │ - Maximal Marginal Relevance (MMR)
    └─────────┬───────────────┬─────────┘
              │               │
              ▼               ▼
    ┌──────────────────┐ ┌────────────────────────────────────────┐
    │ 4. Extractive    │ │ 5. Semantic Concept Knowledge Graph    │
    │    Summarizer    │ │    - KeyBERT phrase candidate scoring  │
    │  - Problem       │ │    - Phrase deduplication & passage map│
    │  - Methodology   │ │    - NetworkX topology modeling        │
    │  - Conclusion    │ │    - Degree sparsification (anti-clique│
    └──────────────────┘ │    - Interactive PyVis force physics   │
                         └────────────────────────────────────────┘
    ```
    """)

with doc_tab2:
    st.markdown("### 🧠 How to Explain PaperLens to Your College Teacher")
    st.markdown("""
    Here are the core concepts and mathematical foundations to explain during your project viva or evaluation:

    #### 1. Why Extractive Summarization instead of Generative LLMs?
    - **Academic Fidelity & Hallucination Prevention:** Generative LLMs can synthesize non-existent empirical results or misstate statistical significance. PaperLens guarantees 100% extractive fidelity: every sentence displayed is verbatim from the author's original manuscript.
    - **Privacy and Local Execution:** The system runs completely offline on the local CPU without transmitting sensitive unpublished research papers to proprietary third-party cloud APIs.
    - **Algorithmic Transparency:** Sentences are selected via well-defined cosine similarity against document centroid vectors and targeted research objective probes.

    #### 2. How the Three-Bullet Summary Works Mathematically:
    1. Each sentence $s_i$ is mapped to a vector $\\vec{v}_i \\in \\mathbb{R}^{384}$ using `all-MiniLM-L6-v2`.
    2. The document centroid $\\vec{\\mu} = \\frac{1}{N}\\sum_{i=1}^N \\vec{v}_i$ represents the global topic center.
    3. Three archetypal probe queries $\\vec{q}_{\\text{prob}}$, $\\vec{q}_{\\text{meth}}$, $\\vec{q}_{\\text{conc}}$ represent the three canonical pillars of scientific communication.
    4. Composite relevance score:
       $$\\text{Score}(s_i, q) = 0.50 \\cdot \\cos(\\vec{v}_i, \\vec{q}) + 0.30 \\cdot \\cos(\\vec{v}_i, \\vec{\\mu}) + \\text{Prior}(s_i)$$
    5. A diversity penalty (Maximal Marginal Relevance) prevents the model from choosing near-duplicate sentences.

    #### 3. How the Concept Graph Prevents "Hairball" Graphs:
    - Many basic graph implementations connect every keyword to every other keyword, creating an unreadable visual hairball.
    - PaperLens employs **degree sparsification**: each concept node connects only to its top-$k$ (default 3) highest semantic neighbors that exceed the cosine similarity cutoff threshold.
    - This preserves semantic clusters (e.g. machine learning models clustering together, evaluation metrics clustering together) while maintaining high readability.
    """)

with doc_tab3:
    st.markdown("### 🛠️ Technology Stack Breakdown")
    st.markdown("""
    | Component | Technology | Rationale |
    |---|---|---|
    | **Frontend UI** | Streamlit | Clean, reactive Python web application with rapid session-state synchronization |
    | **PDF Extraction** | PyMuPDF (`fitz`) | 10x faster than traditional PDF parsers with superior text coordinate extraction |
    | **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | Lightweight 384-dimensional dense semantic vectors optimized for fast CPU inference |
    | **Similarity** | `scikit-learn` Cosine Similarity | High-performance vectorized cosine distance computations |
    | **Graph Modeling** | NetworkX | Comprehensive graph analytics, degree centrality, and topology sparsification |
    | **Graph Rendering** | PyVis | Dynamic HTML5 canvas with smooth physics simulation and drag-and-drop interactivity |
    | **Discovery API** | CORE v3 API | Global open-access research repository integration with offline fallback |
    | **Testing** | pytest | Automated unit and integration testing suite |
    """)

with doc_tab4:
    st.markdown("### ❓ Frequently Asked Questions")
    st.markdown("""
    **Q1: What happens if I upload a scanned PDF?**  
    A: PaperLens examines the character and word density across all pages. If total extracted words are near zero, it alerts the user that the document is image-based and requires Optical Character Recognition (OCR), rather than crashing.

    **Q2: Does PaperLens require an internet connection?**  
    A: No! Once the lightweight embedding model is downloaded on first run (~80MB), the entire core application (PDF extraction, summarization, concept extraction, and graph visualization) works 100% offline.

    **Q3: Can I export the graph to open it in other tools?**  
    A: Yes! PaperLens provides exports in Standalone HTML (for any web browser), JSON (standard Node-Link format), and CSV (edge list compatible with Gephi and Cytoscape).
    """)
