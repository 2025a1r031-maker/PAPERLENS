"""
pages/1_🔬_Research_Analyzer.py
Main Analysis Workspace for PaperLens.
Provides end-to-end PDF processing, multi-dimensional Detailed Theory in Crux Form,
Hierarchical Concept Mind Maps, KeyBERT concept extraction with contextual definitions,
interactive PyVis network modeling, and comprehensive multi-format exports.
"""

import io
import os
import streamlit as st
import streamlit.components.v1 as components

from modules.pdf_processor import extract_pdf_document, PDFDocumentResult
from modules.text_preprocessor import segment_document_sentences, SentenceItem
from modules.summarizer import (
    generate_research_summary,
    load_sentence_transformer_model,
    SummaryResult,
)
from modules.concept_extractor import (
    extract_academic_concepts,
    ExtractedConcept,
)
from modules.concept_graph import (
    create_concept_graph,
    ConceptGraphResult,
)
from modules.mindmap_generator import (
    generate_concept_mindmap,
    MindMapResult,
)
from modules.export_utils import (
    export_summary_txt,
    export_summary_md,
    export_concepts_csv,
    export_graph_json,
    export_graph_edges_csv,
    export_standalone_graph_html,
)

# Page configuration
st.set_page_config(
    page_title="PaperLens — Research Paper Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load custom CSS
css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Initializing Sentence-Transformers NLP Model (all-MiniLM-L6-v2)...")
def get_cached_nlp_model():
    """Loads and caches the SentenceTransformer embedding model."""
    return load_sentence_transformer_model("all-MiniLM-L6-v2")


def render_header():
    st.markdown("""
    <div class="brand-hero">
        <div class="brand-badge">🔬 Academic AI Workspace</div>
        <h1 class="brand-title">Research Paper Analyzer</h1>
        <p class="brand-tagline">
            Understand Research. Discover Connections.
        </p>
        <div style="margin-top: 10px; font-size: 14px; color: #cbd5e1; max-width: 720px; line-height: 1.55;">
            Extract comprehensive theory in crux form, explore hierarchical concept mind maps,
            and inspect verbatim context passages without hallucination.
        </div>
    </div>
    """, unsafe_allow_html=True)


def init_session_state():
    if "analysis_done" not in st.session_state:
        st.session_state.analysis_done = False
    if "pdf_result" not in st.session_state:
        st.session_state.pdf_result = None
    if "sentences" not in st.session_state:
        st.session_state.sentences = []
    if "summary_result" not in st.session_state:
        st.session_state.summary_result = None
    if "concepts" not in st.session_state:
        st.session_state.concepts = []
    if "graph_result" not in st.session_state:
        st.session_state.graph_result = None
    if "mindmap_result" not in st.session_state:
        st.session_state.mindmap_result = None
    if "selected_concept_idx" not in st.session_state:
        st.session_state.selected_concept_idx = 0


def run_pipeline(file_bytes: bytes, filename: str, num_concepts: int, sim_threshold: float, max_edges: int):
    """Executes the full PaperLens analysis pipeline with live step updates."""
    status = st.status("Analyzing Academic Document...", expanded=True)

    with status:
        st.write("📄 **Step 1/6:** Parsing PDF structure & extracting text with PyMuPDF...")
        pdf_res = extract_pdf_document(file_bytes, filename=filename)

        if not pdf_res.is_valid:
            status.update(label="❌ PDF Validation Failed", state="error", expanded=True)
            st.error(pdf_res.error_message or "Invalid PDF file.")
            return

        if pdf_res.is_scanned:
            status.update(label="⚠️ Scanned Document Detected", state="error", expanded=True)
            st.warning(pdf_res.error_message)
            st.session_state.pdf_result = pdf_res
            st.session_state.analysis_done = False
            return

        st.write("✂️ **Step 2/6:** Segmenting sentences & detecting academic section headers...")
        sentence_items = segment_document_sentences(pdf_res.pages)

        if len(sentence_items) < 3:
            status.update(label="⚠️ Insufficient Extractable Text", state="error", expanded=True)
            st.warning(
                f"Document contains only {len(sentence_items)} informative sentences. "
                "At least 3 complete sentences are required to perform semantic analysis."
            )
            st.session_state.pdf_result = pdf_res
            st.session_state.analysis_done = False
            return

        st.write("🧠 **Step 3/6:** Embedding text & extracting detailed theory in crux form...")
        model = get_cached_nlp_model()
        summary_res = generate_research_summary(sentence_items, model=model)

        st.write(f"🏷️ **Step 4/6:** Extracting top {num_concepts} key concepts with theoretical crux definitions...")
        concepts_list = extract_academic_concepts(
            sentence_items,
            top_k=num_concepts,
            model=model,
        )

        st.write("🌳 **Step 5/6:** Generating Hierarchical Concept Mind Map (5 Academic Pillars)...")
        paper_title = pdf_res.metadata.get("title") or filename
        mindmap_res = generate_concept_mindmap(paper_title, concepts_list, model=model)

        st.write("🕸️ **Step 6/6:** Modeling NetworkX semantic graph & generating PyVis physics...")
        graph_res = create_concept_graph(
            concepts_list,
            similarity_threshold=sim_threshold,
            max_edges_per_node=max_edges,
        )

        # Store in session state
        st.session_state.pdf_result = pdf_res
        st.session_state.sentences = sentence_items
        st.session_state.summary_result = summary_res
        st.session_state.concepts = concepts_list
        st.session_state.graph_result = graph_res
        st.session_state.mindmap_result = mindmap_res
        st.session_state.analysis_done = True
        st.session_state.selected_concept_idx = 0

        status.update(label="✅ Analysis Complete!", state="complete", expanded=False)


def main():
    init_session_state()
    render_header()

    # Sidebar: Document Selection & Configuration Controls
    with st.sidebar:
        st.markdown("### ⚙️ Analysis Settings")

        sample_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_papers")
        sample_files = []
        if os.path.exists(sample_dir):
            sample_files = [f for f in os.listdir(sample_dir) if f.endswith(".pdf")]

        input_choice = st.radio(
            "Document Input Mode:",
            ["Upload Custom PDF", "Use Benchmark Sample Paper"],
            index=0 if not sample_files else 1,
        )

        uploaded_file = None
        selected_sample = None

        if input_choice == "Upload Custom PDF":
            uploaded_file = st.file_uploader(
                "Upload Academic Research Paper (PDF):",
                type=["pdf"],
                help="Accepts academic papers up to 50MB. Text-based PDFs work best.",
            )
        else:
            if sample_files:
                selected_sample = st.selectbox(
                    "Choose Benchmark Paper:",
                    sample_files,
                    format_func=lambda x: x.replace("_", " ").replace(".pdf", "").title(),
                )
            else:
                st.info("No local sample papers found. Please upload a PDF.")

        st.markdown("---")
        st.markdown("### 🎛️ NLP & Graph Tuning")

        num_concepts = st.slider(
            "Extracted Concepts Count:",
            min_value=5,
            max_value=25,
            value=14,
            step=1,
            help="Number of core scientific concepts and multiword phrases to extract.",
        )

        sim_threshold = st.slider(
            "Semantic Similarity Cutoff:",
            min_value=0.20,
            max_value=0.85,
            value=0.40,
            step=0.02,
            help="Minimum cosine similarity required to draw an edge between two concepts.",
        )

        max_edges = st.slider(
            "Max Connections Per Concept:",
            min_value=1,
            max_value=6,
            value=3,
            step=1,
            help="Sparsification limit to keep the graph readable and avoid visual clutter.",
        )

        start_btn = st.button("🚀 Analyze Document", type="primary", use_container_width=True)

    # Check if a document was forwarded from Home Dashboard
    if "home_uploaded_bytes" in st.session_state and not st.session_state.analysis_done:
        home_bytes = st.session_state.pop("home_uploaded_bytes")
        home_name = st.session_state.pop("home_uploaded_name", "document.pdf")
        run_pipeline(home_bytes, home_name, num_concepts, sim_threshold, max_edges)

    # Trigger analysis on button click or file change
    if start_btn:
        if input_choice == "Upload Custom PDF":
            if uploaded_file is not None:
                file_bytes = uploaded_file.read()
                run_pipeline(file_bytes, uploaded_file.name, num_concepts, sim_threshold, max_edges)
            else:
                st.warning("Please upload a PDF file first.")
        else:
            if selected_sample:
                sample_path = os.path.join(sample_dir, selected_sample)
                with open(sample_path, "rb") as f:
                    file_bytes = f.read()
                run_pipeline(file_bytes, selected_sample, num_concepts, sim_threshold, max_edges)

    # Display Analysis Dashboard if available
    if st.session_state.analysis_done and st.session_state.pdf_result is not None:
        pdf_res: PDFDocumentResult = st.session_state.pdf_result
        summary_res: SummaryResult = st.session_state.summary_result
        concepts: list[ExtractedConcept] = st.session_state.concepts
        graph_res: ConceptGraphResult = st.session_state.graph_result
        mindmap_res: MindMapResult = st.session_state.mindmap_result

        # Re-generate mindmap if missing
        if mindmap_res is None:
            model = get_cached_nlp_model()
            mindmap_res = generate_concept_mindmap(
                pdf_res.metadata.get("title") or pdf_res.filename,
                concepts,
                model=model,
            )
            st.session_state.mindmap_result = mindmap_res

        # 1. Document Overview Stats Banner
        st.markdown(f"""
        <div class="stats-container">
            <div class="stat-box">
                <div class="stat-label">Pages</div>
                <div class="stat-value">{pdf_res.page_count}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Words Extracted</div>
                <div class="stat-value">{pdf_res.total_word_count:,}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Analyzed Sentences</div>
                <div class="stat-value">{len(st.session_state.sentences):,}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Est. Reading Time</div>
                <div class="stat-value">{pdf_res.estimated_reading_time_mins} min</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Concepts Extracted</div>
                <div class="stat-value">{len(concepts)}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Mind Map Pillars</div>
                <div class="stat-value">{mindmap_res.total_branches}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        meta_col1, meta_col2 = st.columns([2, 1])
        with meta_col1:
            st.markdown(f"**📄 Document Title:** {pdf_res.metadata.get('title', pdf_res.filename)}")
        with meta_col2:
            st.markdown(f"**✍️ Author:** {pdf_res.metadata.get('author', 'Not specified')}")

        st.markdown("<br/>", unsafe_allow_html=True)

        # Tabbed Dashboard Sections
        tab_theory, tab_mindmap, tab_concepts, tab_text, tab_exports, tab_method = st.tabs([
            "📚 Detailed Theory in Crux Form",
            "🧠 Concept Mind Map & Network",
            "🏷️ Concept Encyclopedia & Passages",
            "📑 Text & Page Browser",
            "📥 Export & Downloads",
            "🔬 Methodology & Architecture",
        ])

        # TAB 1: DETAILED THEORY IN CRUX FORM
        with tab_theory:
            st.markdown("### 📚 Detailed Academic Theory in Crux Form")
            st.caption(
                "In-depth theoretical breakdown extracted via semantic centrality across the entire paper. "
                "Covers research gap, theoretical framework, system architecture, empirical benchmarks, and strategic implications."
            )

            if not summary_res.is_success:
                st.warning(summary_res.error_message or "Summary could not be generated.")
            else:
                # Part A: Executive 3-Bullet Snapshot
                st.markdown("#### ⚡ Executive Summary Snapshot")
                pill_classes = {
                    "Problem & Objective": ("pill-problem", "🎯"),
                    "Methodology & Approach": ("pill-method", "⚙️"),
                    "Key Contribution & Conclusion": ("pill-contribution", "🏆"),
                }

                for b in summary_res.bullets:
                    p_class, icon = pill_classes.get(b.category, ("pill-method", "📌"))
                    st.markdown(f"""
                    <div class="summary-card">
                        <span class="summary-pill {p_class}">{icon} {b.category}</span>
                        <div class="summary-text">"{b.text}"</div>
                        <div class="summary-meta">
                            <span>📄 Source: Page {b.page_number} ({b.section_hint} Section)</span>
                            <span>•</span>
                            <span>🎯 Centrality Score: <code>{b.centrality_score:.3f}</code></span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("---")

                # Part B: Multi-Section Detailed Theory Breakdown
                st.markdown("#### 📖 Comprehensive Theory & Crux Breakdown")

                if summary_res.theory_crux and summary_res.theory_crux.sections:
                    for sec in summary_res.theory_crux.sections:
                        with st.expander(f"{sec.icon} {sec.title} ({len(sec.crux_points)} Crux Points)", expanded=True):
                            st.caption(sec.description)
                            for pt in sec.crux_points:
                                st.markdown(f"""
                                <div class="theory-crux-item">
                                    <div class="theory-crux-quote">"{pt.text}"</div>
                                    <div class="theory-crux-footer">
                                        <span>📄 Page {pt.page_number}</span>
                                        <span>•</span>
                                        <span>Section: <b>{pt.section_hint}</b></span>
                                        <span>•</span>
                                        <span>Relevance Centrality: <code>{pt.centrality_score:.3f}</code></span>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                else:
                    st.info("Additional detailed theory sections are available when analyzing documents with 10+ sentences.")

                st.markdown("""
                <div class="methodology-callout">
                    ℹ️ <b>Academic Integrity Notice:</b> This theory breakdown is strictly extractive. PaperLens does not use
                    a generative LLM that could hallucinate findings or invent synthetic data. Every statement shown above
                    is verbatim from the uploaded paper.
                </div>
                """, unsafe_allow_html=True)

        # TAB 2: INTERACTIVE CONCEPT MIND MAP & NETWORK
        with tab_mindmap:
            st.markdown("### 🧠 Interactive Concept Mind Map & Network")
            st.caption(
                "Explore the paper's conceptual architecture as a hierarchical Mind Map or switch to an associative semantic network."
            )

            map_mode = st.radio(
                "Select Visualization Perspective:",
                ["🌳 Hierarchical Mind Map (Left-to-Right Tree)", "🕸️ Semantic Association Network (Force Physics)"],
                horizontal=True,
            )

            if map_mode == "🌳 Hierarchical Mind Map (Left-to-Right Tree)":
                st.markdown(f"""
                <div class="mindmap-header-card">
                    <div style="font-size: 13px; text-transform: uppercase; letter-spacing: 0.05em; opacity: 0.8;">Hierarchical Cognitive Mind Map</div>
                    <div style="font-size: 20px; font-weight: 700; margin: 4px 0;">{mindmap_res.root_title}</div>
                    <div style="font-size: 13px; opacity: 0.9;">
                        Spans 5 Academic Pillars: Problem & Motivation, Theoretical Foundations, Architecture & Methods, Empirical Findings, Significance & Impact.
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Render PyVis Hierarchical Mind Map HTML
                components.html(mindmap_res.pyvis_html, height=620, scrolling=False)

                # Pillar Summary Cards Grid
                st.markdown("#### 📂 Mind Map Pillar Breakdown")
                p_cols = st.columns(len(mindmap_res.branches))
                for idx, branch in enumerate(mindmap_res.branches):
                    with p_cols[idx]:
                        st.markdown(f"""
                        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-top: 4px solid {branch.pillar_color}; border-radius: 10px; padding: 12px; height: 100%;">
                            <strong style="color: #0f172a; font-size: 13px;">{branch.pillar_title}</strong>
                            <div style="font-size: 11px; color: #64748b; margin-top: 4px; margin-bottom: 8px;">{len(branch.concepts)} Concepts</div>
                            {''.join([f'<div style=\"background: #f1f5f9; font-size: 11px; padding: 3px 6px; border-radius: 4px; margin-bottom: 4px;\">{c.text}</div>' for c in branch.concepts[:4]])}
                        </div>
                        """, unsafe_allow_html=True)

                # Mermaid Diagram Expander
                with st.expander("📊 View Live Mermaid.js Mindmap Diagram", expanded=False):
                    st.code(mindmap_res.mermaid_markdown, language="mermaid")

            else:
                # Semantic Network Graph View
                st.markdown("#### 🕸️ Associative Concept Relationship Network")
                st.caption(
                    "Network topology calculated via pairwise cosine similarity between sentence-transformer embeddings."
                )

                # Re-compute network if tuning sliders were modified
                graph_res_active = create_concept_graph(
                    concepts,
                    similarity_threshold=sim_threshold,
                    max_edges_per_node=max_edges,
                )

                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                with m_col1:
                    st.metric("Graph Nodes", graph_res_active.metrics.total_nodes)
                with m_col2:
                    st.metric("Semantic Edges", graph_res_active.metrics.total_edges)
                with m_col3:
                    st.metric("Network Density", f"{graph_res_active.metrics.density:.3f}")
                with m_col4:
                    hub_text = ", ".join([h[0] for h in graph_res_active.metrics.top_hubs[:2]]) if graph_res_active.metrics.top_hubs else "None"
                    st.metric("Top Hubs", hub_text)

                if graph_res_active.metrics.total_nodes > 0:
                    components.html(graph_res_active.pyvis_html, height=580, scrolling=False)
                else:
                    st.warning("Not enough concepts extracted to construct a relationship graph.")

        # TAB 3: CONCEPT ENCYCLOPEDIA & PASSAGE LOOKUP
        with tab_concepts:
            st.markdown("### 🏷️ Concept Encyclopedia & Theory Cards")
            st.caption("Each extracted concept includes its contextual theoretical definition and verbatim citations.")

            # Concept Cards Grid with Theoretical Crux
            for c in concepts:
                with st.container():
                    st.markdown(f"""
                    <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 18px 20px; margin-bottom: 14px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <strong style="color: #0f172a; font-size: 16px;">#{c.concept_id} {c.text}</strong>
                            <div>
                                <span style="background: #eef2ff; color: #4338ca; font-size: 11px; font-weight: 700; padding: 3px 8px; border-radius: 6px; margin-right: 6px;">Relevance: {c.relevance_score:.2f}</span>
                                <span style="background: #f1f5f9; color: #475569; font-size: 11px; font-weight: 600; padding: 3px 8px; border-radius: 6px;">{c.n_gram_type}</span>
                            </div>
                        </div>
                        <div style="font-size: 13.5px; color: #334155; line-height: 1.6; margin-bottom: 8px;">
                            <b>Theoretical Crux:</b> <em>"{c.theoretical_crux}"</em>
                        </div>
                        <div style="font-size: 12px; color: #64748b;">
                            <span>Occurrences: <b>{c.frequency}</b></span>
                            <span style="margin: 0 8px;">•</span>
                            <span>Associated Passages: <b>{len(c.top_passages)}</b></span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("#### 🔍 Context Passage Inspector")
            st.write("Inspect verbatim sentences from the original paper discussing any concept:")

            concept_names = [f"#{c.concept_id}: {c.text} ({c.n_gram_type})" for c in concepts]
            selected_concept_str = st.selectbox(
                "Select Concept to Inspect:",
                concept_names,
                index=st.session_state.selected_concept_idx,
            )

            selected_id = int(selected_concept_str.split(":")[0].replace("#", ""))
            matched_concept = next((c for c in concepts if c.concept_id == selected_id), concepts[0])

            if matched_concept.top_passages:
                st.markdown(f"Found **{len(matched_concept.top_passages)} verbatim passages** for **{matched_concept.text}**:")
                for p in matched_concept.top_passages:
                    exact_badge = "🎯 Exact Match" if p.is_exact_match else "🔍 Semantic Proximity"
                    st.markdown(f"""
                    <div class="passage-card">
                        <div class="passage-page-tag">Page {p.page_number} • {exact_badge}</div>
                        <div class="passage-text">"{p.text}"</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info(f"No isolated sentences found explicitly matching '{matched_concept.text}'.")

        # TAB 4: TEXT & PAGE BROWSER
        with tab_text:
            st.markdown("### 📑 Extracted Text & Page Browser")
            st.caption("Review the sanitized text extracted by PyMuPDF across each page of the document.")

            search_query = st.text_input("🔍 Search within extracted document text:", placeholder="e.g. algorithm, dataset, error...")

            page_tabs = [f"Page {p.page_number}" for p in pdf_res.pages]
            selected_page_tab = st.radio("Select Page:", page_tabs, horizontal=True)
            page_index = int(selected_page_tab.replace("Page ", "")) - 1

            page_obj = pdf_res.pages[page_index]
            display_text = page_obj.clean_text

            if search_query:
                highlighted = display_text.replace(search_query, f"**_{search_query.upper()}_**")
                st.markdown(highlighted)
            else:
                st.text_area(
                    label=f"Page {page_obj.page_number} Content ({page_obj.word_count} words):",
                    value=display_text,
                    height=350,
                )

        # TAB 5: EXPORT & DOWNLOAD CENTER
        with tab_exports:
            st.markdown("### 📥 Export & Download Center")
            st.caption("Download all generated analysis, detailed theory crux, mind maps, and graphs.")

            exp_col1, exp_col2 = st.columns(2)

            with exp_col1:
                st.markdown("#### 📄 Theory Crux & Executive Summary")
                summary_txt_data = export_summary_txt(summary_res, pdf_res.metadata, pdf_res.filename)
                st.download_button(
                    label="💾 Download Summary & Theory as TXT (.txt)",
                    data=summary_txt_data,
                    file_name=f"{os.path.splitext(pdf_res.filename)[0]}_theory_crux.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

                summary_md_data = export_summary_md(summary_res, pdf_res.metadata, pdf_res.filename)
                st.download_button(
                    label="📝 Download Summary & Theory as Markdown (.md)",
                    data=summary_md_data,
                    file_name=f"{os.path.splitext(pdf_res.filename)[0]}_theory_crux.md",
                    mime="text/markdown",
                    use_container_width=True,
                )

                st.markdown("#### 🏷️ Concepts Encyclopedia")
                concepts_csv_data = export_concepts_csv(concepts)
                st.download_button(
                    label="📊 Download Concepts as CSV (.csv)",
                    data=concepts_csv_data,
                    file_name=f"{os.path.splitext(pdf_res.filename)[0]}_concepts.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            with exp_col2:
                st.markdown("#### 🧠 Mind Map & Network Topology")
                st.download_button(
                    label="🌐 Download Standalone Mind Map HTML (.html)",
                    data=mindmap_res.pyvis_html,
                    file_name=f"{os.path.splitext(pdf_res.filename)[0]}_mindmap.html",
                    mime="text/html",
                    use_container_width=True,
                    help="Can be opened independently in Google Chrome, Edge, or Firefox.",
                )

                st.download_button(
                    label="📊 Download Mermaid Mindmap Markdown (.md)",
                    data=f"```mermaid\n{mindmap_res.mermaid_markdown}\n```",
                    file_name=f"{os.path.splitext(pdf_res.filename)[0]}_mindmap_mermaid.md",
                    mime="text/markdown",
                    use_container_width=True,
                )

                graph_html_data = export_standalone_graph_html(graph_res)
                st.download_button(
                    label="🕸️ Download Concept Network as HTML (.html)",
                    data=graph_html_data,
                    file_name=f"{os.path.splitext(pdf_res.filename)[0]}_network.html",
                    mime="text/html",
                    use_container_width=True,
                )

                graph_json_data = export_graph_json(graph_res, pdf_res.metadata)
                st.download_button(
                    label="📦 Download Graph as JSON (.json)",
                    data=graph_json_data,
                    file_name=f"{os.path.splitext(pdf_res.filename)[0]}_graph.json",
                    mime="application/json",
                    use_container_width=True,
                )

        # TAB 6: METHODOLOGY & ARCHITECTURE
        with tab_method:
            st.markdown("### 🔬 PaperLens Methodology & Algorithmic Pipeline")
            st.markdown("""
            PaperLens is designed specifically for academic transparency, rigorous NLP principles, and local execution.

            #### 1. PDF Parsing Pipeline (PyMuPDF / `fitz`)
            - PyMuPDF reads the binary PDF structure directly into page-level streams.
            - Cleans line-break hyphens (`hyphen-\\nated` &rarr; `hyphenated`), non-standard spaces, and header/footer pagination artefacts.
            - Automatically detects scanned/image-only PDFs by monitoring character and word density per page.

            #### 2. Detailed Theory Crux Extraction Engine
            - Candidate sentences are projected into a 384-dimensional dense semantic space via `all-MiniLM-L6-v2`.
            - Extracts multi-point theoretical crux points across 5 dimensions:
              1. *Research Gap & Problem Statement*
              2. *Theoretical Foundations & Principles*
              3. *Architectural Pipeline & Algorithmic Design*
              4. *Empirical Evidence & Benchmark Performance*
              5. *Significance & Future Horizons*

            #### 3. Hierarchical Concept Mind Map (NetworkX + PyVis)
            - Roots the central theme and branches into 5 cognitive pillars.
            - Automatically classifies concepts into branches using cosine similarity against archetypal pillar queries.
            - Renders a Left-to-Right hierarchical tree with directional links and crux tooltips.
            """)

    else:
        # Empty State
        st.info("👈 Upload an academic research paper or choose a benchmark sample from the sidebar to begin analysis.")


if __name__ == "__main__":
    main()
