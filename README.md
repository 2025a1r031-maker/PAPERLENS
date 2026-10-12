# 🔬 PaperLens — AI-Powered Academic Research Paper Analyzer

> **"Understand Research. Discover Connections."**  
> *An AI-powered academic intelligence application built with Python, Streamlit, PyMuPDF, Sentence-Transformers, and NetworkX.*

---

## 🌟 Overview

**PaperLens** is an academic research paper analysis web application designed for students, researchers, and university evaluators. Navigating lengthy, dense academic PDFs is traditionally time-consuming and cognitively demanding. PaperLens streamlines this process by:

1. **Extracting & Sanitizing** text directly from uploaded academic PDFs without server-side third-party cloud dependencies.
2. **Detailed Theory in Crux Form**: Breaks down the paper into 5 comprehensive theoretical pillars (Research Gap & Problem, Theoretical Foundations & Principles, Architectural Pipeline & Methods, Empirical Evidence & Benchmarks, and Significance & Strategic Horizons) with verbatim citations and centrality scores.
3. **Executive 3-Bullet Summary**: High-level snapshot capturing the research **Problem & Objective**, **Methodology & Approach**, and **Key Contribution & Conclusion**.
4. **Hierarchical Concept Mind Map**: Models a multi-level cognitive tree (Paper Theme $\to$ 5 Thematic Pillars $\to$ Scientific Concepts $\to$ Theoretical Crux Snippets) in PyVis and live Mermaid.js diagrams.
5. **Extracting Key Scientific Concepts**: Multi-word phrases ranked via semantic topic similarity, paired with theoretical contextual definitions.
6. **Interactive Passage Inspector**: Links every extracted concept directly to the verbatim sentences and page citations in the manuscript.
7. **Interactive Concept Relationship Network**: Tunable NetworkX & PyVis physics graph with degree sparsification.
8. **Multi-Format Export Center**: Downloads theory crux summaries (TXT, Markdown), concepts (CSV), mind maps (HTML, Mermaid MD), and network graphs (JSON, CSV, HTML).
9. **Academic Literature Discovery**: Connects to the open-access **CORE API** (`core.ac.uk`) with curated benchmark offline fallbacks.

---

## 📐 Algorithmic Pipeline & Methodology

```
[Academic PDF Document]
           │
           ▼
┌───────────────────────────────────────┐
│ 1. Ingestion & Sanitization Engine    │ (PyMuPDF / fitz)
│    - Byte signature & size validation │ - De-hyphenates broken words across lines
│    - Scanned/raster document detection│ - Extracts page coordinates & metadata
└──────────────────┬────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────┐
│ 2. Sentence & Structural Preprocessor │ (Abbreviation-Aware Regex Segmentation)
│    - Protects academic abbreviations  │ - Heuristic section classification
│      (e.g., et al., i.e., Fig., 3.14) │   (Abstract, Intro, Methods, Results, Conclusion)
└──────────────────┬────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────┐
│ 3. Dense Semantic Embedding Space     │ (Sentence-Transformers all-MiniLM-L6-v2)
│    - 384-dimensional dense vectors    │ - Document centroid representation
│    - Cosine similarity distance matrix│ - Maximal Marginal Relevance (MMR)
└──────────┬─────────────────┬──────────┘
           │                 │
           ▼                 ▼
┌────────────────────┐ ┌────────────────────────────────────────┐
│ 4. Extractive      │ │ 5. Concept Knowledge Graph Engine      │
│    Summarizer      │ │    - KeyBERT-style semantic phrase rank│
│  - Problem Probe   │ │    - Synonymous phrase deduplication   │
│  - Method Probe    │ │    - NetworkX topology modeling        │
│  - Findings Probe  │ │    - Degree sparsification (anti-clique│
└────────────────────┘ │    - PyVis interactive physics canvas  │
                       └────────────────────────────────────────┘
```

### 1. Extractive Summarization vs. Generative LLMs
- **Zero Hallucination:** PaperLens does not use a generative LLM that could invent non-existent statistics or misstate empirical findings. Every sentence displayed in the summary is verbatim from the original manuscript.
- **Mathematical Centrality:** Sentences are ranked using cosine similarity against document centroid vectors and targeted research objective probes with Maximal Marginal Relevance (MMR) to prevent redundancy.

### 2. Concept Relationship Graph & Anti-Hairball Sparsification
- **Semantic Distance:** Concept nodes are connected when the cosine similarity between their dense embeddings meets or exceeds the user-configured cutoff.
- **Degree Sparsification:** Many keyword graphs degenerate into an unintelligible visual "hairball." PaperLens applies degree sparsification (top-$k$ strongest connections per node) to preserve meaningful semantic clusters while maintaining visual clarity.

---

## 🛠️ Technology Stack

| Component | Library / Model | Purpose |
|---|---|---|
| **Programming Language** | Python 3.11 / 3.13 | High-performance backend execution |
| **User Interface** | Streamlit | Modern reactive multi-page web dashboard |
| **PDF Extraction** | PyMuPDF (`fitz` / `pymupdf`) | Fast native PDF text extraction & page parsing |
| **Semantic Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | 384-D dense embeddings running locally on CPU |
| **Similarity Computation** | `scikit-learn` | Vectorized cosine similarity distance |
| **Graph Modeling** | NetworkX | Graph data structures, metrics, and sparsification |
| **Interactive Graph UI**| PyVis | Dynamic HTML5 canvas with physics simulation |
| **Data Handling** | pandas | Tabular data processing and CSV generation |
| **Paper Discovery** | CORE API v3 (`requests`) | Open-access academic paper search |
| **Testing** | pytest | Automated unit and integration test suite |

---

## 📁 Project Folder Structure

```
paperlens prototype/
├── app.py                           # Main application landing dashboard
├── requirements.txt                 # Project dependencies
├── README.md                        # Documentation and defense notes
├── pytest.ini                       # Test configuration with root pythonpath
├── .gitignore                       # Git exclusions (venv, cache, env)
├── .env.example                     # Environment template (CORE API key)
├── assets/
│   └── style.css                    # Professional academic CSS design system
├── modules/
│   ├── __init__.py                  # Package initializer
│   ├── pdf_processor.py             # PyMuPDF extraction, validation & OCR detection
│   ├── text_preprocessor.py         # Abbreviation-aware segmentation & section tagging
│   ├── summarizer.py                # 3-bullet extractive semantic summarizer
│   ├── concept_extractor.py         # KeyBERT-style concept extractor & passage finder
│   ├── concept_graph.py             # NetworkX graph modeling & PyVis visualizer
│   ├── export_utils.py              # TXT, MD, CSV, JSON, HTML export generators
│   └── core_discovery.py            # CORE API discovery client with fallbacks
├── pages/
│   ├── 1_🔬_Research_Analyzer.py    # Full interactive research paper analysis studio
│   ├── 2_🔍_Paper_Discovery.py      # Academic paper discovery search engine
│   └── 3_ℹ️_About_PaperLens.py      # Architecture, methodology & college defense guide
├── sample_papers/
│   ├── generate_samples.py          # Script to generate benchmark academic papers
│   ├── attention_mechanism_transformer.pdf
│   └── quantum_error_correction_algorithms.pdf
└── tests/
    ├── test_pdf_processor.py        # PDF extraction & scanned detection tests
    ├── test_summarizer.py           # Summarizer & diversity tests
    ├── test_concept_extractor.py    # Concept extraction & passage lookup tests
    ├── test_concept_graph.py        # Graph modeling & PyVis tests
    ├── test_export_utils.py         # Export file generator tests
    └── test_benchmark_papers.py     # End-to-end integration tests on sample papers
```

---

## 🚀 Getting Started (Windows PowerShell)

Follow these steps to run PaperLens locally on Windows:

### Step 1: Open Windows PowerShell & Navigate to Project Directory
```powershell
cd "c:\Users\krish\Desktop\paperlens prototype"
```

### Step 2: Create and Activate Virtual Environment
```powershell
# Create virtual environment (.venv)
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1
```
*(If you see an execution policy error in PowerShell, run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`)*

### Step 3: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Run Automated Tests
```powershell
pytest -v
```
*Expected: 26 passed tests verifying extraction, summarization, graph generation, and exports.*

### Step 5: Launch the Streamlit Web Application
```powershell
streamlit run app.py
```

The application will launch and open automatically in your browser at:
👉 **`http://localhost:8501`**

---

## 🔑 Optional Configuration (CORE API)

PaperLens runs 100% locally on your machine without requiring an API key.

If you wish to search live global repositories (200M+ papers) on the **Paper Discovery** page:
1. Register for a free API key at [https://core.ac.uk/services/api](https://core.ac.uk/services/api).
2. Copy `.env.example` to `.env`:
   ```powershell
   Copy-Item .env.example .env
   ```
3. Open `.env` and add your key:
   ```env
   CORE_API_KEY=your_actual_core_api_key_here
   ```
4. If left blank, PaperLens seamlessly uses built-in curated academic papers.

---

## 🎓 College Viva & Evaluation Defense Points

When presenting PaperLens to faculty evaluators or your project guide:

1. **Academic Honesty:** Explain why extractive semantic summarization was chosen over generative LLMs (prevents fabrication of research data, runs offline, guarantees citations are verifiable).
2. **Embeddings & Geometry:** Explain how `all-MiniLM-L6-v2` converts sentences into 384-dimensional vectors and how cosine similarity identifies the document's central semantic coordinates.
3. **Graph Theory:** Explain how NetworkX builds the graph and how degree sparsification prevents dense visual clutter while highlighting primary domain hubs.
4. **Scanned PDF Handling:** Explain that the PDF engine monitors word density per page, preventing crashes on flat image scans and prompting the user for OCR.

---

## 📄 License
Academic & Educational Project License. Built for university demonstration and academic evaluation.
