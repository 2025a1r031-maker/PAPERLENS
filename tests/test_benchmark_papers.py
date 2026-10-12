"""
tests/test_benchmark_papers.py
Integration tests running the complete end-to-end PaperLens pipeline on realistic
academic benchmark PDFs generated in sample_papers/.
"""

import os
import pytest
from modules.pdf_processor import extract_pdf_document
from modules.text_preprocessor import segment_document_sentences
from modules.summarizer import generate_research_summary
from modules.concept_extractor import extract_academic_concepts
from modules.concept_graph import create_concept_graph
from modules.export_utils import (
    export_summary_txt,
    export_summary_md,
    export_concepts_csv,
    export_graph_json,
    export_standalone_graph_html,
)

SAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_papers")
TRANSFORMER_PDF = os.path.join(SAMPLE_DIR, "attention_mechanism_transformer.pdf")
QUANTUM_PDF = os.path.join(SAMPLE_DIR, "quantum_error_correction_algorithms.pdf")


def test_transformer_paper_end_to_end():
    assert os.path.exists(TRANSFORMER_PDF), f"Sample file missing: {TRANSFORMER_PDF}"

    # 1. PDF Extraction
    pdf_res = extract_pdf_document(TRANSFORMER_PDF, filename="attention_mechanism_transformer.pdf")
    assert pdf_res.is_valid is True
    assert pdf_res.page_count == 3
    assert pdf_res.total_word_count > 400
    assert pdf_res.is_scanned is False
    assert "Transformer" in pdf_res.cleaned_full_text

    # 2. Sentence Segmentation
    sentences = segment_document_sentences(pdf_res.pages)
    assert len(sentences) >= 10
    sections = {s.section_hint for s in sentences}
    assert "Methodology" in sections or "Introduction" in sections

    # 3. 3-Bullet Summary
    summary_res = generate_research_summary(sentences, model=None)
    assert summary_res.is_success is True
    assert len(summary_res.bullets) == 3

    categories = [b.category for b in summary_res.bullets]
    assert "Problem & Objective" in categories
    assert "Methodology & Approach" in categories
    assert "Key Contribution & Conclusion" in categories

    for b in summary_res.bullets:
        assert b.page_number in [1, 2, 3]
        assert len(b.text) > 30

    # 4. Concept Extraction
    concepts = extract_academic_concepts(sentences, top_k=10, model=None)
    assert len(concepts) >= 5
    concept_names = [c.text.lower() for c in concepts]
    # Check for presence of core domain concepts
    assert any("attention" in c or "transformer" in c or "sequence" in c for c in concept_names)

    # 5. Concept Relationship Graph
    graph_res = create_concept_graph(concepts, similarity_threshold=0.1, max_edges_per_node=3)
    assert graph_res.metrics.total_nodes >= 5
    assert len(graph_res.pyvis_html) > 500
    assert "mynetwork" in graph_res.pyvis_html

    # 6. Export Validation
    txt_export = export_summary_txt(summary_res, pdf_res.metadata, pdf_res.filename)
    assert "EXECUTIVE THREE-BULLET RESEARCH SUMMARY" in txt_export

    md_export = export_summary_md(summary_res, pdf_res.metadata, pdf_res.filename)
    assert "# 📄 PaperLens Research Summary" in md_export

    csv_export = export_concepts_csv(concepts)
    assert "Concept_ID,Concept_Name,Relevance_Score" in csv_export

    html_export = export_standalone_graph_html(graph_res)
    assert "<!DOCTYPE html>" in html_export


def test_quantum_paper_end_to_end():
    assert os.path.exists(QUANTUM_PDF), f"Sample file missing: {QUANTUM_PDF}"

    pdf_res = extract_pdf_document(QUANTUM_PDF, filename="quantum_error_correction_algorithms.pdf")
    assert pdf_res.is_valid is True
    assert pdf_res.page_count == 3
    assert pdf_res.total_word_count > 400
    assert pdf_res.is_scanned is False

    sentences = segment_document_sentences(pdf_res.pages)
    assert len(sentences) >= 10

    summary_res = generate_research_summary(sentences, model=None)
    assert summary_res.is_success is True
    assert len(summary_res.bullets) == 3

    concepts = extract_academic_concepts(sentences, top_k=10, model=None)
    assert len(concepts) >= 5
    concept_names = [c.text.lower() for c in concepts]
    assert any("quantum" in c or "error" in c or "surface" in c or "decoherence" in c for c in concept_names)

    graph_res = create_concept_graph(concepts, similarity_threshold=0.1, max_edges_per_node=3)
    assert graph_res.metrics.total_nodes >= 5
