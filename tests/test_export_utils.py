"""
tests/test_export_utils.py
Unit tests for exporting summary (TXT, MD), concepts (CSV), and graph data (JSON, CSV, HTML).
"""

import json
import pytest
from modules.summarizer import SummaryResult, SummaryBullet
from modules.concept_extractor import ExtractedConcept, RelevantPassage
from modules.concept_graph import create_concept_graph
from modules.export_utils import (
    export_summary_txt,
    export_summary_md,
    export_concepts_csv,
    export_graph_json,
    export_graph_edges_csv,
    export_standalone_graph_html,
)
from tests.test_concept_graph import create_dummy_concepts


def get_dummy_summary() -> SummaryResult:
    bullets = [
        SummaryBullet("Problem & Objective", "🎯", "The primary challenge is computational complexity.", 1, 0.88, "Introduction"),
        SummaryBullet("Methodology & Approach", "⚙️", "We introduce multi-head attention blocks.", 2, 0.91, "Methodology"),
        SummaryBullet("Key Contribution & Conclusion", "🏆", "The model achieves record BLEU translation score.", 3, 0.85, "Conclusion"),
    ]
    return SummaryResult(True, bullets)


def test_export_summary_txt():
    summary = get_dummy_summary()
    meta = {"title": "Transformer Attention", "author": "Vaswani et al."}
    txt_out = export_summary_txt(summary, meta, "paper.pdf")

    assert "PAPERLENS" in txt_out
    assert "Transformer Attention" in txt_out
    assert "The primary challenge is computational complexity." in txt_out
    assert "Page 1" in txt_out


def test_export_summary_md():
    summary = get_dummy_summary()
    meta = {"title": "Transformer Attention", "author": "Vaswani et al."}
    md_out = export_summary_md(summary, meta, "paper.pdf")

    assert "# 📄 PaperLens Research Summary" in md_out
    assert "Vaswani et al." in md_out
    assert "### 🎯 1. Problem & Objective" in md_out


def test_export_concepts_csv():
    concepts = [
        ExtractedConcept(
            1, "Self Attention", 0.92, 14, "Bigram",
            [RelevantPassage(1, 2, "Self attention connects all tokens.", 1.0, True)]
        )
    ]
    csv_out = export_concepts_csv(concepts)

    assert "Concept_ID,Concept_Name,Relevance_Score,Frequency" in csv_out
    assert "Self Attention" in csv_out
    assert "0.920" in csv_out


def test_export_graph_json():
    concepts = create_dummy_concepts()
    graph_res = create_concept_graph(concepts, similarity_threshold=0.0, max_edges_per_node=2)
    json_out = export_graph_json(graph_res, {"title": "Test Paper"})

    parsed = json.loads(json_out)
    assert "nodes" in parsed
    assert "edges" in parsed
    assert len(parsed["nodes"]) == 5


def test_export_graph_edges_csv():
    edges = [{"source": "Concept A", "target": "Concept B", "similarity": 0.75}]
    csv_out = export_graph_edges_csv(edges)

    assert "Source,Target,Semantic_Similarity,Type" in csv_out
    assert "Concept A,Concept B,0.75,Undirected" in csv_out
