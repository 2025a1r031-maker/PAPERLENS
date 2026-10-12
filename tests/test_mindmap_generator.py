"""
tests/test_mindmap_generator.py
Unit tests for hierarchical concept mind map generation, pillar classification,
PyVis tree layout, and Mermaid syntax rendering.
"""

import pytest
from modules.mindmap_generator import (
    generate_concept_mindmap,
    classify_concepts_into_pillars,
    build_hierarchical_mindmap_network,
    generate_hierarchical_pyvis_html,
    generate_mermaid_mindmap_markdown,
    MindMapResult,
    MINDMAP_PILLARS,
)
from tests.test_concept_graph import create_dummy_concepts


def test_classify_concepts_into_pillars():
    concepts = create_dummy_concepts()
    classified = classify_concepts_into_pillars(concepts, model=None)

    assert isinstance(classified, dict)
    assert len(classified) == len(MINDMAP_PILLARS)
    total_assigned = sum(len(c_list) for c_list in classified.values())
    assert total_assigned == len(concepts)


def test_build_hierarchical_mindmap_network():
    concepts = create_dummy_concepts()
    result = generate_concept_mindmap("Attention Mechanisms", concepts, model=None)

    assert isinstance(result, MindMapResult)
    assert result.root_title == "Attention Mechanisms"
    assert result.total_branches == 5
    assert result.total_nodes >= 6
    assert "mynetwork" in result.pyvis_html
    assert "mindmap" in result.mermaid_markdown


def test_mermaid_mindmap_syntax():
    concepts = create_dummy_concepts()
    result = generate_concept_mindmap("Transformer Paper", concepts, model=None)

    md = result.mermaid_markdown
    assert md.startswith("mindmap")
    assert "Transformer Paper" in md
    assert "Problem & Motivation" in md


def test_empty_concepts_mindmap():
    result = generate_concept_mindmap("Empty Paper", [], model=None)
    assert isinstance(result, MindMapResult)
    assert result.total_branches == 5
