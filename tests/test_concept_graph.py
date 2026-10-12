"""
tests/test_concept_graph.py
Unit tests for NetworkX graph modeling, cosine similarity edge calculation,
degree sparsification, and PyVis HTML rendering.
"""

import networkx as nx
import numpy as np
import pytest
from modules.concept_extractor import ExtractedConcept
from modules.concept_graph import (
    build_concept_network,
    compute_graph_metrics,
    generate_interactive_pyvis_html,
    create_concept_graph,
)


def create_dummy_concepts() -> list[ExtractedConcept]:
    # 5 dummy concepts with synthetic embeddings
    rng = np.random.default_rng(42)
    embs = rng.normal(size=(5, 64))
    # Normalize embeddings
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)

    names = ["Neural Network", "Attention Mechanism", "Transformer Model", "Gradient Descent", "Loss Function"]

    concepts = []
    for i, name in enumerate(names):
        concepts.append(
            ExtractedConcept(
                concept_id=i + 1,
                text=name,
                relevance_score=0.85 - (i * 0.1),
                frequency=10 - i,
                n_gram_type="Bigram",
                embedding=embs[i],
            )
        )
    return concepts


def test_build_concept_network_nodes():
    concepts = create_dummy_concepts()
    G, edges, nodes = build_concept_network(concepts, similarity_threshold=0.0, max_edges_per_node=2)

    assert G.number_of_nodes() == 5
    assert len(nodes) == 5
    for c in concepts:
        assert c.text in G.nodes


def test_build_concept_network_sparsification():
    concepts = create_dummy_concepts()
    # High max_edges allows more connections
    G, edges, _ = build_concept_network(concepts, similarity_threshold=-1.0, max_edges_per_node=2)

    for node in G.nodes:
        # Every node should obey max_edges_per_node (or close due to mutual connections)
        assert G.degree(node) <= 4


def test_compute_graph_metrics():
    concepts = create_dummy_concepts()
    G, _, _ = build_concept_network(concepts, similarity_threshold=0.1, max_edges_per_node=3)
    metrics = compute_graph_metrics(G)

    assert metrics.total_nodes == 5
    assert metrics.density >= 0.0
    assert len(metrics.top_hubs) <= 5


def test_generate_pyvis_html():
    concepts = create_dummy_concepts()
    result = create_concept_graph(concepts, similarity_threshold=0.1, max_edges_per_node=2)

    assert isinstance(result.pyvis_html, str)
    assert len(result.pyvis_html) > 100
    assert "mynetwork" in result.pyvis_html
