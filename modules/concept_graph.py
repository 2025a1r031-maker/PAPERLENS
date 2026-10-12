"""
modules/concept_graph.py
Interactive academic concept relationship graph modeling using NetworkX and PyVis.
Calculates semantic similarity between concept embeddings, applies intelligent
degree sparsification to avoid dense hairballs, and exports embeddable HTML.
"""

from dataclasses import dataclass
import json
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple
import networkx as nx
import numpy as np
from pyvis.network import Network
from sklearn.metrics.pairwise import cosine_similarity

from modules.concept_extractor import ExtractedConcept

logger = logging.getLogger(__name__)


@dataclass
class GraphMetrics:
    total_nodes: int
    total_edges: int
    density: float
    top_hubs: List[Tuple[str, int]]  # (Concept name, degree)
    is_connected: bool
    num_isolated_nodes: int


@dataclass
class ConceptGraphResult:
    graph: nx.Graph
    metrics: GraphMetrics
    pyvis_html: str
    edge_list: List[Dict[str, Any]]
    node_list: List[Dict[str, Any]]


def build_concept_network(
    concepts: List[ExtractedConcept],
    similarity_threshold: float = 0.45,
    max_edges_per_node: int = 3,
) -> Tuple[nx.Graph, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Constructs a NetworkX graph from extracted concepts using semantic cosine similarity.

    Args:
        concepts: List of extracted concepts with valid embeddings.
        similarity_threshold: Minimum cosine similarity required to form an edge.
        max_edges_per_node: Maximum number of strongest connections allowed per node (sparsification).

    Returns:
        (graph, edge_list, node_list)
    """
    G = nx.Graph()

    if not concepts:
        return G, [], []

    # 1. Add Nodes
    node_list: List[Dict[str, Any]] = []

    # Colors for different tiers
    for c in concepts:
        # Node size based on relevance and frequency
        size = int(18 + (c.relevance_score * 20) + min(12, c.frequency * 2))

        # Color mapping by n-gram type and importance
        if c.relevance_score > 0.7:
            color = "#4f46e5"  # Vibrant Indigo for top tier
        elif c.n_gram_type in ["Bigram", "Trigram"]:
            color = "#7c3aed"  # Royal Violet for multi-word phrases
        else:
            color = "#0284c7"  # Deep Cyan for single keywords

        # Hover tooltip
        tooltip_html = (
            f"<div style='font-family: sans-serif; padding: 6px; font-size: 13px;'>"
            f"<strong style='color: #1e293b; font-size: 14px;'>{c.text}</strong><br/>"
            f"<span style='color: #475569;'>Relevance: <b>{c.relevance_score:.2f}</b></span><br/>"
            f"<span style='color: #475569;'>Occurrences: <b>{c.frequency}</b> ({c.n_gram_type})</span><br/>"
            f"<span style='color: #64748b; font-size: 11px;'>Drag to reposition | Scroll to zoom</span>"
            f"</div>"
        )

        G.add_node(
            c.text,
            id=c.concept_id,
            label=c.text,
            title=tooltip_html,
            value=size,
            color=color,
            relevance=c.relevance_score,
            frequency=c.frequency,
            n_gram_type=c.n_gram_type,
            font={"color": "#0f172a", "size": 13, "face": "Inter, system-ui, sans-serif", "strokeWidth": 2, "strokeColor": "#ffffff"},
        )

        node_list.append({
            "id": c.concept_id,
            "label": c.text,
            "relevance": c.relevance_score,
            "frequency": c.frequency,
            "type": c.n_gram_type,
        })

    # 2. Extract embeddings matrix
    valid_concepts = [c for c in concepts if c.embedding is not None]

    edge_list: List[Dict[str, Any]] = []

    if len(valid_concepts) >= 2:
        embeddings_matrix = np.array([c.embedding for c in valid_concepts])
        sim_matrix = cosine_similarity(embeddings_matrix)

        n = len(valid_concepts)

        # To prevent excessive density / hairball graph:
        # Collect top candidates per node, then add them
        candidate_edges_per_node: Dict[int, List[Tuple[float, int, int]]] = {i: [] for i in range(n)}

        for i in range(n):
            for j in range(i + 1, n):
                sim = float(sim_matrix[i, j])
                if sim >= similarity_threshold:
                    candidate_edges_per_node[i].append((sim, i, j))
                    candidate_edges_per_node[j].append((sim, j, i))

        # Add top-k strongest edges for each node
        added_edges = set()

        for i in range(n):
            # Sort candidates by similarity descending
            sorted_cands = sorted(candidate_edges_per_node[i], key=lambda x: x[0], reverse=True)
            top_cands = sorted_cands[:max_edges_per_node]

            for sim, u_idx, v_idx in top_cands:
                edge_key = tuple(sorted([valid_concepts[u_idx].text, valid_concepts[v_idx].text]))
                if edge_key not in added_edges:
                    added_edges.add(edge_key)
                    src_text = valid_concepts[u_idx].text
                    dst_text = valid_concepts[v_idx].text

                    edge_tooltip = (
                        f"<div style='font-family: sans-serif; padding: 4px; font-size: 12px;'>"
                        f"<strong>Semantic Proximity:</strong> {sim:.2f}<br/>"
                        f"<span style='color: #64748b;'>{src_text} &harr; {dst_text}</span>"
                        f"</div>"
                    )

                    G.add_edge(
                        src_text,
                        dst_text,
                        weight=round(sim, 3),
                        title=edge_tooltip,
                        value=max(1, int(sim * 5)),
                        color={"color": "#cbd5e1", "highlight": "#4f46e5", "hover": "#6366f1"},
                    )

                    edge_list.append({
                        "source": src_text,
                        "target": dst_text,
                        "similarity": round(sim, 3),
                    })

    return G, edge_list, node_list


def compute_graph_metrics(G: nx.Graph) -> GraphMetrics:
    """
    Computes analytical network properties for reporting.
    """
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    if num_nodes == 0:
        return GraphMetrics(0, 0, 0.0, [], False, 0)

    density = float(round(nx.density(G), 3))
    degrees = dict(G.degree())
    sorted_degrees = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
    top_hubs = sorted_degrees[:5]

    is_connected = nx.is_connected(G) if num_nodes > 1 and num_edges > 0 else False
    num_isolated = sum(1 for d in degrees.values() if d == 0)

    return GraphMetrics(
        total_nodes=num_nodes,
        total_edges=num_edges,
        density=density,
        top_hubs=top_hubs,
        is_connected=is_connected,
        num_isolated_nodes=num_isolated,
    )


def generate_interactive_pyvis_html(
    G: nx.Graph,
    height: str = "560px",
    width: str = "100%",
) -> str:
    """
    Renders NetworkX graph into an interactive PyVis HTML widget with physics simulation.
    """
    if G.number_of_nodes() == 0:
        return (
            "<div style='text-align: center; padding: 40px; color: #64748b; font-family: sans-serif;'>"
            "No concepts available to render graph.</div>"
        )

    # Initialize PyVis Network
    net = Network(
        height=height,
        width=width,
        bgcolor="#f8fafc",
        font_color="#0f172a",
        directed=False,
        notebook=False,
    )

    # Load from NetworkX
    net.from_nx(G)

    # Configure physics simulation for organic, responsive layout
    physics_options = {
        "nodes": {
            "borderWidth": 2,
            "borderWidthSelected": 4,
            "shadow": {"enabled": True, "color": "rgba(0,0,0,0.08)", "size": 6, "x": 2, "y": 2},
        },
        "edges": {
            "smooth": {"type": "continuous", "roundness": 0.3},
            "shadow": {"enabled": False},
        },
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -50,
                "centralGravity": 0.015,
                "springLength": 110,
                "springConstant": 0.08,
                "damping": 0.45,
                "avoidOverlap": 0.8,
            },
            "maxVelocity": 50,
            "minVelocity": 0.1,
            "solver": "forceAtlas2Based",
            "stabilization": {"enabled": True, "iterations": 150},
        },
        "interaction": {
            "hover": True,
            "tooltipDelay": 150,
            "hideEdgesOnDrag": False,
            "zoomView": True,
            "navigationButtons": True,
        },
    }

    net.set_options(json.dumps(physics_options))

    # Generate HTML string
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w", encoding="utf-8") as tmp:
        tmp_path = tmp.name

    try:
        net.save_graph(tmp_path)
        with open(tmp_path, "r", encoding="utf-8") as f:
            html_content = f.read()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # Inject responsive CSS wrapper to ensure clean embedding in Streamlit
    wrapper_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{
                margin: 0;
                padding: 0;
                background-color: #f8fafc;
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                overflow: hidden;
            }}
            #mynetwork {{
                border: 1px solid #e2e8f0;
                border-radius: 12px;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            }}
            .vis-navigation {{
                bottom: 15px !important;
                right: 15px !important;
            }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """

    return wrapper_html


def create_concept_graph(
    concepts: List[ExtractedConcept],
    similarity_threshold: float = 0.45,
    max_edges_per_node: int = 3,
) -> ConceptGraphResult:
    """
    Main interface: converts concepts to NetworkX, calculates metrics, and produces PyVis HTML.
    """
    G, edge_list, node_list = build_concept_network(
        concepts,
        similarity_threshold=similarity_threshold,
        max_edges_per_node=max_edges_per_node,
    )

    metrics = compute_graph_metrics(G)
    pyvis_html = generate_interactive_pyvis_html(G)

    return ConceptGraphResult(
        graph=G,
        metrics=metrics,
        pyvis_html=pyvis_html,
        edge_list=edge_list,
        node_list=node_list,
    )
