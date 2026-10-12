"""
modules/mindmap_generator.py
Hierarchical Concept Mind Map generator for PaperLens.
Transforms extracted scientific concepts and theory into a structured, multi-level Mind Map.
Supports:
1. Hierarchical NetworkX & PyVis interactive mind map visualization (Left-to-Right tree)
2. Interactive Mermaid.js mind map definition
3. Color-coded branch clusters categorized by scientific role
"""

from dataclasses import dataclass, field
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


# Thematic pillars representing the standard cognitive branches of a scientific research paper
MINDMAP_PILLARS = [
    {
        "id": "pillar_problem",
        "title": "🎯 Problem & Motivation",
        "theme": "Problem & Motivation",
        "query": "The core research question, computational bottleneck, limitations of existing methods, and primary objective.",
        "color": "#e11d48",  # Rose Red
        "font_color": "#ffffff",
    },
    {
        "id": "pillar_theory",
        "title": "📐 Theoretical Foundations",
        "theme": "Theoretical Foundations",
        "query": "The fundamental mathematical principles, theoretical framework, representation space, and core hypotheses.",
        "color": "#7c3aed",  # Royal Violet
        "font_color": "#ffffff",
    },
    {
        "id": "pillar_architecture",
        "title": "⚙️ Architecture & Methods",
        "theme": "Architecture & Methods",
        "query": "The proposed model architecture, algorithmic pipeline, technical components, and implementation design.",
        "color": "#2563eb",  # Royal Blue
        "font_color": "#ffffff",
    },
    {
        "id": "pillar_results",
        "title": "📊 Empirical Findings",
        "theme": "Empirical Findings",
        "query": "The benchmark evaluation, empirical experimental results, accuracy metrics, BLEU scores, and performance validation.",
        "color": "#059669",  # Emerald Green
        "font_color": "#ffffff",
    },
    {
        "id": "pillar_impact",
        "title": "💡 Significance & Impact",
        "theme": "Significance & Impact",
        "query": "The principal conclusions, theoretical contributions, practical applications, and future research directions.",
        "color": "#d97706",  # Amber
        "font_color": "#ffffff",
    },
]


@dataclass
class MindMapBranch:
    pillar_id: str
    pillar_title: str
    pillar_color: str
    concepts: List[ExtractedConcept] = field(default_factory=list)
    key_crux: List[str] = field(default_factory=list)


@dataclass
class MindMapResult:
    root_title: str
    branches: List[MindMapBranch]
    pyvis_html: str
    mermaid_markdown: str
    total_nodes: int
    total_branches: int


def classify_concepts_into_pillars(
    concepts: List[ExtractedConcept],
    model: Optional[Any] = None,
) -> Dict[str, List[ExtractedConcept]]:
    """
    Classifies extracted concepts into the 5 academic pillars using semantic similarity.
    """
    classified: Dict[str, List[ExtractedConcept]] = {p["id"]: [] for p in MINDMAP_PILLARS}

    if not concepts:
        return classified

    # Calculate embeddings for pillars
    pillar_queries = [p["query"] for p in MINDMAP_PILLARS]

    if model is not None:
        try:
            pillar_embs = model.encode(pillar_queries, normalize_embeddings=True)
            concept_embs = [c.embedding for c in concepts]

            # If some concepts do not have embeddings, encode them
            missing_indices = [i for i, emb in enumerate(concept_embs) if emb is None]
            if missing_indices:
                missing_texts = [concepts[i].text for i in missing_indices]
                computed = model.encode(missing_texts, normalize_embeddings=True)
                for idx, c_emb in zip(missing_indices, computed):
                    concept_embs[idx] = c_emb

            sim_matrix = cosine_similarity(np.array(concept_embs), pillar_embs)

            for c_idx, concept in enumerate(concepts):
                best_pillar_idx = int(np.argmax(sim_matrix[c_idx]))
                pillar_id = MINDMAP_PILLARS[best_pillar_idx]["id"]
                classified[pillar_id].append(concept)
            return classified
        except Exception as e:
            logger.warning(f"Embedding classification failed for mind map: {e}. Using round-robin heuristic.")

    # Fallback heuristic using keyword matches and distribution
    for idx, concept in enumerate(concepts):
        c_text = concept.text.lower()
        if any(w in c_text for w in ["problem", "bottleneck", "challenge", "cost", "limit", "complexity", "recurrent"]):
            classified["pillar_problem"].append(concept)
        elif any(w in c_text for w in ["attention", "dot", "scale", "space", "representation", "matrix", "vector", "layer", "theory"]):
            classified["pillar_theory"].append(concept)
        elif any(w in c_text for w in ["model", "network", "transformer", "encoder", "decoder", "architecture", "optimizer", "schedule"]):
            classified["pillar_architecture"].append(concept)
        elif any(w in c_text for w in ["result", "bleu", "score", "benchmark", "accuracy", "evaluation", "task", "training", "error"]):
            classified["pillar_results"].append(concept)
        else:
            # Distribute across pillars
            target_pillar = MINDMAP_PILLARS[idx % len(MINDMAP_PILLARS)]["id"]
            classified[target_pillar].append(concept)

    return classified


def build_hierarchical_mindmap_network(
    root_title: str,
    branches: List[MindMapBranch],
) -> nx.DiGraph:
    """
    Constructs a directed tree graph for hierarchical mind map visualization.
    Root -> 5 Pillars -> Concepts -> Context Snippets.
    """
    G = nx.DiGraph()

    # 1. Root Node (Center / Level 0)
    root_label = root_title if len(root_title) <= 36 else root_title[:33] + "..."
    root_tooltip = (
        f"<div style='font-family: Inter, sans-serif; padding: 8px; font-size: 13px;'>"
        f"<strong style='color: #0f172a; font-size: 15px;'>📄 Central Research Theme</strong><br/>"
        f"<span style='color: #475569;'>{root_title}</span>"
        f"</div>"
    )

    G.add_node(
        "ROOT",
        label=root_label,
        title=root_tooltip,
        level=0,
        shape="box",
        color={"background": "#0f172a", "border": "#334155", "highlight": {"background": "#1e293b", "border": "#4f46e5"}},
        font={"color": "#ffffff", "size": 16, "face": "Inter, sans-serif", "bold": True},
        margin=14,
        borderWidth=2,
        shadow=True,
    )

    # 2. Level 1: Thematic Pillars
    for b_idx, branch in enumerate(branches):
        p_id = branch.pillar_id
        pillar_title = branch.pillar_title

        p_tooltip = (
            f"<div style='font-family: Inter, sans-serif; padding: 6px; font-size: 13px;'>"
            f"<strong>{pillar_title}</strong><br/>"
            f"<span style='color: #64748b;'>Associated Concepts: {len(branch.concepts)}</span>"
            f"</div>"
        )

        G.add_node(
            p_id,
            label=pillar_title,
            title=p_tooltip,
            level=1,
            shape="box",
            color={
                "background": branch.pillar_color,
                "border": branch.pillar_color,
                "highlight": {"background": branch.pillar_color, "border": "#ffffff"},
            },
            font={"color": "#ffffff", "size": 14, "face": "Inter, sans-serif", "bold": True},
            margin=10,
            borderWidth=2,
            shadow=True,
        )

        # Edge from Root to Pillar
        G.add_edge(
            "ROOT",
            p_id,
            color={"color": branch.pillar_color, "highlight": branch.pillar_color},
            width=3,
            smooth={"type": "cubicBezier", "forceDirection": "horizontal"},
        )

        # 3. Level 2: Concepts belonging to this Pillar
        for c_idx, concept in enumerate(branch.concepts):
            c_node_id = f"concept_{p_id}_{c_idx}"
            relevance_pct = int(concept.relevance_score * 100)

            c_tooltip = (
                f"<div style='font-family: Inter, sans-serif; padding: 6px; font-size: 13px;'>"
                f"<strong style='color: #0f172a;'>{concept.text}</strong><br/>"
                f"<span style='color: #475569;'>Relevance: <b>{relevance_pct}%</b> | Freq: <b>{concept.frequency}</b></span><br/>"
                f"<span style='color: #64748b;'>Branch: {pillar_title}</span>"
                f"</div>"
            )

            G.add_node(
                c_node_id,
                label=f"{concept.text}\n({relevance_pct}%)",
                title=c_tooltip,
                level=2,
                shape="ellipse",
                color={
                    "background": "#ffffff",
                    "border": branch.pillar_color,
                    "highlight": {"background": "#f1f5f9", "border": branch.pillar_color},
                },
                font={"color": "#1e293b", "size": 12, "face": "Inter, sans-serif"},
                borderWidth=2,
                shadow=True,
            )

            G.add_edge(
                p_id,
                c_node_id,
                color={"color": "#cbd5e1", "highlight": branch.pillar_color},
                width=1.8,
                smooth={"type": "cubicBezier", "forceDirection": "horizontal"},
            )

            # 4. Level 3: Top Passage Crux (if available)
            if concept.top_passages:
                crux_text = concept.top_passages[0].text
                short_crux = crux_text[:60] + "..." if len(crux_text) > 60 else crux_text
                leaf_id = f"leaf_{c_node_id}"

                leaf_tooltip = (
                    f"<div style='font-family: Inter, sans-serif; padding: 6px; font-size: 12px; max-width: 320px;'>"
                    f"<strong>Theoretical Context (Page {concept.top_passages[0].page_number}):</strong><br/>"
                    f"<em>\"{crux_text}\"</em>"
                    f"</div>"
                )

                G.add_node(
                    leaf_id,
                    label=short_crux,
                    title=leaf_tooltip,
                    level=3,
                    shape="text",
                    font={"color": "#475569", "size": 11, "face": "Inter, sans-serif"},
                )

                G.add_edge(
                    c_node_id,
                    leaf_id,
                    color={"color": "#e2e8f0"},
                    width=1,
                    dashes=True,
                )

    return G


def generate_hierarchical_pyvis_html(G: nx.DiGraph, height: str = "600px") -> str:
    """
    Renders NetworkX DiGraph into a sleek Left-to-Right hierarchical Mind Map.
    """
    if G.number_of_nodes() == 0:
        return "<div style='text-align: center; color: #64748b; padding: 40px;'>No mind map data available.</div>"

    net = Network(
        height=height,
        width="100%",
        bgcolor="#f8fafc",
        font_color="#0f172a",
        directed=True,
        notebook=False,
    )

    net.from_nx(G)

    # Hierarchical Left-to-Right layout configuration
    hierarchical_options = {
        "layout": {
            "hierarchical": {
                "enabled": True,
                "direction": "LR",  # Left to Right flow
                "sortMethod": "directed",
                "levelSeparation": 220,
                "nodeSpacing": 90,
                "treeSpacing": 160,
                "blockShifting": True,
                "edgeMinimization": True,
                "parentCentralization": True,
            }
        },
        "physics": {
            "hierarchicalRepulsion": {
                "nodeDistance": 120,
                "centralGravity": 0.0,
                "springLength": 100,
                "springConstant": 0.01,
                "damping": 0.09,
            },
            "solver": "hierarchicalRepulsion",
            "stabilization": {"iterations": 100},
        },
        "interaction": {
            "hover": True,
            "tooltipDelay": 100,
            "zoomView": True,
            "dragView": True,
            "navigationButtons": True,
        },
    }

    net.set_options(json.dumps(hierarchical_options))

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w", encoding="utf-8") as tmp:
        tmp_path = tmp.name

    try:
        net.save_graph(tmp_path)
        with open(tmp_path, "r", encoding="utf-8") as f:
            html_content = f.read()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

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
                font-family: 'Inter', system-ui, sans-serif;
                overflow: hidden;
            }}
            #mynetwork {{
                border: 1px solid #e2e8f0;
                border-radius: 14px;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04);
            }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    return wrapper_html


def generate_mermaid_mindmap_markdown(
    root_title: str,
    branches: List[MindMapBranch],
) -> str:
    """
    Generates Mermaid.js mindmap syntax for native Streamlit or Markdown rendering.
    """
    clean_root = root_title.replace('"', '').replace('(', '').replace(')', '')
    if len(clean_root) > 40:
        clean_root = clean_root[:38] + "..."

    lines = ["mindmap", f'  root(("{clean_root}"))']

    for branch in branches:
        clean_pillar = branch.pillar_title.replace('"', '').replace('(', '').replace(')', '')
        lines.append(f'    ["{clean_pillar}"]')

        for c in branch.concepts[:5]:  # Show top 5 concepts per branch for clean syntax
            clean_concept = c.text.replace('"', '').replace('(', '').replace(')', '')
            lines.append(f'      ("{clean_concept}")')
            if c.top_passages:
                sample_p = c.top_passages[0].text[:45].replace('"', '').replace('(', '').replace(')', '') + "..."
                lines.append(f'        ["{sample_p}"]')

    return "\n".join(lines)


def generate_concept_mindmap(
    root_title: str,
    concepts: List[ExtractedConcept],
    model: Optional[Any] = None,
) -> MindMapResult:
    """
    Primary interface: Constructs hierarchical mind map branches, builds network,
    and produces interactive PyVis HTML and Mermaid syntax.
    """
    classified = classify_concepts_into_pillars(concepts, model=model)

    branches: List[MindMapBranch] = []
    total_nodes = 1  # root

    for p in MINDMAP_PILLARS:
        c_list = classified.get(p["id"], [])
        branch = MindMapBranch(
            pillar_id=p["id"],
            pillar_title=p["title"],
            pillar_color=p["color"],
            concepts=c_list,
        )
        branches.append(branch)
        total_nodes += 1 + len(c_list) + sum(1 for c in c_list if c.top_passages)

    G = build_hierarchical_mindmap_network(root_title, branches)
    pyvis_html = generate_hierarchical_pyvis_html(G)
    mermaid_md = generate_mermaid_mindmap_markdown(root_title, branches)

    return MindMapResult(
        root_title=root_title,
        branches=branches,
        pyvis_html=pyvis_html,
        mermaid_markdown=mermaid_md,
        total_nodes=total_nodes,
        total_branches=len(branches),
    )
