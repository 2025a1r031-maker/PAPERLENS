"""
modules/summarizer.py
Extractive academic research paper summarization using Sentence Transformers
and semantic centrality ranking.
Generates:
1. Executive 3-bullet summary (Problem, Methodology, Conclusion)
2. Comprehensive multi-section Theory Breakdown in Crux form:
   - Research Gap & Problem Statement
   - Theoretical Foundations & Principles
   - Architecture & Methodological Pipeline
   - Quantitative Benchmarks & Experimental Findings
   - Significance, Impact & Future Horizons
"""

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from modules.text_preprocessor import SentenceItem

logger = logging.getLogger(__name__)

# Global model cache to avoid repeated disk reads / downloads
_GLOBAL_MODEL_CACHE: Dict[str, Any] = {}


@dataclass
class SummaryBullet:
    category: str
    icon: str
    text: str
    page_number: int
    centrality_score: float
    section_hint: str


@dataclass
class TheoryCruxSection:
    title: str
    icon: str
    description: str
    crux_points: List[SummaryBullet] = field(default_factory=list)


@dataclass
class DetailedTheoryAnalysis:
    sections: List[TheoryCruxSection] = field(default_factory=list)
    total_crux_points: int = 0


@dataclass
class SummaryResult:
    is_success: bool
    bullets: List[SummaryBullet]
    summary_type: str = "Extractive Semantic Centrality (all-MiniLM-L6-v2)"
    error_message: Optional[str] = None
    document_coverage_pct: float = 0.0
    theory_crux: Optional[DetailedTheoryAnalysis] = None


def load_sentence_transformer_model(model_name: str = "all-MiniLM-L6-v2"):
    """
    Loads and caches the SentenceTransformer model.
    Falls back gracefully if torch or transformers are still loading.
    """
    if model_name in _GLOBAL_MODEL_CACHE:
        return _GLOBAL_MODEL_CACHE[model_name]

    try:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading SentenceTransformer model: {model_name}")
        model = SentenceTransformer(model_name)
        _GLOBAL_MODEL_CACHE[model_name] = model
        return model
    except Exception as e:
        logger.warning(f"Could not load SentenceTransformer ({str(e)}). Using fallback.")
        return None


# Primary 3-Bullet Probes
PROBE_QUERIES = {
    "problem": (
        "The primary research problem, objective, motivation, and research question addressed by this study.",
        "Problem & Objective",
        "🎯",
        ["Abstract", "Introduction"],
    ),
    "method": (
        "The proposed methodology, experimental framework, algorithm, model architecture, or technique applied.",
        "Methodology & Approach",
        "⚙️",
        ["Methodology", "Proposed Approach", "Methods", "System Design"],
    ),
    "contribution": (
        "The principal findings, experimental results, key contributions, and conclusion or implications.",
        "Key Contribution & Conclusion",
        "🏆",
        ["Conclusion", "Results", "Discussion", "Findings"],
    ),
}

# In-depth Crux Dimensions for Detailed Theory Breakdown
THEORY_CRUX_DIMENSIONS = [
    {
        "id": "gap",
        "title": "🎯 Research Gap & Problem Formulation",
        "icon": "🎯",
        "desc": "The central question, literature bottleneck, and primary motivation.",
        "query": "The core research question, fundamental problem, computational bottleneck, and limitations of existing methods.",
        "favored_sections": ["Abstract", "Introduction", "Related Work"],
        "pos_range": (0.0, 0.35),
    },
    {
        "id": "theory",
        "title": "📐 Theoretical Foundations & Core Mechanisms",
        "icon": "📐",
        "desc": "Mathematical foundations, representation space, and conceptual mechanics.",
        "query": "The core theoretical framework, mathematical formulation, attention mechanism, error threshold, and representation theory.",
        "favored_sections": ["Methodology", "Introduction", "General"],
        "pos_range": (0.10, 0.60),
    },
    {
        "id": "architecture",
        "title": "⚙️ Architectural Pipeline & Algorithmic Design",
        "icon": "⚙️",
        "desc": "Concrete model layers, implementation pipeline, and technical steps.",
        "query": "The proposed network architecture, encoder decoder layers, algorithm steps, stabilizer measurements, optimizer, and parameters.",
        "favored_sections": ["Methodology", "Methods", "System Design"],
        "pos_range": (0.25, 0.75),
    },
    {
        "id": "empirical",
        "title": "📊 Empirical Evidence & Benchmark Performance",
        "icon": "📊",
        "desc": "Quantitative results, comparative baselines, and benchmark gains.",
        "query": "The experimental evaluation results, BLEU score, accuracy benchmarks, chemical accuracy, runtime speedup, and quantitative improvements.",
        "favored_sections": ["Results", "Discussion"],
        "pos_range": (0.45, 0.90),
    },
    {
        "id": "impact",
        "title": "💡 Theoretical Contributions & Strategic Horizons",
        "icon": "💡",
        "desc": "Broader implications, key takeaways, and future research directions.",
        "query": "The principal conclusions, theoretical significance, practical implications, limitations acknowledged, and future research directions.",
        "favored_sections": ["Conclusion", "Discussion"],
        "pos_range": (0.60, 1.0),
    },
]


def extract_crux_points_for_dimension(
    dim_config: Dict[str, Any],
    sentence_items: List[SentenceItem],
    sentence_embeddings: np.ndarray,
    doc_centroid: np.ndarray,
    model: Optional[Any],
    tfidf_vectorizer: Optional[Any],
    selected_indices_global: set,
    num_points: int = 3,
    diversity_threshold: float = 0.68,
) -> List[SummaryBullet]:
    """
    Selects top diverse crux sentences for a specific theoretical dimension.
    """
    total_sentences = len(sentence_items)
    query_text = dim_config["query"]
    favored = dim_config["favored_sections"]
    min_pos, max_pos = dim_config["pos_range"]

    # Compute query embedding
    if model is not None and tfidf_vectorizer is None:
        try:
            query_emb = model.encode([query_text], normalize_embeddings=True)
        except Exception:
            query_emb = np.ones((1, sentence_embeddings.shape[1]))
    elif tfidf_vectorizer is not None:
        query_emb = tfidf_vectorizer.transform([query_text]).toarray()
    else:
        query_emb = np.ones((1, sentence_embeddings.shape[1]))

    # Cosine similarities
    query_sims = cosine_similarity(sentence_embeddings, query_emb).flatten()
    doc_sims = cosine_similarity(sentence_embeddings, doc_centroid).flatten()

    combined_scores = np.zeros(total_sentences)

    for i, item in enumerate(sentence_items):
        rel_pos = i / max(total_sentences, 1)

        # Section matching bonus
        sec_bonus = 0.15 if item.section_hint in favored else 0.0

        # Position preference bonus
        pos_bonus = 0.12 if (min_pos <= rel_pos <= max_pos) else 0.0

        # Substantive sentence length bonus (longer sentences convey more theory)
        length_bonus = min(0.08, len(item.text.split()) / 250.0)

        # Heavily penalize already selected globally to maintain novelty across sections
        if i in selected_indices_global:
            penalty = -0.40
        else:
            penalty = 0.0

        combined_scores[i] = (
            0.50 * query_sims[i]
            + 0.25 * doc_sims[i]
            + sec_bonus
            + pos_bonus
            + length_bonus
            + penalty
        )

    ranked_indices = np.argsort(combined_scores)[::-1]

    chosen_bullets: List[SummaryBullet] = []
    chosen_local_indices: List[int] = []

    for cand_idx in ranked_indices:
        if len(chosen_bullets) >= num_points:
            break

        # Check local diversity within this section
        is_redundant = False
        for loc_idx in chosen_local_indices:
            sim = cosine_similarity(
                sentence_embeddings[cand_idx].reshape(1, -1),
                sentence_embeddings[loc_idx].reshape(1, -1),
            )[0][0]
            if sim > diversity_threshold:
                is_redundant = True
                break

        if not is_redundant:
            chosen_local_indices.append(cand_idx)
            selected_indices_global.add(cand_idx)
            item = sentence_items[cand_idx]

            chosen_bullets.append(
                SummaryBullet(
                    category=dim_config["title"],
                    icon=dim_config["icon"],
                    text=item.text,
                    page_number=item.page_number,
                    centrality_score=float(round(combined_scores[cand_idx], 3)),
                    section_hint=item.section_hint,
                )
            )

    return chosen_bullets


def generate_research_summary(
    sentence_items: List[SentenceItem],
    model: Optional[Any] = None,
    diversity_threshold: float = 0.72,
) -> SummaryResult:
    """
    Performs multi-criteria extractive semantic summarization over candidate sentences.
    Generates both the core 3-bullet executive summary and comprehensive theory crux breakdown.
    """
    if not sentence_items or len(sentence_items) < 3:
        return SummaryResult(
            is_success=False,
            bullets=[],
            error_message=(
                "Insufficient text content: The document does not contain enough informative "
                "sentences (minimum 3 required) to construct a reliable academic summary. "
                "Please verify if the PDF contains readable text."
            ),
        )

    candidate_texts = [item.text for item in sentence_items]
    all_probe_texts = [q[0] for q in PROBE_QUERIES.values()] + [d["query"] for d in THEORY_CRUX_DIMENSIONS]
    tfidf_vectorizer = None

    if model is None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        tfidf_vectorizer = TfidfVectorizer(max_features=256, stop_words="english")
        try:
            tfidf_vectorizer.fit(candidate_texts + all_probe_texts)
            sentence_embeddings = tfidf_vectorizer.transform(candidate_texts).toarray()
        except ValueError:
            sentence_embeddings = np.ones((len(candidate_texts), 1))
    else:
        try:
            sentence_embeddings = model.encode(candidate_texts, show_progress_bar=False, normalize_embeddings=True)
        except Exception as e:
            logger.warning(f"SentenceTransformer encoding failed: {e}. Reverting to TF-IDF.")
            from sklearn.feature_extraction.text import TfidfVectorizer
            tfidf_vectorizer = TfidfVectorizer(max_features=256, stop_words="english")
            tfidf_vectorizer.fit(candidate_texts + all_probe_texts)
            sentence_embeddings = tfidf_vectorizer.transform(candidate_texts).toarray()

    # Document Centroid Embedding
    doc_centroid = np.mean(sentence_embeddings, axis=0, keepdims=True)
    if np.linalg.norm(doc_centroid) > 0:
        doc_centroid = doc_centroid / np.linalg.norm(doc_centroid)
    global_centrality = cosine_similarity(sentence_embeddings, doc_centroid).flatten()

    # Part A: Core 3-Bullet Executive Summary
    selected_bullets: List[SummaryBullet] = []
    selected_indices: set = set()
    total_sentences = len(sentence_items)

    for role_key in ["problem", "method", "contribution"]:
        query_text, category_name, icon, favored_sections = PROBE_QUERIES[role_key]

        if model is not None and tfidf_vectorizer is None:
            try:
                query_emb = model.encode([query_text], normalize_embeddings=True)
            except Exception:
                query_emb = np.ones((1, sentence_embeddings.shape[1]))
        elif tfidf_vectorizer is not None:
            query_emb = tfidf_vectorizer.transform([query_text]).toarray()
        else:
            query_emb = np.ones((1, sentence_embeddings.shape[1]))

        query_sims = cosine_similarity(sentence_embeddings, query_emb).flatten()
        combined_scores = np.zeros(total_sentences)

        for i, item in enumerate(sentence_items):
            if i in selected_indices:
                combined_scores[i] = -999.0
                continue

            section_bonus = 0.15 if item.section_hint in favored_sections else 0.0
            rel_pos = i / max(total_sentences, 1)
            pos_bonus = 0.0
            if role_key == "problem":
                pos_bonus = max(0.0, 0.15 * (1.0 - (rel_pos * 3.0)))
            elif role_key == "method":
                pos_bonus = 0.12 if 0.15 <= rel_pos <= 0.80 else 0.0
            elif role_key == "contribution":
                pos_bonus = max(0.0, 0.15 * (rel_pos - 0.4))

            combined_scores[i] = (
                0.50 * query_sims[i]
                + 0.30 * global_centrality[i]
                + section_bonus
                + pos_bonus
            )

        ranked_candidate_indices = np.argsort(combined_scores)[::-1]
        chosen_idx = None

        for cand_idx in ranked_candidate_indices:
            if combined_scores[cand_idx] <= -990.0:
                break

            is_redundant = False
            for sel_idx in selected_indices:
                sim_between = cosine_similarity(
                    sentence_embeddings[cand_idx].reshape(1, -1),
                    sentence_embeddings[sel_idx].reshape(1, -1),
                )[0][0]
                if sim_between > diversity_threshold:
                    is_redundant = True
                    break

            if not is_redundant:
                chosen_idx = cand_idx
                break

        if chosen_idx is None:
            for cand_idx in ranked_candidate_indices:
                if cand_idx not in selected_indices:
                    chosen_idx = cand_idx
                    break

        if chosen_idx is not None:
            selected_indices.add(chosen_idx)
            chosen_item = sentence_items[chosen_idx]
            selected_bullets.append(
                SummaryBullet(
                    category=category_name,
                    icon=icon,
                    text=chosen_item.text,
                    page_number=chosen_item.page_number,
                    centrality_score=float(round(combined_scores[chosen_idx], 3)),
                    section_hint=chosen_item.section_hint,
                )
            )

    # Part B: Multi-Section Detailed Theory Crux Breakdown
    theory_sections: List[TheoryCruxSection] = []
    crux_points_per_dim = 3 if total_sentences >= 15 else 2
    total_crux_count = 0

    for dim in THEORY_CRUX_DIMENSIONS:
        points = extract_crux_points_for_dimension(
            dim_config=dim,
            sentence_items=sentence_items,
            sentence_embeddings=sentence_embeddings,
            doc_centroid=doc_centroid,
            model=model,
            tfidf_vectorizer=tfidf_vectorizer,
            selected_indices_global=selected_indices,
            num_points=crux_points_per_dim,
            diversity_threshold=diversity_threshold,
        )
        total_crux_count += len(points)
        theory_sections.append(
            TheoryCruxSection(
                title=dim["title"],
                icon=dim["icon"],
                description=dim["desc"],
                crux_points=points,
            )
        )

    theory_analysis = DetailedTheoryAnalysis(
        sections=theory_sections,
        total_crux_points=total_crux_count,
    )

    if len(selected_bullets) == 3:
        return SummaryResult(
            is_success=True,
            bullets=selected_bullets,
            summary_type="Extractive Semantic Centrality (Sentence-Transformers)",
            document_coverage_pct=round(((3 + total_crux_count) / total_sentences) * 100, 1),
            theory_crux=theory_analysis,
        )

    return SummaryResult(
        is_success=False,
        bullets=selected_bullets,
        error_message="Could not extract sufficient distinct sentences from the document.",
        theory_crux=theory_analysis,
    )
