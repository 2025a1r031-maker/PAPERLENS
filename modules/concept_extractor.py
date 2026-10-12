"""
modules/concept_extractor.py
Academic concept and multi-word phrase extraction using semantic embedding relevance
(KeyBERT-style) and statistical phrase scoring with robust deduplication.
"""

from collections import Counter
from dataclasses import dataclass, field
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from modules.text_preprocessor import SentenceItem

logger = logging.getLogger(__name__)

# Standard English stop words
STANDARD_STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll",
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me",
    "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once",
    "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there",
    "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd",
    "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where",
    "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"
}

# Academic generic boilerplate that rarely represents a unique scientific concept
ACADEMIC_BOILERPLATE: Set[str] = {
    "paper", "papers", "study", "studies", "research", "article", "author", "authors",
    "approach", "approaches", "method", "methods", "methodology", "technique", "techniques",
    "system", "systems", "model", "models", "result", "results", "table", "figure", "figures",
    "section", "sections", "et", "al", "et al", "work", "works", "analysis", "data", "dataset",
    "proposed", "using", "used", "shows", "shown", "show", "based", "use", "different",
    "experiment", "experiments", "evaluation", "experimental", "conclusion", "abstract",
    "introduction", "future", "state", "art", "performance", "number", "level", "high",
    "low", "large", "small", "first", "second", "third", "main", "new", "various", "order",
    "good", "better", "best", "case", "cases", "type", "types", "way", "value", "values"
}

COMBINED_STOPWORDS = STANDARD_STOPWORDS | ACADEMIC_BOILERPLATE


@dataclass
class RelevantPassage:
    sentence_id: int
    page_number: int
    text: str
    match_score: float
    is_exact_match: bool


@dataclass
class ExtractedConcept:
    concept_id: int
    text: str
    relevance_score: float
    frequency: int
    n_gram_type: str
    top_passages: List[RelevantPassage] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None
    theoretical_crux: str = ""


def clean_phrase_candidate(phrase: str) -> Optional[str]:
    """
    Sanitizes and checks validity of an n-gram phrase candidate.
    """
    clean = phrase.strip().lower()
    clean = re.sub(r"[^\w\s-]", "", clean)
    clean = re.sub(r"\s+", " ", clean).strip()

    tokens = clean.split()
    if not tokens or len(tokens) > 4:
        return None

    # Length constraints
    if any(len(t) < 2 for t in tokens):
        return None

    # Must not start or end with a stopword
    if tokens[0] in COMBINED_STOPWORDS or tokens[-1] in COMBINED_STOPWORDS:
        return None

    # Entire phrase must not be in boilerplate
    if clean in COMBINED_STOPWORDS:
        return None

    # Must contain alphabetic characters
    if not all(re.search(r"[a-z]", t) for t in tokens):
        return None

    # Exclude pure numbers
    if all(t.isdigit() for t in tokens):
        return None

    return clean


def extract_ngram_candidates(
    sentences: List[SentenceItem],
    min_n: int = 1,
    max_n: int = 3,
) -> Tuple[Dict[str, int], Dict[str, Set[int]]]:
    """
    Extracts n-gram candidates across preprocessed academic sentences.
    Returns:
        candidate_counts: frequency of each candidate
        candidate_sentences: set of sentence_ids where candidate appeared
    """
    candidate_counts: Counter = Counter()
    candidate_sentences: Dict[str, Set[int]] = {}

    for s_item in sentences:
        tokens = re.findall(r"\b[A-Za-z][A-Za-z0-9\-]*\b", s_item.text.lower())
        n_tokens = len(tokens)

        for n in range(min_n, max_n + 1):
            for i in range(n_tokens - n + 1):
                raw_ngram = " ".join(tokens[i : i + n])
                cleaned = clean_phrase_candidate(raw_ngram)
                if cleaned:
                    candidate_counts[cleaned] += 1
                    if cleaned not in candidate_sentences:
                        candidate_sentences[cleaned] = set()
                    candidate_sentences[cleaned].add(s_item.sentence_id)

    return dict(candidate_counts), candidate_sentences


def find_top_passages_for_concept(
    concept: str,
    sentences: List[SentenceItem],
    max_passages: int = 4,
) -> List[RelevantPassage]:
    """
    Locates the most relevant sentences in the paper discussing this concept.
    """
    exact_matches: List[RelevantPassage] = []
    partial_matches: List[RelevantPassage] = []

    pattern = re.compile(rf"\b{re.escape(concept)}\b", re.IGNORECASE)

    for s in sentences:
        if pattern.search(s.text):
            exact_matches.append(
                RelevantPassage(
                    sentence_id=s.sentence_id,
                    page_number=s.page_number,
                    text=s.text,
                    match_score=1.0,
                    is_exact_match=True,
                )
            )
        else:
            # Word overlap match
            concept_words = set(concept.lower().split())
            sentence_words = set(s.text.lower().split())
            overlap = len(concept_words.intersection(sentence_words))
            if overlap >= max(1, len(concept_words) - 1):
                score = overlap / len(concept_words)
                if score >= 0.5:
                    partial_matches.append(
                        RelevantPassage(
                            sentence_id=s.sentence_id,
                            page_number=s.page_number,
                            text=s.text,
                            match_score=round(score * 0.8, 2),
                            is_exact_match=False,
                        )
                    )

    # Sort and combine
    combined = exact_matches + sorted(partial_matches, key=lambda p: p.match_score, reverse=True)
    return combined[:max_passages]


def extract_academic_concepts(
    sentences: List[SentenceItem],
    top_k: int = 15,
    model: Optional[Any] = None,
    diversity_threshold: float = 0.82,
) -> List[ExtractedConcept]:
    """
    Extracts, ranks, and deduplicates key academic concepts from document sentences.

    Args:
        sentences: Preprocessed sentences from the document.
        top_k: Number of concepts to return (between 5 and 25).
        model: Optional SentenceTransformer model for KeyBERT-style semantic embedding.
        diversity_threshold: Similarity cutoff for deduplicating synonymous concepts.

    Returns:
        List of ExtractedConcept objects sorted by relevance.
    """
    top_k = max(5, min(25, top_k))

    if not sentences:
        return []

    # 1. Generate n-gram candidates
    candidate_counts, candidate_sentences = extract_ngram_candidates(sentences, min_n=1, max_n=3)

    if not candidate_counts:
        return []

    # Filter out candidates with frequency 1 if unigram unless vocabulary is very small
    filtered_candidates: Dict[str, int] = {}
    total_candidates = len(candidate_counts)

    for phrase, count in candidate_counts.items():
        words = phrase.split()
        if len(words) == 1:
            # Single words need at least 2 occurrences unless document is very small
            if count >= 2 or total_candidates < 30:
                filtered_candidates[phrase] = count
        else:
            # Bigrams & trigrams are informative even with count 1
            filtered_candidates[phrase] = count

    if not filtered_candidates:
        filtered_candidates = candidate_counts

    candidate_list = list(filtered_candidates.keys())
    counts_array = np.array([filtered_candidates[p] for p in candidate_list], dtype=float)
    # Normalize frequency log-scale
    log_freq = np.log1p(counts_array)
    max_freq = np.max(log_freq) if np.max(log_freq) > 0 else 1.0
    norm_freq = log_freq / max_freq

    # 2. Semantic Embedding Relevance Ranking (KeyBERT approach)
    all_texts = [s.text for s in sentences]
    full_text_sample = " ".join(all_texts[:40])  # Sample up to first 40 sentences for document topic representation

    if model is not None:
        try:
            doc_emb = model.encode([full_text_sample], normalize_embeddings=True)
            cand_embs = model.encode(candidate_list, show_progress_bar=False, normalize_embeddings=True)
            sem_sims = cosine_similarity(cand_embs, doc_emb).flatten()
        except Exception as e:
            logger.warning(f"Embedding encoding failed for concepts: {e}. Using frequency ranking.")
            cand_embs = None
            sem_sims = norm_freq.copy()
    else:
        cand_embs = None
        sem_sims = norm_freq.copy()

    # N-gram bonus: multi-word phrases (e.g. "attention mechanism") carry more specific academic value
    ngram_bonuses = np.array([0.08 if len(p.split()) > 1 else 0.0 for p in candidate_list])

    # Composite score
    composite_scores = (0.60 * sem_sims) + (0.32 * norm_freq) + ngram_bonuses
    ranked_indices = np.argsort(composite_scores)[::-1]

    # 3. Deduplication and Selection
    selected_concepts: List[ExtractedConcept] = []
    selected_texts: List[str] = []
    selected_embs: List[np.ndarray] = []

    concept_id_counter = 1

    for idx in ranked_indices:
        cand_phrase = candidate_list[idx]
        cand_score = float(composite_scores[idx])
        cand_emb = cand_embs[idx] if cand_embs is not None else None

        # Check redundancy against already selected concepts
        is_duplicate = False

        for sel_text, sel_emb in zip(selected_texts, selected_embs):
            # Text substring / token overlap check
            if cand_phrase in sel_text or sel_text in cand_phrase:
                is_duplicate = True
                break

            # Embedding similarity check
            if cand_emb is not None and sel_emb is not None:
                sim = cosine_similarity(cand_emb.reshape(1, -1), sel_emb.reshape(1, -1))[0][0]
                if sim > diversity_threshold:
                    is_duplicate = True
                    break

        if not is_duplicate:
            n_words = len(cand_phrase.split())
            ngram_type = "Unigram" if n_words == 1 else ("Bigram" if n_words == 2 else "Trigram")

            # Extract matching sentences for passage viewer
            passages = find_top_passages_for_concept(cand_phrase, sentences)

            # Extract the most definitive contextual theoretical crux
            if passages:
                best_crux = passages[0].text
                for p in passages:
                    p_lower = p.text.lower()
                    if any(kw in p_lower for kw in ["is ", "defined as", "consists of", "propose", "mechanism", "technique", "algorithm", "architecture", "framework", "formulate"]):
                        best_crux = p.text
                        break
                crux_summary = best_crux
            else:
                crux_summary = f"Core conceptual entity in the paper's representation space."

            concept_obj = ExtractedConcept(
                concept_id=concept_id_counter,
                text=cand_phrase.title(),
                relevance_score=float(round(min(1.0, max(0.1, cand_score)), 3)),
                frequency=filtered_candidates[cand_phrase],
                n_gram_type=ngram_type,
                top_passages=passages,
                embedding=cand_emb,
                theoretical_crux=crux_summary,
            )
            selected_concepts.append(concept_obj)
            selected_texts.append(cand_phrase)
            if cand_emb is not None:
                selected_embs.append(cand_emb)

            concept_id_counter += 1

            if len(selected_concepts) >= top_k:
                break

    return selected_concepts
