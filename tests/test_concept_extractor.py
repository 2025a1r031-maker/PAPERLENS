"""
tests/test_concept_extractor.py
Unit tests for academic concept extraction, n-gram parsing, deduplication,
and passage context mapping.
"""

import pytest
from modules.text_preprocessor import SentenceItem
from modules.concept_extractor import (
    extract_ngram_candidates,
    extract_academic_concepts,
    find_top_passages_for_concept,
    ExtractedConcept,
)
from tests.test_summarizer import create_sample_sentences


def test_extract_ngram_candidates():
    sentences = create_sample_sentences()
    counts, sentence_map = extract_ngram_candidates(sentences, min_n=1, max_n=3)

    assert len(counts) > 0
    # Stop words like 'the', 'in', 'is' should NOT be extracted
    assert "the" not in counts
    assert "in" not in counts
    assert "is" not in counts


def test_extract_academic_concepts_structure():
    sentences = create_sample_sentences()
    concepts = extract_academic_concepts(sentences, top_k=8, model=None)

    assert len(concepts) > 0
    assert len(concepts) <= 8

    for c in concepts:
        assert isinstance(c, ExtractedConcept)
        assert c.concept_id >= 1
        assert len(c.text) > 0
        assert c.relevance_score > 0
        assert c.frequency >= 1
        assert c.n_gram_type in ["Unigram", "Bigram", "Trigram"]


def test_concept_passage_lookup():
    sentences = create_sample_sentences()
    passages = find_top_passages_for_concept("attention", sentences)

    assert len(passages) > 0
    assert any("attention" in p.text.lower() for p in passages)
    assert passages[0].page_number in [1, 2, 3]


def test_concept_empty_input():
    concepts = extract_academic_concepts([], top_k=10, model=None)
    assert concepts == []
