"""
tests/test_summarizer.py
Unit tests for extractive research paper summarization, semantic centrality,
and diversity constraints.
"""

import pytest
from modules.text_preprocessor import SentenceItem
from modules.summarizer import (
    generate_research_summary,
    SummaryResult,
    SummaryBullet,
)


def create_sample_sentences() -> list[SentenceItem]:
    raw_texts = [
        ("The primary challenge in modern machine learning is the extreme computational cost of training dense models.", "Introduction", 1),
        ("Current recurrent architectures suffer from sequential execution bottlenecks that prevent parallel computation.", "Introduction", 1),
        ("We formulate the problem of high-throughput sequence transduction without relying on recurrence.", "Introduction", 1),
        ("We propose a novel self-attention transformer architecture that connects all positions via attention weights.", "Methodology", 2),
        ("Our method applies scaled dot-product multi-head attention mechanisms across eight parallel representation subspaces.", "Methodology", 2),
        ("The model optimization utilizes Adam with warm-up learning rate schedules and dropout regularization.", "Methodology", 2),
        ("Empirical evaluation demonstrates that our model achieves 28.4 BLEU on English-to-German translation tasks.", "Results", 3),
        ("The proposed architecture achieves superior translation quality while requiring a fraction of previous training time.", "Results", 3),
        ("In conclusion, the attention-based approach establishes a new state of the art in machine translation.", "Conclusion", 3),
    ]

    items = []
    for idx, (txt, sec, page) in enumerate(raw_texts, 1):
        items.append(
            SentenceItem(
                sentence_id=idx,
                text=txt,
                page_number=page,
                word_count=len(txt.split()),
                char_count=len(txt),
                section_hint=sec,
            )
        )
    return items


def test_summarizer_produces_three_bullets():
    sentences = create_sample_sentences()
    result = generate_research_summary(sentences, model=None)

    assert result.is_success is True
    assert len(result.bullets) == 3
    assert result.error_message is None

    categories = [b.category for b in result.bullets]
    assert "Problem & Objective" in categories
    assert "Methodology & Approach" in categories
    assert "Key Contribution & Conclusion" in categories


def test_summarizer_bullet_metadata():
    sentences = create_sample_sentences()
    result = generate_research_summary(sentences, model=None)

    for bullet in result.bullets:
        assert isinstance(bullet, SummaryBullet)
        assert len(bullet.text) > 20
        assert bullet.page_number in [1, 2, 3]
        assert bullet.centrality_score != 0.0
        assert bullet.section_hint in ["Introduction", "Methodology", "Results", "Conclusion"]


def test_summarizer_produces_detailed_theory_crux():
    sentences = create_sample_sentences()
    result = generate_research_summary(sentences, model=None)

    assert result.is_success is True
    assert result.theory_crux is not None
    assert len(result.theory_crux.sections) == 5
    assert result.theory_crux.total_crux_points > 0

    section_titles = [s.title for s in result.theory_crux.sections]
    assert any("Research Gap" in t for t in section_titles)
    assert any("Theoretical Foundations" in t for t in section_titles)
    assert any("Architectural Pipeline" in t for t in section_titles)


def test_summarizer_insufficient_text():
    # Only 2 sentences
    sparse_sentences = [
        SentenceItem(1, "Deep learning is popular.", 1, 4, 25, "Abstract"),
        SentenceItem(2, "It works well.", 1, 3, 14, "Abstract"),
    ]
    result = generate_research_summary(sparse_sentences, model=None)

    assert result.is_success is False
    assert len(result.bullets) == 0
    assert "Insufficient text content" in result.error_message
