"""
tests/test_pdf_processor.py
Unit and integration tests for PDF validation, extraction, sanitization,
and scanned document detection.
"""

import io
try:
    import pymupdf as fitz
except ImportError:
    import fitz
import pytest
from modules.pdf_processor import (
    validate_pdf_file,
    clean_academic_text,
    extract_pdf_document,
    PDFDocumentResult,
)


def create_in_memory_pdf(text_pages: list[str]) -> bytes:
    """Helper to generate an in-memory valid PDF for testing."""
    doc = fitz.open()
    for text in text_pages:
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(50, 50, 500, 700), text, fontsize=11)
    pdf_bytes = doc.write()
    doc.close()
    return pdf_bytes


def test_validate_pdf_valid():
    sample_pdf = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF"
    is_valid, err = validate_pdf_file(sample_pdf, "research_paper.pdf")
    assert is_valid is True
    assert err is None


def test_validate_pdf_invalid_extension():
    sample_pdf = b"%PDF-1.4\n..."
    is_valid, err = validate_pdf_file(sample_pdf, "data.txt")
    assert is_valid is False
    assert "Invalid file extension" in err


def test_validate_pdf_empty():
    is_valid, err = validate_pdf_file(b"", "empty.pdf")
    assert is_valid is False
    assert "empty" in err.lower()


def test_validate_pdf_corrupt_header():
    is_valid, err = validate_pdf_file(b"NOT_A_PDF_FILE_CONTENT", "corrupt.pdf")
    assert is_valid is False
    assert "missing %PDF-" in err


def test_clean_academic_text():
    raw = "The deep learn-\ning methodo-\nlogy was tested.   It achieved    98.5% accuracy."
    cleaned = clean_academic_text(raw)
    assert "deep learning" in cleaned
    assert "methodology" in cleaned
    assert "   " not in cleaned
    assert "98.5% accuracy." in cleaned


def test_extract_pdf_document_success():
    page1 = (
        "Abstract\nRecent advances in deep learning have transformed natural language processing. "
        "We propose a novel attention-based framework that eliminates recurrent connections."
    )
    page2 = (
        "Methodology\nOur architecture consists of an encoder-decoder stack with multi-head attention. "
        "The model was trained on standard machine translation benchmarks."
    )
    pdf_bytes = create_in_memory_pdf([page1, page2])

    result = extract_pdf_document(pdf_bytes, filename="transformer_test.pdf")

    assert result.is_valid is True
    assert result.page_count == 2
    assert result.total_word_count > 20
    assert result.is_scanned is False
    assert len(result.pages) == 2
    assert "attention-based framework" in result.cleaned_full_text
    assert result.error_message is None


def test_extract_scanned_pdf_detection():
    # Empty page simulating scanned image with no text layer
    pdf_bytes = create_in_memory_pdf(["", ""])
    result = extract_pdf_document(pdf_bytes, filename="scanned_paper.pdf")

    assert result.is_valid is True
    assert result.page_count == 2
    assert result.total_word_count == 0
    assert result.is_scanned is True
    assert "Scanned Document Detected" in result.error_message
    assert "OCR" in result.error_message


def test_extract_corrupted_pdf_graceful_handling():
    corrupted_bytes = b"%PDF-1.4\ncorrupted content that cannot be parsed by fitz"
    result = extract_pdf_document(corrupted_bytes, filename="bad.pdf")

    assert result.is_valid is False
    assert result.error_message is not None
