"""
modules/pdf_processor.py
Robust extraction and validation of academic research papers in PDF format using PyMuPDF (fitz).
Handles text sanitization, page-level tracking, and scanned-document detection.
"""

from dataclasses import dataclass, field
import io
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union
try:
    import pymupdf as fitz
except ImportError:
    import fitz


@dataclass
class PageData:
    page_number: int  # 1-indexed
    raw_text: str
    clean_text: str
    word_count: int
    char_count: int


@dataclass
class PDFDocumentResult:
    is_valid: bool
    filename: str
    page_count: int = 0
    total_word_count: int = 0
    total_char_count: int = 0
    raw_full_text: str = ""
    cleaned_full_text: str = ""
    pages: List[PageData] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_scanned: bool = False
    estimated_reading_time_mins: float = 0.0
    error_message: Optional[str] = None


def validate_pdf_file(
    file_bytes: bytes,
    filename: str,
    max_size_mb: float = 50.0,
) -> Tuple[bool, Optional[str]]:
    """
    Validates PDF file size, extension, and minimal signature.
    """
    if not filename.lower().endswith(".pdf"):
        return False, f"Invalid file extension for '{filename}'. Only .pdf files are accepted."

    file_size_mb = len(file_bytes) / (1024 * 1024)
    if file_size_mb == 0:
        return False, "Uploaded file is empty (0 bytes)."

    if file_size_mb > max_size_mb:
        return False, (
            f"File size ({file_size_mb:.1f} MB) exceeds maximum allowed limit of {max_size_mb:.1f} MB."
        )

    # Check for PDF magic header (%PDF-)
    if not file_bytes.startswith(b"%PDF-"):
        return False, "File does not appear to be a valid PDF document (missing %PDF- header)."

    return True, None


def clean_academic_text(text: str) -> str:
    """
    Sanitizes extracted academic PDF text:
    - Normalizes line breaks and whitespace
    - De-hyphenates words broken across lines (e.g., 'trans-\nformer' -> 'transformer')
    - Cleans isolated page number artefacts and orphaned bullet points
    - Preserves sentence boundaries and mathematical notations
    """
    if not text:
        return ""

    # Normalize CRLF / CR to LF
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Fix hyphenated words broken across line breaks (e.g., "methodo-\nlogy" -> "methodology")
    text = re.sub(r"(\b[A-Za-z]+)-\n([A-Za-z]+\b)", r"\1\2", text)

    # Replace lone form feed characters
    text = text.replace("\x0c", "\n")

    # Clean excessive non-breaking spaces and tabs
    text = text.replace("\u00a0", " ").replace("\t", " ")

    # Collapse multiple consecutive spaces (keeping single spaces)
    text = re.sub(r"[ ]{2,}", " ", text)

    # Collapse excessive newlines (more than 2 -> 2)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip leading and trailing whitespace
    return text.strip()


def extract_pdf_document(
    file_input: Union[bytes, str],
    filename: str = "document.pdf",
    max_size_mb: float = 50.0,
) -> PDFDocumentResult:
    """
    Parses and extracts structured academic text from a PDF file using PyMuPDF.

    Args:
        file_input: Either bytes of the PDF file or a filesystem path string.
        filename: Human-readable name of the file for reporting.
        max_size_mb: Maximum allowed file size in megabytes.

    Returns:
        PDFDocumentResult containing extraction metrics, pages, and sanitized text.
    """
    # 1. Byte normalization and pre-validation
    if isinstance(file_input, str):
        if not os.path.exists(file_input):
            return PDFDocumentResult(
                is_valid=False,
                filename=filename,
                error_message=f"File not found on disk: {file_input}",
            )
        try:
            with open(file_input, "rb") as f:
                raw_bytes = f.read()
            if filename == "document.pdf":
                filename = os.path.basename(file_input)
        except Exception as e:
            return PDFDocumentResult(
                is_valid=False,
                filename=filename,
                error_message=f"Could not read file from disk: {str(e)}",
            )
    else:
        raw_bytes = file_input

    # Validate header and size
    is_valid_file, validation_error = validate_pdf_file(raw_bytes, filename, max_size_mb)
    if not is_valid_file:
        return PDFDocumentResult(
            is_valid=False,
            filename=filename,
            error_message=validation_error,
        )

    # 2. PyMuPDF Document Opening
    try:
        doc = fitz.open(stream=raw_bytes, filetype="pdf")
    except Exception as e:
        return PDFDocumentResult(
            is_valid=False,
            filename=filename,
            error_message=f"PyMuPDF failed to parse PDF document. File may be corrupted. Error: {str(e)}",
        )

    try:
        if doc.is_encrypted:
            return PDFDocumentResult(
                is_valid=False,
                filename=filename,
                error_message="The uploaded PDF is password-protected or encrypted. Please upload an unprotected document.",
            )

        page_count = len(doc)
        if page_count == 0:
            return PDFDocumentResult(
                is_valid=False,
                filename=filename,
                error_message="The uploaded PDF has 0 pages.",
            )

        # 3. Extract Document Metadata
        metadata = {}
        try:
            raw_meta = doc.metadata or {}
            metadata = {
                "title": (raw_meta.get("title") or "").strip() or filename,
                "author": (raw_meta.get("author") or "").strip() or "Not specified",
                "subject": (raw_meta.get("subject") or "").strip() or "Academic Research",
                "keywords": (raw_meta.get("keywords") or "").strip(),
                "creator": (raw_meta.get("creator") or "").strip(),
                "producer": (raw_meta.get("producer") or "").strip(),
                "creation_date": (raw_meta.get("creationDate") or "").strip(),
            }
        except Exception:
            metadata = {"title": filename, "author": "Not specified"}

        # 4. Page-by-page extraction
        pages_data: List[PageData] = []
        raw_texts: List[str] = []
        cleaned_texts: List[str] = []
        total_words = 0
        total_chars = 0

        for page_idx in range(page_count):
            try:
                page = doc[page_idx]
                page_raw_text = page.get_text("text") or ""
            except Exception:
                page_raw_text = ""

            page_clean = clean_academic_text(page_raw_text)
            words = page_clean.split()
            w_count = len(words)
            c_count = len(page_clean)

            total_words += w_count
            total_chars += c_count

            raw_texts.append(page_raw_text)
            cleaned_texts.append(page_clean)

            pages_data.append(
                PageData(
                    page_number=page_idx + 1,
                    raw_text=page_raw_text,
                    clean_text=page_clean,
                    word_count=w_count,
                    char_count=c_count,
                )
            )

        full_raw = "\n\n".join(raw_texts)
        full_clean = "\n\n".join([p for p in cleaned_texts if p])

        # 5. Scanned Document Detection
        # Scanned/image-only documents have 0 or negligible extractable words (< 10 across the document)
        is_scanned = False
        if total_words < 10 and page_count >= 1:
            is_scanned = True

        # Estimated reading time: standard academic reading speed ~ 200 words per minute
        reading_time = round(total_words / 200.0, 1)

        error_msg = None
        if is_scanned:
            error_msg = (
                "Scanned Document Detected: This PDF contains no selectable or extractable text layer. "
                "The pages appear to be scanned images. To analyze this document, Optical Character "
                "Recognition (OCR) would be required to digitize image text."
            )

        return PDFDocumentResult(
            is_valid=True,
            filename=filename,
            page_count=page_count,
            total_word_count=total_words,
            total_char_count=total_chars,
            raw_full_text=full_raw,
            cleaned_full_text=full_clean,
            pages=pages_data,
            metadata=metadata,
            is_scanned=is_scanned,
            estimated_reading_time_mins=reading_time,
            error_message=error_msg,
        )

    finally:
        doc.close()
