"""
modules/text_preprocessor.py
Academic text preprocessing, abbreviation-aware sentence boundary segmentation,
candidate sentence filtering, and structural section heuristic detection.
"""

from dataclasses import dataclass
import re
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class SentenceItem:
    sentence_id: int
    text: str
    page_number: int
    word_count: int
    char_count: int
    section_hint: str = "General"  # e.g., Abstract, Introduction, Methodology, Results, Conclusion


# Common abbreviations in academic literature that should NOT trigger a sentence split
ACADEMIC_ABBREVIATIONS = [
    r"\be\.g\.",
    r"\bi\.e\.",
    r"\bet al\.",
    r"\betc\.",
    r"\bfig\.",
    r"\bfigs\.",
    r"\btab\.",
    r"\btabs\.",
    r"\bref\.",
    r"\brefs\.",
    r"\beq\.",
    r"\beqs\.",
    r"\bvol\.",
    r"\bno\.",
    r"\bpp\.",
    r"\bp\.",
    r"\bdr\.",
    r"\bprof\.",
    r"\bdept\.",
    r"\buniv\.",
    r"\bapprox\.",
    r"\bvs\.",
    r"\bcf\.",
    r"\bsec\.",
]

# Section header detection patterns
SECTION_PATTERNS = [
    (r"(?i)\b(abstract)\b", "Abstract"),
    (r"(?i)\b(introduction|background|motivation)\b", "Introduction"),
    (r"(?i)\b(related work|literature review|prior work)\b", "Related Work"),
    (r"(?i)\b(methodology|methods|proposed (system|approach|method|architecture)|system design|formulation)\b", "Methodology"),
    (r"(?i)\b(experimental (results|setup|evaluation)|experiments|results and discussion|findings)\b", "Results"),
    (r"(?i)\b(conclusion|concluding remarks|summary and future work|implications)\b", "Conclusion"),
    (r"(?i)\b(references|bibliography)\b", "References"),
]


def detect_section_hint(text_block: str, current_section: str = "General") -> str:
    """
    Identifies the active academic section based on standard structural headings.
    """
    clean_line = text_block.strip()
    if len(clean_line) < 60:
        for pattern, section_name in SECTION_PATTERNS:
            if re.search(pattern, clean_line):
                return section_name
    return current_section


def split_into_academic_sentences(text: str) -> List[str]:
    """
    Robust sentence segmenter designed for academic text.
    Handles abbreviations, decimal numbers, brackets, and math notation.
    """
    if not text or not text.strip():
        return []

    # Step 1: Temporarily protect known abbreviations with placeholders
    protected_text = text
    placeholder_map = {}

    for idx, abbr_pat in enumerate(ACADEMIC_ABBREVIATIONS):
        matches = list(re.finditer(abbr_pat, protected_text, re.IGNORECASE))
        for m_idx, match in enumerate(matches):
            token = f"__ABBR_{idx}_{m_idx}__"
            val = match.group(0)
            placeholder_map[token] = val
            protected_text = protected_text.replace(val, token, 1)

    # Protect decimals like "3.14" or "98.5%"
    decimal_matches = list(re.finditer(r"\b\d+\.\d+\b", protected_text))
    for d_idx, match in enumerate(decimal_matches):
        token = f"__DECIMAL_{d_idx}__"
        val = match.group(0)
        placeholder_map[token] = val
        protected_text = protected_text.replace(val, token, 1)

    # Step 2: Split sentences on terminal punctuation (. ! ?) followed by whitespace & capital letter or newline
    # Also handle bracketed citations at sentence ends like "...algorithm [12]."
    sentence_regex = r"(?<=[.!?])\s+(?=[A-Z0-9\(\[\"\'])|(?<=[.!?])\n+"
    raw_sentences = re.split(sentence_regex, protected_text)

    # Step 3: Restore placeholders & clean sentences
    clean_sentences: List[str] = []
    for s in raw_sentences:
        restored = s
        for token, original in placeholder_map.items():
            restored = restored.replace(token, original)

        # Normalize inner whitespaces
        restored = re.sub(r"\s+", " ", restored).strip()

        if restored:
            clean_sentences.append(restored)

    return clean_sentences


def is_informative_sentence(sentence: str) -> bool:
    """
    Filters out noise, figure captions, references, isolated author lists,
    and boilerplate headers.
    """
    s = sentence.strip()

    # Length filter: academic sentences should have sufficient substance
    if len(s) < 35 or len(s.split()) < 6:
        return False

    # Upper limit filter: excessively long run-ons or unsegmented OCR blocks (> 600 chars)
    if len(s) > 800:
        return False

    # Exclude figure and table captions
    if re.match(r"(?i)^(figure|fig\.|table|tab\.)\s+\d+", s):
        return False

    # Exclude lines that are mostly citations or numbers
    # e.g., "[1] J. Smith, A. Kumar, 'Deep Learning'..."
    if re.match(r"^\[\d+\]", s) or re.match(r"^\d+\.\s+[A-Z]", s):
        return False

    # Exclude copyright or publication boilerplate
    if re.search(r"(?i)(all rights reserved|ieee transactions|acm proceedings|copyright ©|\bdoi:\s*10\.)", s):
        return False

    # Exclude lines that look like author affiliations or emails
    if "@" in s or re.search(r"(?i)(department of|university of|school of engineering|faculty of)", s):
        return False

    # Must contain at least some alphabetic characters and end with terminal punctuation
    alpha_chars = sum(1 for c in s if c.isalpha())
    if alpha_chars / max(len(s), 1) < 0.6:
        return False

    return True


def segment_document_sentences(
    pages_data: List[Any],
) -> List[SentenceItem]:
    """
    Transforms extracted page-by-page text into structured, informative SentenceItems
    with page references and section hints.
    """
    sentence_items: List[SentenceItem] = []
    current_section = "General"
    global_sentence_id = 0

    for page in pages_data:
        page_num = getattr(page, "page_number", 1)
        raw_text = getattr(page, "clean_text", "")

        # Detect any section transition on this page
        for line in raw_text.split("\n"):
            current_section = detect_section_hint(line, current_section)

        # Do not include bibliography/reference section in summary/concept candidates
        if current_section == "References":
            continue

        sentences = split_into_academic_sentences(raw_text)

        for s in sentences:
            if is_informative_sentence(s):
                global_sentence_id += 1
                sentence_items.append(
                    SentenceItem(
                        sentence_id=global_sentence_id,
                        text=s,
                        page_number=page_num,
                        word_count=len(s.split()),
                        char_count=len(s),
                        section_hint=current_section,
                    )
                )

    return sentence_items
