import re
import logging
from typing import Union, IO
from PyPDF2 import PdfReader, errors as pdf_errors

logger = logging.getLogger(__name__)


def extract_text(source: Union[str, IO]) -> str:
    """
    Extracts and returns cleaned text from all pages of a PDF.
    :param source: Path to PDF file or file-like object.
    :return: Combined cleaned text from all pages.
    :raises FileNotFoundError: If the file path is invalid.
    :raises PdfReadError: If PyPDF2 fails to read the PDF.
    """
    try:
        reader = PdfReader(source)
    except FileNotFoundError as e:
        logger.error(f"PDF file not found: {e}")
        raise
    except pdf_errors.PdfReadError as e:
        logger.error(f"Error reading PDF: {e}")
        raise

    pages = []
    for page in reader.pages:
        raw_text = page.extract_text() or ""
        cleaned = cleanup_text(raw_text)
        pages.append(cleaned)
    return "\n\n".join(pages)


def cleanup_text(raw: str) -> str:
    """
    Cleans raw PDF-extracted text by normalizing whitespace,
    removing stray punctuation patterns, and stripping URLs.
    :param raw: Raw text from PDF page.
    :return: Cleaned text.
    """
    patterns = {
        r"\s[,.]": ",",
        r"[\n]+": "\n",
        r"[\s]+": " ",
        r"http[s]?(://)?": "",
    }
    text = raw
    for pattern, repl in patterns.items():
        text = re.sub(pattern, repl, text)
    return text.strip()
