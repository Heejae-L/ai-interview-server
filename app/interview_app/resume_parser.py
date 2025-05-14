# resume_parser.py
# Extract structured resume data from PDF using LLM

import json
import logging
from typing import Union, IO, Optional

from .pdf_utils import extract_text
from .llm_client import LLMClient, clean_llm_response
from .prompts import PARSER_PROMPT
from .config import LlamaConfig

logger = logging.getLogger(__name__)


class ResumeJsonParser:
    """
    Parse a candidate's PDF resume into structured JSON using an LLM.
    """
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        config: Optional[LlamaConfig] = None
    ):
        # Initialize LLM client if not provided
        if llm_client is None:
            if config is None:
                config = LlamaConfig()
            llm_client = LLMClient(config)
        self.llm_client = llm_client

    def parse(self, pdf_source: Union[str, IO]) -> dict:
        """
        Extract text from PDF and parse into JSON dict.
        :param pdf_source: file path or binary stream of PDF
        :return: Parsed JSON as Python dict
        :raises: FileNotFoundError, PdfReadError, ValueError
        """
        # 1) Extract and clean text
        try:
            raw_text = extract_text(pdf_source)
        except Exception as e:
            logger.error(f"Failed to extract text from PDF: {e}")
            raise

        # 2) Build prompt and call LLM
        prompt = PARSER_PROMPT + "\n" + raw_text
        try:
            raw_response = self.llm_client.call(prompt)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise

        # 3) Clean and parse JSON
        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON from LLM: {e}\nResponse content: {raw_response}")
            raise ValueError("LLM response is not valid JSON")

        return parsed

    def parse_to_file(
        self,
        pdf_path: str,
        output_path: str = "resume.json"
    ) -> dict:
        """
        Parse PDF to JSON dict and save to a file.
        :param pdf_path: Path to the PDF file
        :param output_path: Destination JSON file path
        :return: Parsed JSON as Python dict
        """
        data = self.parse(pdf_path)
        try:
            with open(output_path, "w", encoding="utf-8") as fp:
                json.dump(data, fp, ensure_ascii=False, indent=4)
        except Exception as e:
            logger.error(f"Failed to write JSON to {output_path}: {e}")
            raise
        return data
