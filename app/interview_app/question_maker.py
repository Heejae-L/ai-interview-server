# question_maker.py
# Generate interview questions based on resume using LLM

import json
import logging
from typing import Union, IO, Optional, List

from .pdf_utils import extract_text
from .llm_client import LLMClient
from .prompts import QUESTION_PROMPT
from .config import LlamaConfig

logger = logging.getLogger(__name__)


class InterviewQuestionMaker:
    """
    Generate a sequence of interview questions from a candidate's resume using an LLM.
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

    def generate_questions(
        self,
        pdf_source: Union[str, IO]
    ) -> List[str]:
        """
        Extract text from the resume PDF and generate a list of interview questions.
        :param pdf_source: Path or file-like object for the PDF resume
        :return: List of questions in Korean
        :raises ValueError: if LLM response is invalid or JSON parsing fails
        """
        # 1) Extract and clean text
        try:
            resume_text = extract_text(pdf_source)
        except Exception as e:
            logger.error(f"Failed to extract text from PDF: {e}")
            raise

        # 2) Construct prompt
        # prompt = QUESTION_PROMPT + "\n" + resume_text + "\n" + '"""'
        prompt = f"{QUESTION_PROMPT}\n{resume_text}\n"

        # 3) Call LLM and parse JSON
        try:
            raw_response = self.llm_client.call(prompt)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise

        try:
            data = json.loads(raw_response)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON from LLM: {e}\nRaw response: {raw_response}")
            raise ValueError("LLM response is not valid JSON")

        
        # dict 형태로 {"questions": [...]} 를 받을 때
        if isinstance(data, dict) and isinstance(data.get("questions"), list):
            return data["questions"]
        
        # 순수 리스트 형태를 받을 때
        if isinstance(data, list) and all(isinstance(q, str) for q in data):
            return data

         # 그 외 형식 오류
        logger.error(f"Unexpected LLM JSON format: {data}")
        raise ValueError("LLM JSON did not contain a question list")

    def generate_and_save(
        self,
        pdf_path: str,
        output_path: str = "questions.json"
    ) -> List[str]:
        """
        Generate questions and save the result JSON to a file.
        :param pdf_path: Path to the PDF resume
        :param output_path: Destination JSON file
        :return: List of questions
        """
        questions = self.generate_questions(pdf_path)
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump({"questions": questions}, f, ensure_ascii=False, indent=4)
        except Exception as e:
            logger.error(f"Failed to write questions to {output_path}: {e}")
            raise
        return questions
