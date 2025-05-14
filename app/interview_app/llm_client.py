# llm_client.py
# Wrapper around the Ollama LLM for prompt execution and response cleaning

import json
import logging
from typing import Optional

from langchain_ollama import OllamaLLM
from .config import LlamaConfig

logger = logging.getLogger(__name__)


def clean_llm_response(response: str) -> str:
    """
    Strip markdown code fences and surrounding whitespace from an LLM response.
    """
    text = response.strip()
    # Remove leading ``` or ```json
    if text.startswith("```"):
        # Split off first line if it's a code fence
        parts = text.split("\n", 1)
        if len(parts) == 2 and parts[0].startswith("```"):
            text = parts[1]
    # Remove trailing ```
    if text.endswith("```"):
        text = text[: -3]
    return text.strip()


class LLMClient:
    """
    Client to interact with OllamaLLM using a shared configuration.
    """
    def __init__(self, config: LlamaConfig):
        self.config = config
        # Initialize OllamaLLM with parameters from config
        self.llm = OllamaLLM(
            model=self.config.model,
            temperature=self.config.temperature,
            max_new_tokens=self.config.max_new_tokens,
            top_p=self.config.top_p,
            frequency_penalty=self.config.frequency_penalty,
            presence_penalty=self.config.presence_penalty,
        )

    def call(self, prompt: str) -> str:
        """
        Send a prompt to the LLM and return the cleaned text response.
        """
        try:
            raw = self.llm.invoke(prompt)
            return clean_llm_response(raw)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise

    def call_json(self, prompt: str) -> Optional[dict]:
        """
        Send a prompt to the LLM, clean the response, and parse it as JSON.
        Returns None if parsing fails.
        """
        text = self.call(prompt)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse LLM JSON response:\n{text}")
            return None
