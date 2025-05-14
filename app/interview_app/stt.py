# stt.py
# Speech-to-Text utilities using faster-whisper

import os
import time
import wave
import logging
from typing import List, Dict, Optional

from faster_whisper import WhisperModel
from .config import STTConfig

logger = logging.getLogger(__name__)

class STTClient:
    """
    Wrapper around faster-whisper WhisperModel for speech-to-text processing.

    Usage:
        config = STTConfig()
        stt = STTClient(config)
        word_timestamps = stt.transcribe("response.wav")
    """
    def __init__(self, config: Optional[STTConfig] = None):
        # Load configuration (model name, device, compute type, beam size, language)
        self.config = config or STTConfig()
        # Disable HF symlink warnings and allow duplicate OpenMP libs
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        # Initialize Whisper model
        self.model = WhisperModel(
            model_size_or_path=self.config.model_name,
            device=self.config.device,
            compute_type=self.config.compute_type
        )
        logger.info(f"Initialized WhisperModel({self.config.model_name}) on {self.config.device}")

    def transcribe(self, audio_path: str) -> List[Dict]:
        """
        Transcribe the given audio file, returning a list of word-level timestamps.
        Each entry has keys: 'word', 'start', 'end'.

        :param audio_path: Path to the audio file
        :return: List of dicts with word and timing
        """
        start_time = time.time()
        segments, info = self.model.transcribe(
            audio_path,
            beam_size=self.config.beam_size,
            word_timestamps=True,
            language=self.config.language
        )
        word_timestamps: List[Dict] = []
        for segment in segments:
            for w in segment.words:
                word_timestamps.append({
                    "word": w.word,
                    "start": round(w.start, 2),
                    "end": round(w.end, 2)
                })
        elapsed = time.time() - start_time
        logger.info(f"Transcription completed in {elapsed:.2f}s, {len(word_timestamps)} words detected")
        return word_timestamps

    def get_text(self, audio_path: str) -> str:
        """
        Convenience method: transcribe and concatenate words into a single text string.
        """
        timestamps = self.transcribe(audio_path)
        return " ".join(item["word"] for item in timestamps)


def calculate_silence_duration(word_timestamps: List[Dict]) -> float:
    """
    Calculate total silence duration (in seconds) between consecutive words.

    :param word_timestamps: List of dicts with 'start' and 'end' times
    :return: Total silence duration rounded to 2 decimal places
    """
    total_silence = 0.0
    for prev, curr in zip(word_timestamps, word_timestamps[1:]):
        gap = curr["start"] - prev["end"]
        if gap > 0:
            total_silence += gap
    return round(total_silence, 2)


def calculate_audio_duration(audio_path: str) -> float:
    """
    Return the duration of an audio file in seconds.

    :param audio_path: Path to WAV audio file
    :return: Duration in seconds (2 decimal places)
    """
    try:
        with wave.open(audio_path, 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration = frames / float(rate)
        return round(duration, 2)
    except FileNotFoundError:
        logger.error(f"Audio file not found: {audio_path}")
        return 0.0
    except wave.Error as e:
        logger.error(f"Error reading audio file {audio_path}: {e}")
        return 0.0
