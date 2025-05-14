# config.py
# Global configuration definitions for the interview application

from dataclasses import dataclass, field
from typing import Tuple

@dataclass
class LlamaConfig:
    """
    Configuration for the Ollama LLM client.
    """
    model: str = field(default="gemma3:4b")
    temperature: float = field(default=0.0)
    max_new_tokens: int = field(default=1000)
    top_p: float = field(default=1.0)
    frequency_penalty: float = field(default=0.0)
    presence_penalty: float = field(default=0.0)

@dataclass
class AudioConfig:
    """
    Configuration for audio recording and denoising.
    """
    sample_rate: int = field(default=16000)
    chunk_duration_ms: int = field(default=30)
    vad_aggressiveness: int = field(default=0)
    denoise_prop_decrease_default: float = field(default=0.0)
    denoise_prop_decrease_noise: float = field(default=0.0)
    energy_threshold_offset: int = field(default=100)
    vad_timeout_sec: float = 1

@dataclass
class VideoConfig:
    """
    Configuration for video recording.
    """
    output_file: str = field(default="interview_recording.avi")
    fps: float = field(default=20.0)
    resolution: Tuple[int, int] = field(default=(640, 480))

@dataclass
class STTConfig:
    """
    Configuration for speech-to-text via faster-whisper.
    """
    model_name: str = field(default="base")
    device: str = field(default="cpu")
    compute_type: str = field(default="int8")
    beam_size: int = field(default=5)
    language: str = field(default="ko")

@dataclass
class CLIConfig:
    """
    Default CLI options.
    """
    default_output_video: str = field(default="interview_recording.avi")
    default_resume_path: str = field(default="resume.pdf")
