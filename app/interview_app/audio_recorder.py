# audio_recorder.py

import logging
import wave
from typing import Optional
import noisereduce as nr
import webrtcvad
import threading
import numpy as np

# pydub의 playback 기능
from pydub import AudioSegment

import pyaudio
import speech_recognition as sr
from rich.console import Console
from speech_recognition.exceptions import WaitTimeoutError

from .config import AudioConfig

logger = logging.getLogger(__name__)
console = Console()

class AudioRecorder:
    """
    마이크에서 음성을 듣고, 말소리 시작 후 non_speaking_duration(무음 유지 시간)만큼
    침묵이 지속되면 자동으로 녹음을 멈추는 구현체.
    """

    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()
        self.chunk_size = int(self.config.sample_rate * self.config.chunk_duration_ms / 1000)
        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold = self.config.vad_timeout_sec
        self.microphone = sr.Microphone(sample_rate=self.config.sample_rate, chunk_size=self.chunk_size)
        self.buffer = []
        self.recording = False
        self.vad = webrtcvad.Vad(1)
        self.adjust_noise()
        console.print("[bold green]Audio_record 초기화 성공[/bold green]")
        
    def adjust_noise(self):
        # 주변 소음 레벨 보정
        with self.microphone as src:
            console.print("[bold green]주변 소음에 맞게 조정 중...[/bold green]")
            self.recognizer.adjust_for_ambient_noise(src)
            self.recognizer.energy_threshold += 100

    def record_start(self):
        if not self.recording:
            self.record_thread = threading.Thread(target=self._record_start)
            self.record_thread.start()

    def _record_start(self):
        self.recording = True
        self.buffer = []
        console.print("[bold green]음성 녹음 시작됨.[/bold green]")
        no_voice_target_cnt = self.config.vad_timeout_sec * 1000
        no_voice_cnt = 0
        with self.microphone as source:
            while self.recording:
                sample_width = self.microphone.SAMPLE_WIDTH if hasattr(self.microphone, 'SAMPLE_WIDTH') else 2
                expected_bytes = self.chunk_size * sample_width
                chunk = source.stream.read(expected_bytes)
                self.buffer.append(chunk)
                if self._vad(chunk, self.config.sample_rate):
                    no_voice_cnt = 0
                else:
                    no_voice_cnt += self.config.chunk_duration_ms
                if no_voice_cnt >= no_voice_target_cnt:
                    self.recording = False

    def _vad(self, chunk, sample_rate):
        sample_width = getattr(self.microphone, "SAMPLE_WIDTH", 2)
        expected_bytes = self.chunk_size * sample_width
        if len(chunk) == 2 * expected_bytes:
            chunk_array = np.frombuffer(chunk, dtype=np.int16)
            mono_array = chunk_array[::2]
            chunk = mono_array.tobytes()
        elif len(chunk) != expected_bytes:
            print(f"Warning: Received chunk size of {len(chunk)} bytes, expected {expected_bytes} bytes.")
        return self.vad.is_speech(chunk, sample_rate)
    
    def record_stop(self, denoise_value, output_file="response_audio.wav"):
        """
        녹음을 중지하고 음성 데이터를 파일로 저장한 뒤 반환합니다.

        :param denoise_value: 디노이즈 강도
        :param output_file: 저장할 음성 파일 경로
        :return: 저장된 음성 파일 경로
        """
        self.sample_rate = self.config.sample_rate
        self.recording = False
        self.record_thread.join()
        audio_data = np.frombuffer(b''.join(self.buffer), dtype=np.int16)
        sample_rate = self.microphone.SAMPLE_RATE
        sample_width = getattr(self.microphone, "SAMPLE_WIDTH", 2)
        denoised_audio = self._denoise_process(audio_data, sample_rate, denoise_value)
        self._save_buffer_to_wav(
            denoised_audio['buffer_denoise'],
            denoised_audio['sample_rate'],
            sample_width,
            output_file
        )
        console.print(f"[bold green]녹음된 음성이 {output_file}에 저장되었습니다.[/bold green]")
        return output_file

    def load_wav(self, path, denoise_value):
        buffer = []
        with wave.open(path, 'rb') as wf:
            chunk_size = self.chunk_size
            data = wf.readframes(chunk_size)
            while data:
                buffer.append(data)
                data = wf.readframes(chunk_size)
        audio_data = np.frombuffer(b''.join(buffer), dtype=np.int16)
        sample_rate = wf.getframerate()
        return self._denoise_process(audio_data, sample_rate, denoise_value)

    def _denoise_process(self, audio_data, sample_rate, denoise_value):
        denoise = nr.reduce_noise(y=audio_data, sr=sample_rate, prop_decrease=denoise_value)
        buffer_denoise = [denoise.tobytes()]
        noise = nr.reduce_noise(y=audio_data, sr=sample_rate, prop_decrease=0.0)
        buffer_noise = [noise.tobytes()]
        self._save_buffer_to_wav(buffer_denoise, sample_rate, self.microphone.SAMPLE_WIDTH, 'input_denoise.wav')
        self._save_buffer_to_wav(buffer_noise, sample_rate, self.microphone.SAMPLE_WIDTH, 'input_noise.wav')
        audio_denoise = self._buffer_to_numpy(buffer_denoise, sample_rate)
        audio_noise = self._buffer_to_numpy(buffer_noise, sample_rate)
        return {'buffer_denoise': buffer_denoise,'audio_denoise': audio_denoise, 'audio_noise': audio_noise, 'sample_rate': sample_rate}

    def _buffer_to_numpy(self, buffer, sample_rate):
        audio_data = np.frombuffer(b''.join(buffer), dtype=np.int16)
        audio_data = audio_data.astype(np.float32) / 32768.0
        return audio_data

    def _save_buffer_to_wav(self, buffer, sample_rate, sample_width, filename):
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            wf.writeframes(b''.join(buffer))
    """       
    def record(self, output_file: str = "response.wav") -> str:
        
        사용자가 말하기를 시작하면 내부적으로 listen()이 녹음을 시작해,
        non_speaking_duration 초 침묵이 감지되면 자동으로 멈추고 WAV로 저장합니다.
        :param output_file: 저장할 WAV 파일 경로
        :return: output_file
       

        with self.microphone as src:
            console.print("[bold green]음성 녹음 시작. 말해주세요![/bold green]")
            try:
                # 음성 인식 시작
                audio_data = self.recognizer.listen(
                    src,
                    timeout=self.config.vad_timeout_sec,
                    phrase_time_limit=None
                )
            except WaitTimeoutError:
                console.print("[bold red]음성이 감지되지 않아 녹음을 종료합니다.[/bold red]")
                return None

        with wave.open(output_file, "wb") as wf:
            wf.setnchannels(1)  # 모노
            wf.setsampwidth(audio_data.sample_width)
            wf.setframerate(audio_data.sample_rate)
            wf.writeframes(audio_data.frame_data)
        # WAV 파일로 저장
        console.print(f"[bold green]녹음된 음성이 {output_file}에 저장되었습니다.[/bold green]")
        return output_file
    """