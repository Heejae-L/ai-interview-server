#!/usr/bin/env python
# coding: utf-8

# In[18]:


import re
import PyPDF2
import cv2
import typer
import threading
import json
import os
import openai
import pyaudio
import wave
import numpy as np
import time
import torch
from whisper import load_model
from typing import Union, IO
from rich import print as pprint
from gtts import gTTS
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from dataclasses import dataclass
from IPython.display import clear_output
from pydub import AudioSegment
from dotenv import load_dotenv

# In[19]:


"""
A module for managing OpenAI API configuration parameters.

This module provides a dataclass, OpenAIConfig, to store and manage the
configuration parameters required for making requests to the OpenAI API.
The OpenAIConfig dataclass can be used to organize and pass these parameters
to other classes or functions that interact with the API.
"""
@dataclass
class OpenAIConfig:
    """
    A dataclass for storing OpenAI API configuration parameters.

    Attributes:
        model (str): The OpenAI model to be used, default is "gpt-3.5-turbo".
        temperature (float): Sampling temperature for the model, default is 0.0.
        max_tokens (int): Maximum number of tokens in the model's response, default is 1000.
        top_p (float): Nucleus sampling parameter to
        control the randomness of the model's response,
        default is 1.
        frequency_penalty (float): Penalty for token frequency, default is 0.
        presence_penalty (float): Penalty for token presence, default is 0.
    """

    model: str = "gpt-3.5-turbo"
    temperature: float = 0.0
    max_tokens: int = 1000
    top_p: float = 1
    frequency_penalty: float = 0
    presence_penalty: float = 0


# In[20]:


"""
A module of pre-defined prompts
"""

PARSER_PROMPT = """
I want to you extract information from a PDF resume.
Summarize it into a JSON with EXACTLY the following structure
///
{"personal_detail":{"first_name":"","last_name":"","email":"","phone_number":"","location":"","portfolio_website_url":"","linkedin_url":"","github_main_page_url":""},"education_history":[{"university":"","education_level":"","graduation_year":"","graduation_month":"","majors":"","GPA":""}],"work_experience":[{"job_title":"","company":"","location":"","begin_time":"","end_time":"","job_summary":""}],"project_experience":[{"project_name":"","project_description":""}]}
///
My original resume is as below
"""


# In[21]:


QUESTION_PROMPT = """
You are an experienced interviewer who specializes in generating specific interview questions based on a candidate's resume text.
Provide questions in exact JSON format with the following structure only. Do not add extra text or explanations outside the JSON format.

Output:
{{
  "technical_questions": [],
  "behavior_questions": [],
}}

Please generate the questions in Korean.

My resume text is as below
\"\"\"
{resume}
\"\"\"
"""


# In[22]:


class InterviewQuestionMaker:
    """
    Class to create interview questions based on a PDF resume.
    """

    def __init__(self, config: OpenAIConfig = OpenAIConfig(), prompt: str = QUESTION_PROMPT):
        """Initialize the InterviewQuestionMaker with the specified configuration."""
        self.config = config
        self.prompt = prompt

    def create_questions(self, pdf_stream: Union[str, IO]) -> str:
        """
        Create interview questions for the given PDF resume file.

        Args:
            pdf_stream (IO): The PDF file as a stream.
        """
        pdf_str = self.pdf_to_str(pdf_stream)
        prompt = self.complete_prompt(pdf_str)
        return query_ai(self.config, prompt)

    def complete_prompt(self, pdf_str: str) -> str:
        """
        Complete the prompt with the given PDF string.

        Args:
            pdf_str (str): PDF content as a string.
        """
        return self.prompt.format(resume=pdf_str)

    def pdf_to_str(self, pdf_stream: Union[str, IO]) -> str:
        """
        Convert the given PDF file to a string.

        Args:
            pdf_stream (IO): The PDF file as a stream.
        """
        pdf = PyPDF2.PdfReader(pdf_stream)
        pages = [self.format_pdf(p.extract_text()) for p in pdf.pages]
        return "\n\n".join(pages)

    def format_pdf(self, pdf_str: str) -> str:
        """
        Format the given PDF string by applying pattern replacements.

        Args:
            pdf_str (str): PDF content as a string.
        """

        pattern_replacements = {
            r"\s[,.]": ",",
            r"[\n]+": "\n",
            r"[\s]+": " ",
            r"http[s]?(://)?": "",
        }

        for pattern, replacement in pattern_replacements.items():
            pdf_str = re.sub(pattern, replacement, pdf_str)

        return pdf_str


# In[23]:


class ResumeJsonParser:
    """A class to parse resume PDF files and convert them into JSON format using GPT-3."""

    def __init__(self, config: OpenAIConfig = OpenAIConfig(), prompt: str = PARSER_PROMPT):
        """
        Initialize the ResumeJsonParser with the specified configuration.

        Args:
            config (OpenAIConfig): OpenAI API configuration.
            prompt (str): Custom prompt for GPT-3.
        """
        self.config = config
        self.prompt = prompt

    def pdf2json(self, pdf_path: str):
        """
        Convert the PDF resume file to a JSON representation.

        Args:
            pdf_path (str): Path to the PDF resume file.

        Returns:
            dict: JSON representation of the resume.
        """
        pdf_str = self.pdf2str(pdf_path)
        json_data = self.__str2json(pdf_str)
        return json_data

    def __str2json(self, pdf_str: str):
        """
        Convert the resume string to a JSON representation using GPT-3.

        Args:
            pdf_str (str): Resume string.

        Returns:
            dict: JSON representation of the resume.
        """
        prompt = self.__complete_prompt(pdf_str)
        return query_ai(self.config, prompt)

    def __complete_prompt(self, pdf_str: str) -> str:
        """
        Create a complete prompt by appending the resume string to the initial prompt.

        Args:
            pdf_str (str): Resume string.

        Returns:
            str: The complete prompt.
        """
        return self.prompt + pdf_str

    def pdf2str(self, pdf_path: str) -> str:
        """
        Convert a PDF file to a plain text string.

        Args:
            pdf_path (str): Path to the PDF file.

        Returns:
            str: Plain text string representing the PDF content.
        """
        with open(pdf_path, "rb") as pdf_file:
            pdf = PyPDF2.PdfReader(pdf_file)
            pages = [self.__format_pdf(p.extract_text()) for p in pdf.pages]
            return "\n\n".join(pages)

    def __format_pdf(self, pdf_str: str) -> str:
        """
        Clean and format the PDF text string by applying pattern replacements.

        Args:
            pdf_str (str): Original PDF text string.

        Returns:
            str: Cleaned and formatted PDF text string.
        """
        pattern_replacements = {
            r'\s[,.]': ',',
            r'[\n]+': '\n',
            r'[\s]+': ' ',
            r'http[s]?(://)?': ''
        }

        for pattern, replacement in pattern_replacements.items():
            pdf_str = re.sub(pattern, replacement, pdf_str)

        return pdf_str


# In[24]:


def query_ai(config: OpenAIConfig, prompt: str):
    """
    Query the OpenAI API with the provided configuration and prompt.

    Args:
        config (OpenAIConfig): Configuration parameters for the OpenAI API request.
        prompt (str): The prompt to be sent to the API.

    Returns:
        dict or str: Parsed JSON response from the API if valid, otherwise an error message.
    """
    try:
        response = openai.ChatCompletion.create(
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
            frequency_penalty=config.frequency_penalty,
            presence_penalty=config.presence_penalty,
            messages=[{"role": "user", "content": prompt}],
        )

        response_str = response.choices[0].message.content.strip()

        # JSON 유효성 검사
        try:
            return json.loads(response_str)
        except json.JSONDecodeError:
            # JSON으로 변환할 수 없는 경우, 디버그용 응답 반환
            return f"Error: Response is not in JSON format: {response_str}"

    except openai.APIError as api_exc:
        # Handle exceptions related to the OpenAI API
        return f"API Error: {api_exc}"
    except json.JSONDecodeError as json_exc:
        # Handle exceptions related to JSON decoding
        return f"JSON Decode Error: {json_exc}"


# In[25]:


# 설정값
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
THRESHOLD = 170
SILENCE_DURATION = 10
FILENAME = "response_audio.wav"


# In[26]:


app = typer.Typer()

# 인터뷰 질문 생성기 및 JSON 변환기 인스턴스 생성
json_parser: ResumeJsonParser = ResumeJsonParser()
question_maker: InterviewQuestionMaker = InterviewQuestionMaker()

console = Console()


# In[27]:


# 녹화 시작 및 종료를 관리하는 클래스
class VideoRecorder:
    def __init__(self, output_file="interview_recording.avi", fps=20.0, resolution=(640, 480)):
        self.output_file = output_file
        self.fps = fps
        self.resolution = resolution
        self.running = False
        self.cap = None
        self.out = None
        self.thread = None

    def start_recording(self):
        """Start video recording."""
        self.cap = cv2.VideoCapture(0)  # Open the webcam
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter(self.output_file, fourcc, self.fps, self.resolution)
        self.running = True

        def record():
            while self.running:
                ret, frame = self.cap.read()
                if ret:
                    self.out.write(frame)

        self.thread = threading.Thread(target=record)
        self.thread.start()

    def stop_recording(self):
        """Stop video recording."""
        self.running = False
        self.thread.join()
        if self.cap:
            self.cap.release()
        if self.out:
            self.out.release()
        cv2.destroyAllWindows()


# In[28]:


# 침묵 확인 함수
def is_silent(data):
    audio_data = np.frombuffer(data, dtype=np.int16)
    if len(audio_data) == 0:
        return True, 0.0
    rms = np.sqrt(np.abs(np.mean(np.square(audio_data))))
    return rms < THRESHOLD, rms


# In[29]:


# 녹음 함수
def record_audio(filename=FILENAME):
    console.print("[bold green]음성 녹음을 시작합니다. 말을 시작하세요...[/bold green]")

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    frames = []
    silence_start = None

    while True:
        data = stream.read(CHUNK)
        silent, rms = is_silent(data)
        frames.append(data)

        if not silent:
            silence_start = None
        else:
            if silence_start is None:
                silence_start = time.time()
            elif time.time() - silence_start > SILENCE_DURATION:
                console.print("\n[bold red]침묵이 10초 이상 지속되어 녹음을 중지합니다.[/bold red]")
                break

    # 스트림 종료
    stream.stop_stream()
    stream.close()
    p.terminate()

    # WAV 파일 저장
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    abs_path = os.path.abspath(filename)
    console.print(f"녹음이 완료되었습니다")
    return abs_path


# In[48]:


def transcribe_audio(file_path):
     # GPU 사용 가능 여부에 따라 device 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model("base", device=device)  # 'fp16' 인수 제거
    result = model.transcribe(file_path)
    console.print("[bold yellow]음성 텍스트 변환 결과:[/bold yellow]", result["text"])
    return result["text"]


# In[49]:


def convert_mp3_to_wav(mp3_filename, wav_filename):
    audio = AudioSegment.from_mp3(mp3_filename)
    audio.export(wav_filename, format="wav")


# In[50]:


# WAV 파일 길이 계산 함수
def get_wav_duration(wav_filename):
    with wave.open(wav_filename, 'rb') as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        duration = frames / float(rate)
    return duration


# In[51]:


def display_questions_with_tts(questions):
    console.print("You will be asked a series of questions. Answer verbally after each question.")
    console.print("Speak '그만하겠습니다' at any time to end the interview.\n")

    for i, question in enumerate(questions, 1):
        console.print(f"[bold blue]Question {i}:[/bold blue] {question}")

        # TTS로 질문을 MP3 형식으로 저장
        tts = gTTS(text=question, lang='ko')
        mp3_filename = f"question_{i}.mp3"
        wav_filename = f"question_{i}.wav"
        tts.save(mp3_filename)

        # MP3 파일을 WAV로 변환
        audio = AudioSegment.from_mp3(mp3_filename)
        audio.export(wav_filename, format="wav")

        # WAV 파일 길이를 계산하여 대기 시간 설정
        wav_duration = get_wav_duration(wav_filename)

        # 음성 파일 재생
        os.system(f"start {wav_filename}")  # Windows용, Mac에서는 'open', Linux에서는 'xdg-open' 사용

        # 음성 파일 재생 시간만큼 대기
        time.sleep(wav_duration + 5)

        # 사용자 응답 녹음 및 텍스트 변환
        response_audio_path = record_audio()
        response_text = transcribe_audio(response_audio_path)

        # 종료 조건 확인
        if "그만하겠습니다" in response_text.strip().lower():
            console.print("[bold red]Interview ended by the user.[/bold red]")
            break

        console.print(f"[bold yellow]사용자 답변:[/bold yellow] {response_text}")

        # 재생 후 파일 삭제
        os.remove(mp3_filename)
        os.remove(wav_filename)
        os.remove(response_audio_path)


# In[52]:


# VideoRecorder 인스턴스 생성
video_recorder = VideoRecorder()


# In[53]:


@app.command()
def start_full_interview(file_path: str, output_video: str = "interview_recording.avi"):
    """
    PDF 이력서 파일을 기반으로 JSON 추출, 질문 생성, 면접 진행 및 녹화를 수행합니다.

    Args:
        file_path (str): PDF 이력서 파일의 경로.
        output_video (str): 녹화 영상 파일의 출력 경로.
    """
    # 녹화 설정
    video_recorder = VideoRecorder(output_file=output_video)

    # JSON 파서 및 질문 생성기 초기화
    json_parser = ResumeJsonParser()
    question_maker = InterviewQuestionMaker()

    console.print("[bold green]Starting Full Interview Process[/bold green]")

    try:
        # 면접 시작 시 녹화 시작
        console.print("[bold green]Recording started![/bold green]")
        video_recorder.start_recording()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
            console=console
        ) as progress:
            # Step 1: PDF를 JSON으로 변환
            progress.add_task(description="Parsing resume to JSON...", total=None)
            resume_json = json_parser.pdf2json(file_path)

            if not isinstance(resume_json, dict):
                console.print(f"[bold red]Error parsing resume: {resume_json}[/bold red]")
                return

            console.print("[bold green]Resume successfully parsed to JSON.[/bold green]")

            # Step 2: JSON 기반 질문 생성
            progress.add_task(description="Generating interview questions...", total=None)
            questions = question_maker.create_questions(file_path)

            if isinstance(questions, dict):
                technical_questions = questions.get("technical_questions", [])
                behavioral_questions = questions.get("behavior_questions", [])
                all_questions = technical_questions + behavioral_questions
            else:
                console.print(f"[bold red]Error generating questions: {questions}[/bold red]")
                return

            if not all_questions:
                console.print("[bold red]No questions generated from the resume.[/bold red]")
                return

            console.print("[bold green]Questions successfully generated.[/bold green]")

            # Step 3: 면접 진행
            console.print("[bold green]Starting the Mock Interview![/bold green]")
            display_questions_with_tts(all_questions)

    finally:
        # 면접 종료 시 녹화 중단
        video_recorder.stop_recording()
        console.print(f"[bold green]Recording stopped and saved as {output_video}.[/bold green]")

# 코드를 실행하면 전체 흐름이 동작하며 모든 클래스를 활용하게 됩니다.
# PDF 파일에서 JSON을 추출하고 질문을 생성한 후 면접을 진행하며, 면접 과정을 녹화합니다.


# In[54]:

load_dotenv()  # .env 파일 읽기
# OpenAI API 키 설정
os.getenv("OPENAI_API_KEY")

# 이제 start_mock_interview를 실행해 오류 없이 API 요청을 할 수 있습니다.
if __name__ == "__main__":
    start_full_interview("C:\\ExFile\\sample.pdf", "interview.avi")

