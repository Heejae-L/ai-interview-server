# interview_flow.py
# Core interview orchestration: resume parsing, question generation, interview loop, and evaluation

import os
import time
from typing import List

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play

from .config import VideoConfig
from .resume_parser import ResumeJsonParser
from .question_maker import InterviewQuestionMaker
from .video_recorder import VideoRecorder
from .audio_recorder import AudioRecorder
from .stt import STTClient, calculate_silence_duration, calculate_audio_duration
from .evaluation import evaluate_and_save_responses

console = Console()


def display_questions_with_tts_and_evaluation(
    questions: List[str],
    video_recorder: VideoRecorder
) -> None:
    """
    Play each question via TTS, record user voice until silence, transcribe, and collect responses.
    Then evaluate all responses via LLM.
    """
    console.print("You will be asked a series of questions. Answer verbally after each question.")
    console.print("Speak '그만하겠습니다' at any time to end the interview.\n")

    answers: List[str] = []
    audio_files: List[str] = []
    stt_client = STTClient()

    for idx, question in enumerate(questions, start=1):
        console.print(f"[bold blue]Question {idx}/{len(questions)}:[/bold blue] {question}")

        # Start video recording on first question
        if idx == 1:
            console.print("[bold green]Starting video recording...[/bold green]")
            video_recorder.start_recording()

        # Generate TTS and play
        tts = gTTS(text=question, lang='ko')
        mp3_file = f"question_{idx}.mp3"
        tts.save(mp3_file)
        audio = AudioSegment.from_mp3(mp3_file)
        play(audio)
        time.sleep(len(audio) / 1000.0 + 0.5)
        os.remove(mp3_file)

        # Record user response via microphone
        recorder = AudioRecorder()
        recorder.record_start()
        
        while recorder.recording:
            time.sleep(0.1)
        response_audio_file = recorder.record_stop(denoise_value=0.0, output_file=f"response_{idx}.wav")
        audio_files.append(response_audio_file)  # Save the audio file path
        
        word_timestamps = stt_client.transcribe(response_audio_file)
        response_text = " ".join(item['word'] for item in word_timestamps)
        answers.append(response_text)  # Save the user's response
        if "그만하겠습니다" in response_text.strip().lower():
            console.print("[bold red]Interview ended by the user.[/bold red]")
            break
        console.print(f"[bold yellow]사용자 답변:[/bold yellow] {response_text}")

    # Stop video recording
    video_recorder.stop_recording()
    console.print("[bold green]Video recording stopped.[/bold green]")

    # Evaluate and save all Q&A
    evaluate_and_save_responses(questions, answers, audio_files)


def start_full_interview(
    file_path: str,
    output_video: str = "interview_recording.avi"
) -> None:
    """
    Main entry: parse resume -> generate questions -> run interview loop -> evaluate & save results
    """
    video_config = VideoConfig(output_file=output_video)
    video_recorder = VideoRecorder(config=video_config)
    parser = ResumeJsonParser()
    question_maker = InterviewQuestionMaker()

    console.print("[bold green]Starting Full Interview Process[/bold green]")

    try:
        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
            transient=True, console=console
        ) as progress:
            # 1) Parse resume to JSON
            progress.add_task(description="Parsing resume to JSON...", total=None)
            resume_data = parser.parse_to_file(file_path)
            console.print("[bold green]Resume successfully parsed to JSON.[/bold green]")

            # 2) Generate interview questions
            progress.add_task(description="Generating interview questions...", total=None)
            questions = question_maker.generate_questions(file_path)
            if not isinstance(questions, list) or not questions:
                console.print("[bold red]No questions generated. Exiting.[/bold red]")
                return
            console.print(f"[bold green]{len(questions)} questions generated.[/bold green]")

        # 3) Conduct interview loop and evaluation
        display_questions_with_tts_and_evaluation(questions, video_recorder)

    finally:
        # Ensure video recording is properly stopped
        if video_recorder.running:
            video_recorder.stop_recording()
        console.print(f"[bold green]Interview process completed. Video saved to {output_video}[/bold green]")
