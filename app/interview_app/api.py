# api.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, PlainTextResponse
from typing import List
import shutil, os
import uuid

from interview_app import (
    ResumeJsonParser,
    InterviewQuestionMaker,
    STTClient,
    calculate_silence_duration,
    calculate_audio_duration,
    evaluate_and_save_responses,
    VideoRecorder,
    AudioRecorder,
    VideoConfig,
    CLIConfig,
)

app = FastAPI(title="InterviewApp API")

# 임시 파일을 저장할 폴더
TMP_DIR = "./tmp"
os.makedirs(TMP_DIR, exist_ok=True)

# 전역 비디오 녹화기 (서버 머신의 카메라 사용 시)
video_recorder = VideoRecorder(config=VideoConfig())
    
@app.post("/parse_resume")
async def parse_resume(file: UploadFile = File(...)):
    """
    이력서 PDF를 받아 JSON으로 파싱하여 반환합니다.
    """
    # 1) 파일 저장
    tmp_path = os.path.join(TMP_DIR, f"{uuid.uuid4()}_{file.filename}")
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # 2) 파싱
    parser = ResumeJsonParser()
    try:
        data = parser.parse_to_file(tmp_path)  # resume.json에 쓰임과 동시에 dict 리턴
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"파싱 실패: {e}")

    return JSONResponse(content=data)


@app.post("/generate_questions")
async def generate_questions(file: UploadFile = File(...)):
    """
    이력서 PDF를 받아 면접 질문 리스트를 생성하여 반환합니다.
    """
    tmp_path = os.path.join(TMP_DIR, f"{uuid.uuid4()}_{file.filename}")
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    maker = InterviewQuestionMaker()
    try:
        questions = maker.generate_questions(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"질문 생성 실패: {e}")

    return {"questions": questions}


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    업로드된 WAV/MP3 오디오를 전사(Word timestamps)하여 반환합니다.
    """
    tmp_path = os.path.join(TMP_DIR, f"{uuid.uuid4()}_{file.filename}")
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    stt = STTClient()
    try:
        word_ts = stt.transcribe(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT 실패: {e}")

    return {"word_timestamps": word_ts}


@app.post("/audio_metrics")
async def audio_metrics(file: UploadFile = File(...)):
    """
    업로드된 WAV 오디오 파일에 대해 총 재생시간과 무음 총합을 계산하여 반환합니다.
    """
    tmp_path = os.path.join(TMP_DIR, f"{uuid.uuid4()}_{file.filename}")
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    stt = STTClient()
    # 전사부터 무음 계산
    try:
        word_ts = stt.transcribe(tmp_path)
        silence = calculate_silence_duration(word_ts)
    except Exception:
        silence = 0.0

    duration = calculate_audio_duration(tmp_path)
    return {"duration_sec": duration, "silence_sec": silence}


@app.post("/evaluate")
async def evaluate(
    background: BackgroundTasks,
    questions: List[str] = Form(...),
    answers:   List[str] = Form(...),
    audio_files: List[UploadFile] = File(...),
    output_file: str = Form("interview_evaluation.txt")
):
    """
    질문 리스트, 사용자 답변 리스트, 대응하는 오디오 파일들을 받아
    평가를 수행하고, 텍스트 파일로 저장한 뒤, JSON 결과를 반환합니다.
    """
    # 1) 오디오 파일 저장
    tmp_audio_paths = []
    for af in audio_files:
        path = os.path.join(TMP_DIR, f"{uuid.uuid4()}_{af.filename}")
        with open(path, "wb") as f:
            shutil.copyfileobj(af.file, f)
        tmp_audio_paths.append(path)

    # 2) 평가 함수 실행 (동기)
    try:
        evaluate_and_save_responses(
            questions=questions,
            answers=answers,
            audio_files=tmp_audio_paths,
            output_file=output_file
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"평가 실패: {e}")

    # 3) 결과 파일을 읽어서 JSON으로 가공해 반환
    try:
        # 간단히 텍스트 파일을 그대로 반환하거나,
        # JSON 파싱하여 구조화할 수도 있습니다. 여기서는 텍스트 리턴.
        with open(output_file, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"결과 읽기 실패: {e}")

    return PlainTextResponse(content, media_type="text/plain")


@app.post("/video/start")
async def start_video():
    """
    서버 머신의 기본 카메라로 비디오 녹화를 시작합니다.
    (서버에 웹캠이 연결된 경우에만 동작)
    """
    video_recorder.start_recording()
    return {"status": "video recording started"}


@app.post("/video/stop")
async def stop_video():
    """
    비디오 녹화를 중지하고 파일로 저장합니다.
    """
    video_recorder.stop_recording()
    return {"status": f"video recording stopped, saved to {video_recorder.output_file}"}


@app.post("/audio/record_and_transcribe")
async def record_and_transcribe_audio(
    output_file: str = Form("response.wav")
):
    """
    서버 머신의 마이크로부터 음성을 자동 녹음(침묵 후 종료) → WAV 저장 → STT 전사 결과 반환
    """
    recorder = AudioRecorder()
    wav_path = recorder.record(output_file=output_file)
    if wav_path is None:
        raise HTTPException(status_code=504, detail="녹음 시간 초과: 음성이 감지되지 않았습니다.")
    stt = STTClient()
    word_ts = stt.transcribe(wav_path)
    # 필요 시 파일 삭제: os.remove(wav_path)
    return {"wav_file": wav_path, "word_timestamps": word_ts}
