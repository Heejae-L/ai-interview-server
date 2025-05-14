# unified_api.py

import os, uuid, shutil
from typing import List

from fastapi import FastAPI, APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, PlainTextResponse, FileResponse

# -- 기존 interview_app 기능 --
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
)

# -- 포즈 분석 기능 --
from pose_detection import analyze_video


# ┌─────────────────────────────────────────────────────┐
# │                   전역 설정                        │
# └─────────────────────────────────────────────────────┘
TMP_DIR   = "./tmp"
UPLOAD_DIR= "./uploads"
LOG_DIR   = "./logs"
os.makedirs(TMP_DIR,    exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(LOG_DIR,    exist_ok=True)

# 서버 카메라 녹화기
video_recorder = VideoRecorder(config=VideoConfig())

# ┌─────────────────────────────────────────────────────┐
# │                FastAPI 및 라우터                    │
# └─────────────────────────────────────────────────────┘
app = FastAPI(title="Unified Interview API")

# ——————————————————————
# interview_app 관련 엔드포인트
# ——————————————————————
ia_router = APIRouter(prefix="", tags=["InterviewCore"])

@ia_router.post("/parse_resume")
async def parse_resume(file: UploadFile = File(...)):
    tmp = os.path.join(TMP_DIR, f"{uuid.uuid4()}_{file.filename}")
    with open(tmp, "wb") as f: shutil.copyfileobj(file.file, f)
    try:
        data = ResumeJsonParser().parse_to_file(tmp)
    except Exception as e:
        raise HTTPException(500, f"파싱 실패: {e}")
    return JSONResponse(content=data)

@ia_router.post("/generate_questions")
async def generate_questions(file: UploadFile = File(...)):
    tmp = os.path.join(TMP_DIR, f"{uuid.uuid4()}_{file.filename}")
    with open(tmp, "wb") as f: shutil.copyfileobj(file.file, f)
    try:
        qs = InterviewQuestionMaker().generate_questions(tmp)
    except Exception as e:
        raise HTTPException(500, f"질문 생성 실패: {e}")
    return {"questions": qs}

@ia_router.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    tmp = os.path.join(TMP_DIR, f"{uuid.uuid4()}_{file.filename}")
    with open(tmp, "wb") as f: shutil.copyfileobj(file.file, f)
    try:
        words = STTClient().transcribe(tmp)
    except Exception as e:
        raise HTTPException(500, f"STT 실패: {e}")
    return {"word_timestamps": words}

@ia_router.post("/audio_metrics")
async def audio_metrics(file: UploadFile = File(...)):
    tmp = os.path.join(TMP_DIR, f"{uuid.uuid4()}_{file.filename}")
    with open(tmp, "wb") as f: shutil.copyfileobj(file.file, f)
    stt = STTClient()
    try:
        wts = stt.transcribe(tmp)
        silence = calculate_silence_duration(wts)
    except:
        silence = 0.0
    duration = calculate_audio_duration(tmp)
    return {"duration_sec": duration, "silence_sec": silence}

@ia_router.post("/evaluate")
async def evaluate_endpoint(
    background: BackgroundTasks,
    questions: List[str] = Form(...),
    answers:   List[str] = Form(...),
    audio_files: List[UploadFile] = File(...),
    output_file: str = Form("interview_evaluation.txt")
):
    paths = []
    for af in audio_files:
        p = os.path.join(TMP_DIR, f"{uuid.uuid4()}_{af.filename}")
        with open(p, "wb") as f: shutil.copyfileobj(af.file, f)
        paths.append(p)
    try:
        evaluate_and_save_responses(questions, answers, paths, output_file)
    except Exception as e:
        raise HTTPException(500, f"평가 실패: {e}")
    try:
        content = open(output_file, "r", encoding="utf-8").read()
    except Exception as e:
        raise HTTPException(500, f"결과 읽기 실패: {e}")
    return PlainTextResponse(content, media_type="text/plain")

@ia_router.post("/video/start")
async def start_video():
    video_recorder.start_recording()
    return {"status": "video recording started"}

@ia_router.post("/video/stop")
async def stop_video():
    video_recorder.stop_recording()
    return {"status": f"video recording stopped, saved to {video_recorder.output_file}"}

@ia_router.post("/audio/record_and_transcribe")
async def record_and_transcribe_audio(output_file: str = Form("response.wav")):
    rec = AudioRecorder()
    wav = rec.record(output_file=output_file)
    if not wav:
        raise HTTPException(504, "녹음 실패: 음성 없음")
    words = STTClient().transcribe(wav)
    return {"wav_file": wav, "word_timestamps": words}


# ——————————————————————
# 포즈 분석 관련 엔드포인트
# ——————————————————————
pose_router = APIRouter(prefix="/pose", tags=["PoseAnalysis"])

@pose_router.post(
    "/analyze",
    response_class=PlainTextResponse,
    summary="면접 영상 Pose 분석 → 로그 반환"
)
async def pose_analyze(file: UploadFile = File(...)):
    vid_id = uuid.uuid4().hex
    fname  = f"{vid_id}_{file.filename}"
    vpath  = os.path.join(UPLOAD_DIR, fname)
    with open(vpath, "wb") as f: shutil.copyfileobj(file.file, f)
    logp = os.path.join(LOG_DIR, f"{vid_id}.txt")
    try:
        analyze_video(vpath, logp)
    except Exception as e:
        raise HTTPException(500, f"분석 오류: {e}")
    return PlainTextResponse(open(logp, "r", encoding="utf-8").read())

@pose_router.get(
    "/log/{video_id}",
    response_class=FileResponse,
    summary="분석 로그 다운로드"
)
async def pose_get_log(video_id: str):
    fp = os.path.join(LOG_DIR, f"{video_id}.txt")
    if not os.path.exists(fp):
        raise HTTPException(404, "로그를 찾을 수 없습니다.")
    return FileResponse(fp, media_type="text/plain", filename=os.path.basename(fp))


# ┌─────────────────────────────────────────────────────┐
# │               라우터 등록 및 서버 실행               │
# └─────────────────────────────────────────────────────┘
app.include_router(ia_router)
app.include_router(pose_router)


# uvicorn 으로 직접 띄울 때
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("unified_api:app", host="0.0.0.0", port=8000, reload=True)