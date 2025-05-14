# pose_api.py
import os, uuid, shutil
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import PlainTextResponse, FileResponse

from .pose_detection import analyze_video

app = FastAPI(
    title="Interview Pose Analysis API",
    description="업로드된 면접 영상을 MediaPipe Pose로 분석하고 로그를 반환합니다."
)

# 업로드·로그 저장 폴더
UPLOAD_DIR = "uploads"
LOG_DIR    = "logs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


@app.post(
    "/pose/analyze",
    response_class=PlainTextResponse,
    summary="업로드된 영상 분석 후 로그(텍스트) 반환"
)
async def upload_and_analyze(file: UploadFile = File(...)):
    # 1) 동영상 저장
    video_id   = uuid.uuid4().hex
    filename   = f"{video_id}_{file.filename}"
    video_path = os.path.join(UPLOAD_DIR, filename)

    try:
        with open(video_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(500, f"파일 저장 실패: {e}")

    # 2) 분석 수행
    log_path = os.path.join(LOG_DIR, f"{video_id}.txt")
    try:
        analyze_video(video_path, log_path)
    except Exception as e:
        raise HTTPException(500, f"분석 중 오류 발생: {e}")

    # 3) 생성된 로그 바로 반환
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        raise HTTPException(500, f"로그 읽기 실패: {e}")

    return PlainTextResponse(content, media_type="text/plain")


@app.get(
    "/pose/log/{video_id}",
    response_class=FileResponse,
    summary="기존 분석 로그 파일 다운로드"
)
async def download_log(video_id: str):
    """
    이전에 분석된 결과가 있다면 로그(.txt)를 그대로 다운로드합니다.
    """
    filepath = os.path.join(LOG_DIR, f"{video_id}.txt")
    if not os.path.exists(filepath):
        raise HTTPException(404, f"{video_id} 로그를 찾을 수 없습니다.")
    return FileResponse(
        filepath,
        media_type="text/plain",
        filename=os.path.basename(filepath)
    )
