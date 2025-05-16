import os
import uuid
import shutil
from typing import List

from fastapi import FastAPI, APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
# Add HTMLResponse and FileResponse (FileResponse might already be there)
from fastapi.responses import JSONResponse, PlainTextResponse, FileResponse, HTMLResponse
# No need for StaticFiles if serving favicon directly like this

# ... (your existing imports for interview_app and pose_detection) ...
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
from pose_detection import analyze_video


# ┌─────────────────────────────────────────────────────┐
# │                   전역 설정                        │
# └─────────────────────────────────────────────────────┘
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TMP_DIR   = os.path.join(BASE_DIR, "tmp")
UPLOAD_DIR= os.path.join(BASE_DIR, "uploads")
LOG_DIR   = os.path.join(BASE_DIR, "logs")
STATIC_DIR = os.path.join(BASE_DIR, "static") # Define static directory
INDEX_HTML_PATH = os.path.join(BASE_DIR, "index.html")
FAVICON_PATH = os.path.join(STATIC_DIR, "favicon.ico") # Path to favicon

os.makedirs(TMP_DIR,    exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(LOG_DIR,    exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True) # Create static directory if it doesn't exist

# ... (video_recorder setup) ...
video_recorder = VideoRecorder(config=VideoConfig())

# ┌─────────────────────────────────────────────────────┐
# │                FastAPI 및 라우터                    │
# └─────────────────────────────────────────────────────┘
app = FastAPI(title="Unified Interview API")

# --- Favicon Endpoint ---
@app.get("/favicon.ico", include_in_schema=False) # include_in_schema=False to hide from API docs
async def favicon():
    if os.path.exists(FAVICON_PATH):
        return FileResponse(FAVICON_PATH, media_type="image/vnd.microsoft.icon")
    else:
        # You could return a default 204 No Content if no favicon is present
        # return Response(status_code=204)
        # Or, for now, let it be a 404 if file doesn't exist, but log it.
        print(f"Favicon not found at {FAVICON_PATH}, browser will get 404.")
        # To strictly return 404 from here if not found:
        raise HTTPException(status_code=404, detail="Favicon not found.")


# --- 루트 엔드포인트 ---
@app.get("/", response_class=HTMLResponse, tags=["Root"])
async def read_index():
    # ... (rest of your read_index function) ...
    if not os.path.exists(INDEX_HTML_PATH):
        raise HTTPException(status_code=404, detail="index.html not found")
    with open(INDEX_HTML_PATH, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

# ... (rest of your ia_router and pose_router definitions) ...
ia_router = APIRouter(prefix="", tags=["InterviewCore"])

@ia_router.post("/parse_resume")
async def parse_resume(file: UploadFile = File(...)):
    tmp_filename = f"{uuid.uuid4()}_{file.filename}"
    tmp_filepath = os.path.join(TMP_DIR, tmp_filename)
    with open(tmp_filepath, "wb") as f: shutil.copyfileobj(file.file, f)
    try:
        data = ResumeJsonParser().parse_to_file(tmp_filepath)
    except Exception as e:
        if os.path.exists(tmp_filepath): os.remove(tmp_filepath)
        raise HTTPException(status_code=500, detail=f"파싱 실패: {str(e)}")
    finally:
        if os.path.exists(tmp_filepath): os.remove(tmp_filepath)
    return JSONResponse(content=data)

@ia_router.post("/generate_questions")
async def generate_questions(file: UploadFile = File(...)):
    tmp_filename = f"{uuid.uuid4()}_{file.filename}"
    tmp_filepath = os.path.join(TMP_DIR, tmp_filename)
    with open(tmp_filepath, "wb") as f: shutil.copyfileobj(file.file, f)
    try:
        qs = InterviewQuestionMaker().generate_questions(tmp_filepath)
    except Exception as e:
        if os.path.exists(tmp_filepath): os.remove(tmp_filepath)
        raise HTTPException(status_code=500, detail=f"질문 생성 실패: {str(e)}")
    finally:
        if os.path.exists(tmp_filepath): os.remove(tmp_filepath)
    return {"questions": qs}

@ia_router.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    tmp_filename = f"{uuid.uuid4()}_{file.filename}"
    tmp_filepath = os.path.join(TMP_DIR, tmp_filename)
    with open(tmp_filepath, "wb") as f: shutil.copyfileobj(file.file, f)
    try:
        words = STTClient().transcribe(tmp_filepath)
    except Exception as e:
        if os.path.exists(tmp_filepath): os.remove(tmp_filepath)
        raise HTTPException(status_code=500, detail=f"STT 실패: {str(e)}")
    finally:
        if os.path.exists(tmp_filepath): os.remove(tmp_filepath)
    return {"word_timestamps": words}

@ia_router.post("/audio_metrics")
async def audio_metrics(file: UploadFile = File(...)):
    tmp_filename = f"{uuid.uuid4()}_{file.filename}"
    tmp_filepath = os.path.join(TMP_DIR, tmp_filename)
    with open(tmp_filepath, "wb") as f: shutil.copyfileobj(file.file, f)
    
    silence = 0.0
    duration = 0.0
    stt = STTClient() # Assuming STTClient is properly defined
    try:
        wts = stt.transcribe(tmp_filepath)
        silence = calculate_silence_duration(wts) # Assuming this function is defined
        duration = calculate_audio_duration(tmp_filepath) # Assuming this function is defined
    except Exception as e:
        print(f"Warning: STT or silence calculation failed: {e}, trying to get duration.")
        try:
            duration = calculate_audio_duration(tmp_filepath)
        except Exception as dur_e:
            if os.path.exists(tmp_filepath): os.remove(tmp_filepath)
            raise HTTPException(status_code=500, detail=f"오디오 메트릭 계산 실패 (duration error): {str(dur_e)}")
    finally:
        if os.path.exists(tmp_filepath): os.remove(tmp_filepath)
        
    return {"duration_sec": duration, "silence_sec": silence}


@ia_router.post("/evaluate")
async def evaluate_endpoint(
    background: BackgroundTasks,
    questions: List[str] = Form(...),
    answers:   List[str] = Form(...),
    audio_files: List[UploadFile] = File(...),
    output_file_name: str = Form("interview_evaluation.txt")
):
    if os.path.sep in output_file_name or ".." in output_file_name:
        raise HTTPException(status_code=400, detail="Invalid output file name.")
    
    final_output_path = os.path.join(LOG_DIR, output_file_name)

    temp_audio_paths = []
    try:
        for af_idx, af in enumerate(audio_files):
            # Ensure unique temp filenames even if original filenames are the same
            p = os.path.join(TMP_DIR, f"{uuid.uuid4()}_{af_idx}_{af.filename}")
            with open(p, "wb") as f: shutil.copyfileobj(af.file, f)
            temp_audio_paths.append(p)
        
        evaluate_and_save_responses(questions, answers, temp_audio_paths, final_output_path)
        
        if not os.path.exists(final_output_path):
             raise HTTPException(status_code=500, detail="평가 파일 생성 실패.")

        with open(final_output_path, "r", encoding="utf-8") as f:
            content = f.read()
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"평가 또는 결과 읽기 실패: {str(e)}")
    finally:
        for p in temp_audio_paths:
            if os.path.exists(p):
                os.remove(p)
    
    return PlainTextResponse(content, media_type="text/plain")


@ia_router.post("/video/start")
async def start_video():
    video_recorder.output_file = os.path.join(UPLOAD_DIR, f"server_recording_{uuid.uuid4().hex}.avi")
    video_recorder.start_recording()
    return {"status": "video recording started"}

@ia_router.post("/video/stop")
async def stop_video():
    if not video_recorder.is_recording(): # Assuming VideoRecorder has an is_recording method
        return {"status": "video recording was not active or already stopped."}
    video_recorder.stop_recording()
    return {"status": f"video recording stopped, saved to {video_recorder.output_file}"}

@ia_router.post("/audio/record_and_transcribe")
async def record_and_transcribe_audio(output_file_basename: str = Form("response.wav")):
    if os.path.sep in output_file_basename or ".." in output_file_basename:
        raise HTTPException(status_code=400, detail="Invalid output file name.")
    
    wav_output_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}_{output_file_basename}")
    
    rec = AudioRecorder() # Assuming AudioRecorder is defined
    actual_wav_path = rec.record(output_file=wav_output_path) 
    
    if not actual_wav_path or not os.path.exists(actual_wav_path):
        raise HTTPException(status_code=504, detail="녹음 실패: 음성 없음 또는 파일 저장 실패")
    
    try:
        words = STTClient().transcribe(actual_wav_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT 실패: {str(e)}")
    
    return {"wav_file_server_path": actual_wav_path, "word_timestamps": words}


pose_router = APIRouter(prefix="/pose", tags=["PoseAnalysis"])

@pose_router.post(
    "/analyze",
    summary="면접 영상 Pose 분석 → 로그 반환 (JSON with vid_id and log)"
)
async def pose_analyze(file: UploadFile = File(...)):
    vid_id = uuid.uuid4().hex
    safe_filename_base = "".join(c if c.isalnum() or c in ['.', '_'] else '_' for c in file.filename)
    fname  = f"{vid_id}_{safe_filename_base}"
    vpath  = os.path.join(UPLOAD_DIR, fname)
    
    with open(vpath, "wb") as f: shutil.copyfileobj(file.file, f)
    
    logp = os.path.join(LOG_DIR, f"{vid_id}.txt")
    
    try:
        analyze_video(vpath, logp)
        if not os.path.exists(logp):
            if os.path.exists(vpath): os.remove(vpath)
            raise HTTPException(status_code=500, detail="분석 스크립트가 로그 파일을 생성하지 않았습니다.")
            
        with open(logp, "r", encoding="utf-8") as f:
            log_content = f.read()
        
        return JSONResponse(content={"video_id": vid_id, "log_content": log_content})

    except Exception as e:
        if os.path.exists(vpath): os.remove(vpath)
        if os.path.exists(logp): os.remove(logp)
        raise HTTPException(status_code=500, detail=f"분석 오류: {str(e)}")


@pose_router.get(
    "/log/{video_id}",
    response_class=FileResponse,
    summary="분석 로그 다운로드"
)
async def pose_get_log(video_id: str):
    if not video_id.isalnum() or ".." in video_id or os.path.sep in video_id:
        raise HTTPException(status_code=400, detail="Invalid video ID format.")
        
    fp = os.path.join(LOG_DIR, f"{video_id}.txt")
    if not os.path.exists(fp):
        raise HTTPException(status_code=404, detail="로그를 찾을 수 없습니다.")
    return FileResponse(fp, media_type="text/plain", filename=f"{video_id}.txt")


app.include_router(ia_router)
app.include_router(pose_router)


if __name__ == "__main__":
    import uvicorn
    print(f"Unified API starting. Access the UI at http://localhost:8000/")
    print(f"Index.html expected at: {INDEX_HTML_PATH}")
    if not os.path.exists(INDEX_HTML_PATH):
        print(f"WARNING: index.html not found at {INDEX_HTML_PATH}. The root endpoint '/' will fail.")
    if not os.path.exists(FAVICON_PATH):
        print(f"WARNING: favicon.ico not found at {FAVICON_PATH}. Requests to /favicon.ico will result in 404.")
    uvicorn.run("unified_api:app", host="0.0.0.0", port=8000, reload=True)