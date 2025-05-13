from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import shutil
import os
import uuid
from subprocess import run, CalledProcessError # CalledProcessError 추가
# from question_generator import generate_questions # 사용하지 않는다면 주석 처리 또는 삭제
# from question_generator import generate_questions_from_text # 사용하지 않는다면 주석 처리 또는 삭제
from question_generator import generate_static_questions

# 앱 생성
app = FastAPI()

# CORS 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 폴더 준비
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # main.py 파일의 현재 경로
RESUME_DIR = os.path.join(BASE_DIR, "resumes")
VIDEO_DIR = os.path.join(BASE_DIR, "videos")
LOG_DIR = os.path.join(BASE_DIR, "analysis_logs") # 로그 파일 저장 디렉토리

os.makedirs(RESUME_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True) # 로그 디렉토리 생성

# 임시 저장소 (세션 대용) - resume_id를 키로, 질문과 로그 파일 경로를 값으로 저장
interview_data_storage = {} # questions_storage와 result_storage 통합 및 변경

# 정적 파일 서빙
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

# 루트 접근 시 index.html 제공
@app.get("/", response_class=HTMLResponse)
async def read_root():
    index_html_path = os.path.join(BASE_DIR, "index.html")
    if not os.path.exists(index_html_path):
        raise HTTPException(status_code=404, detail="index.html not found")
    with open(index_html_path, "r", encoding="utf-8") as f:
        return f.read()

@app.get("/favicon.ico", include_in_schema=False) # API 문서에 불필요하게 노출되지 않도록 설정
async def favicon():
    favicon_path = os.path.join(BASE_DIR, "static", "favicon.ico")
    if not os.path.exists(favicon_path):
        raise HTTPException(status_code=404, detail="favicon.ico not found")
    return FileResponse(favicon_path)

# 텍스트 기반 이력서 업로드
class ResumeText(BaseModel):
    text: str

@app.post("/upload_resume_text")
async def upload_resume_text(payload: ResumeText):
    resume_id = str(uuid.uuid4())
    print(f"[Static Resume Received] ID: {resume_id}")
    questions = generate_static_questions() # 이력서 내용과 무관하게 정적 질문 생성
    interview_data_storage[resume_id] = {"questions": questions, "log_file_path": None}
    return {"status": "received", "resume_id": resume_id, "questions": questions}

# 질문 조회
@app.get("/get_questions/{resume_id}")
async def get_questions(resume_id: str):
    if resume_id not in interview_data_storage:
        raise HTTPException(status_code=404, detail="Resume ID not found")
    return {"questions": interview_data_storage[resume_id].get("questions", [])}

# 인터뷰 영상 업로드
@app.post("/upload_video")
async def upload_video(
    user_id: str = Form(...), # 사용자 ID 추가
    resume_id: str = Form(...),
    file: UploadFile = File(...)
):
    if resume_id not in interview_data_storage:
        raise HTTPException(status_code=404, detail="Resume ID not found. Please upload resume first.")

    # 영상 파일명 및 경로 설정 (사용자ID_resumeID.mp4)
    video_filename = f"{user_id}_{resume_id}.mp4"
    video_path = os.path.join(VIDEO_DIR, video_filename)

    # 로그 파일명 및 경로 설정 (사용자ID_resumeID_log.txt)
    log_filename = f"{user_id}_{resume_id}_log.txt"
    log_path = os.path.join(LOG_DIR, log_filename)

    # 이전 로그 파일이 있다면 삭제 (새 분석을 위해)
    if os.path.exists(log_path):
        os.remove(log_path)

    try:
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save video file: {str(e)}")
    finally:
        file.file.close() # 파일 핸들러 명시적 종료

    # 자세 분석 스크립트 실행
    # python3 대신 sys.executable을 사용하여 현재 파이썬 인터프리터 경로를 명시적으로 지정할 수 있습니다.
    # 이는 가상환경 등에서 발생할 수 있는 문제를 줄여줍니다.
    pose_analysis_script_path = os.path.join(BASE_DIR, "0317_correct_pose_detection_video.py") # 분석 스크립트 경로
    
    # 분석 스크립트가 존재하는지 확인
    if not os.path.exists(pose_analysis_script_path):
        raise HTTPException(status_code=500, detail="Pose analysis script not found.")

    try:
        # subprocess.run에 전체 경로를 제공하고, 로그 파일 경로를 인자로 전달
        process_result = run(
            ["python3", pose_analysis_script_path, video_path, log_path],
            capture_output=True, # stdout, stderr 캡처
            text=True, # 텍스트 모드로 디코딩
            check=True # 반환 코드가 0이 아니면 CalledProcessError 발생
        )
        print(f"Analysis script stdout: {process_result.stdout}") # 분석 스크립트 출력 확인 (디버깅용)

    except CalledProcessError as e:
        print(f"Error during pose analysis script execution: {e}")
        print(f"Stderr: {e.stderr}")
        raise HTTPException(status_code=500, detail=f"Error during pose analysis: {e.stderr or 'Unknown error'}")
    except FileNotFoundError:
         raise HTTPException(status_code=500, detail="Python interpreter 'python3' not found or pose analysis script not found at the specified path.")


    # 분석 결과 로그 파일 경로 저장
    if os.path.exists(log_path) and os.path.getsize(log_path) > 0:
        interview_data_storage[resume_id]["log_file_path"] = log_path
        return {"status": "ok", "message": "Video uploaded and analysis started.", "log_file_identifier": log_filename}
    else:
        # 로그 파일이 생성되지 않았거나 비어있는 경우
        # (analyze_mistake_log 함수는 이제 클라이언트가 로그를 직접 요청할 때 사용되므로 여기서 호출하지 않음)
        interview_data_storage[resume_id]["log_file_path"] = None # 로그 파일 없음 표시
        # 여기서 사용자에게 "분석은 완료되었으나, 감지된 자세 오류가 없습니다." 와 같은 메시지를 줄 수도 있습니다.
        # 또는 로그 파일이 아예 생성 안된 경우 에러로 처리할 수도 있습니다.
        # 현재는 로그 파일이 없으면 아래 get_log에서 "Log not found or empty"로 처리됩니다.
        return {"status": "ok", "message": "Video uploaded and analysis completed, but no specific issues logged or log file is empty."}


# 분석 로그 파일 불러오기 (이전의 get_result 대체)
@app.get("/get_log/{resume_id}")
async def get_log_file(resume_id: str):
    if resume_id not in interview_data_storage:
        raise HTTPException(status_code=404, detail="Resume ID not found.")

    log_file_path = interview_data_storage[resume_id].get("log_file_path")

    if not log_file_path or not os.path.exists(log_file_path): # 경로가 없거나, 실제 파일이 없는 경우
        # 이전에는 "분석 결과 없음"으로 나왔지만, 이제는 로그 파일이 없는 경우를 명확히 합니다.
        # 로그 파일이 비어있는 경우도 고려할 수 있습니다. (os.path.getsize(log_file_path) == 0)
        return JSONResponse(
            status_code=404,
            content={"message": "Log file not found for this resume_id. Please ensure the video has been uploaded and processed."}
        )
    
    # 클라이언트가 요청한 파일이므로 log.txt로 다운로드되도록 filename 지정
    return FileResponse(log_file_path, media_type="text/plain", filename=f"{resume_id}_log.txt")

# (선택 사항) 요약된 피드백을 원한다면 analyze_mistake_log 함수를 유지하고 별도 엔드포인트로 제공
@app.get("/get_feedback_summary/{resume_id}")
async def get_feedback_summary(resume_id: str):
    if resume_id not in interview_data_storage:
        raise HTTPException(status_code=404, detail="Resume ID not found.")

    log_file_path = interview_data_storage[resume_id].get("log_file_path")

    if not log_file_path or not os.path.exists(log_file_path):
        return JSONResponse(
            status_code=404,
            content={"message": "Log file not found. Cannot generate summary."}
        )
    
    # analyze_mistake_log 함수가 로그 파일 경로를 인자로 받도록 수정
    feedback = analyze_mistake_log_from_file(log_file_path)
    return {"resume_id": resume_id, "feedback_summary": feedback}

# 로그 파일로부터 피드백 생성 (기존 analyze_mistake_log 수정)
def analyze_mistake_log_from_file(log_file_path: str) -> str:
    if not os.path.exists(log_file_path):
        return "분석 로그 파일을 찾을 수 없습니다." # Log file not found

    with open(log_file_path, "r", encoding="utf-8") as f: # UTF-8 인코딩 명시
        lines = f.readlines()

    if not lines: # 로그 파일은 존재하나 내용이 없는 경우
        return "자세가 매우 양호합니다! (또는 로그 내용 없음)" # Excellent posture! (Or log content is empty)

    unique_mistakes = set()
    for line in lines:
        if ':' in line: # "시간: 메시지" 형식으로 가정
            try:
                unique_mistakes.add(line.strip().split(": ", 1)[1]) # 콜론 첫 등장으로 분리
            except IndexError:
                print(f"Warning: Could not parse line in log: {line.strip()}") # 파싱 오류 시 경고
                unique_mistakes.add(line.strip()) # 파싱 실패 시 원본 라인 추가 (혹은 다른 처리)
    
    if not unique_mistakes: # 파싱 후 유니크한 메시지가 없는 경우
        return "자세 분석 로그에서 특이사항을 찾지 못했습니다." # No specific issues found in pose analysis log.

    feedback = "다음과 같은 자세 문제가 감지되었습니다:\n" + "\n".join(f"- {m}" for m in sorted(list(unique_mistakes))) # 정렬된 결과
    return feedback