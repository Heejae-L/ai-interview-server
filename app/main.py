from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import shutil, os, uuid
from question_generator import generate_questions  # 텍스트 입력도 지원해야 함
from subprocess import run

# 앱 생성
app = FastAPI()

# CORS 허용 (모든 origin 허용 중)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 폴더 준비
RESUME_DIR = "resumes"
VIDEO_DIR = "videos"
os.makedirs(RESUME_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

# 임시 저장소 (세션 대용)
questions_storage = {}
result_storage = {}

# 정적 파일 서빙
app.mount("/static", StaticFiles(directory="static"), name="static")

# 루트 접근 시 index.html 제공
@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/favicon.ico")
async def favicon():
    return FileResponse("static/favicon.ico")

# 텍스트 기반 이력서 업로드
class ResumeText(BaseModel):
    text: str

@app.post("/upload_resume_text")
async def upload_resume_text(payload: ResumeText):
    resume_id = str(uuid.uuid4())
    print(f"[Resume Received] ID: {resume_id}")
    questions = generate_questions(payload.text)
    questions_storage[resume_id] = questions
    return {"status": "received", "resume_id": resume_id, "questions": questions}


# 질문 조회
@app.get("/get_questions/{resume_id}")
async def get_questions(resume_id: str):
    return {"questions": questions_storage.get(resume_id, [])}

# 인터뷰 영상 업로드
@app.post("/upload_video")
async def upload_video(resume_id: str = Form(...), file: UploadFile = File(...)):
    video_path = f"{VIDEO_DIR}/{resume_id}.webm"
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 자세 분석 스크립트 실행 (인자로 파일 경로 넘김)
    run(["python3", "0317_correct_pose_detection_video.py", video_path])

    # 분석 결과 저장
    feedback = f"분석 완료 - {resume_id} 자세가 양호함"
    result_storage[resume_id] = feedback
    return {"status": "ok"}

# 분석 결과 확인
@app.get("/get_result/{resume_id}")
async def get_result(resume_id: str):
    return {"result": result_storage.get(resume_id, "분석 결과 없음")}
