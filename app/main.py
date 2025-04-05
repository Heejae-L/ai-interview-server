from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil, os
import fitz  # PyMuPDF
from question_generator import generate_questions
import uuid
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.responses import HTMLResponse



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

RESUME_DIR = "resumes"
VIDEO_DIR = "videos"
os.makedirs(RESUME_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

questions_storage = {}
result_storage = {}

app.mount("/static", StaticFiles(directory="static"), name="static")
@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()
    

@app.get("/favicon.ico")
async def favicon():
    return FileResponse("static/favicon.ico")
@app.post("/upload_resume")

async def upload_resume(file: UploadFile = File(...)):
    resume_id = str(uuid.uuid4())
    file_path = f"{RESUME_DIR}/{resume_id}.pdf"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 여기에서 텍스트 추출하지 말고 그냥 경로 넘기기
    questions = generate_questions(file_path)

    questions_storage[resume_id] = questions
    return {"resume_id": resume_id, "questions": questions}

@app.get("/get_questions/{resume_id}")
async def get_questions(resume_id: str):
    return {"questions": questions_storage.get(resume_id, [])}

@app.post("/upload_video")
async def upload_video(resume_id: str = Form(...), file: UploadFile = File(...)):
    video_path = f"{VIDEO_DIR}/{resume_id}.webm"
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 자세 분석 호출
    from subprocess import run
    run(["python3", "0317_correct_pose_detection_video.py", video_path])

    # 결과 저장 예시 (단순 텍스트)
    feedback = f"분석 완료 - {resume_id} 자세가 양호함"
    result_storage[resume_id] = feedback
    return {"status": "ok"}

@app.get("/get_result/{resume_id}")
async def get_result(resume_id: str):
    return {"result": result_storage.get(resume_id, "분석 결과 없음")}
