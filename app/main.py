from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import shutil, os, uuid
from question_generator import generate_questions
from subprocess import run
from question_generator import generate_questions_from_text

from question_generator import generate_static_questions

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
    print(f"[Static Resume Received] ID: {resume_id}")
    questions = generate_static_questions()
    questions_storage[resume_id] = questions
    return {"status": "received", "resume_id": resume_id, "questions": questions}
# 질문 조회
@app.get("/get_questions/{resume_id}")
async def get_questions(resume_id: str):
    return {"questions": questions_storage.get(resume_id, [])}

# 분석 결과 불러오기
def analyze_mistake_log(video_path: str) -> str:
    log_file = os.path.join("app", "logs", "mistakes_log.txt")
    if not os.path.exists(log_file):
        return "분석 실패 또는 오류 발생"

    with open(log_file, "r") as f:
        lines = f.readlines()

    if len(lines) == 0:
        return "자세가 매우 양호합니다!"

    unique_mistakes = set()
    for line in lines:
        if ':' in line:
            unique_mistakes.add(line.strip().split(": ")[1])
    
    feedback = "다음과 같은 자세 문제가 감지되었습니다:\n" + "\n".join(f"- {m}" for m in unique_mistakes)
    return feedback

# 인터뷰 영상 업로드
@app.post("/upload_video")
async def upload_video(resume_id: str = Form(...), file: UploadFile = File(...)):
    video_path = f"{VIDEO_DIR}/{resume_id}.webm"
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 자세 분석 실행
    run(["python3", "0317_correct_pose_detection_video.py", video_path])

    # 분석 결과 로그에서 피드백 생성
    feedback = analyze_mistake_log(video_path)
    result_storage[resume_id] = feedback

    return {"status": "ok"}


# 분석 결과 확인
@app.get("/get_result/{resume_id}")
async def get_result(resume_id: str):
    return {"result": result_storage.get(resume_id, "분석 결과 없음")}
