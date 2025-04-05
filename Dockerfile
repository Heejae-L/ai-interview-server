# ▶ 베이스 이미지: Python 3.11
FROM python:3.11-slim

# ▶ 필수 패키지 설치 (apt) — OpenCV, ffmpeg, pyaudio 등 의존성
RUN apt-get update && apt-get install -y \
    gcc \  
    portaudio19-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

# ▶ 작업 디렉토리 설정
WORKDIR /app

# ▶ 프로젝트 파일 복사
COPY ./app /app

# ▶ 필요 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ▶ FastAPI 앱 실행 (uvicorn)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# .env 파일 복사
COPY .env .env