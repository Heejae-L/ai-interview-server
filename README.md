# 🧠 Unified Interview AI API

이 프로젝트는 AI 면접관 시스템을 위한 백엔드 API입니다.  
이력서 기반 질문 생성부터 음성 및 자세 분석을 통해 피드백을 제공하는 통합 서버입니다.

---

## 📁 프로젝트 구조

```
unified_api/
├── unified_api.py          # 메인 FastAPI 애플리케이션
├── interview_app/          # 이력서 분석, 질문 생성, STT, 녹음 모듈 등
├── pose_detection/         # MediaPipe 기반 자세 분석 모듈
├── uploads/                # 업로드된 파일 저장 경로
├── logs/                   # 분석 결과 텍스트 저장 경로
├── tmp/                    # 임시 파일 저장 디렉터리
├── static/
│   └── favicon.ico         # 파비콘
└── index.html              # 클라이언트 뷰 (루트 페이지)
```

---

## 🚀 실행 방법

### 1. 환경 세팅

```bash
pip install fastapi uvicorn
# 추가적으로 필요한 패키지는 interview_app/ 및 pose_detection/에서 정의
```

### 2. 실행

```bash
python unified_api.py
# 또는
uvicorn unified_api:app --reload
```

서버 실행 후 [http://localhost:8000](http://localhost:8000) 에 접속 가능

---

## 🌐 주요 API

### 🔹 루트 및 정적 파일

| Method | Endpoint         | 설명                      |
|--------|------------------|---------------------------|
| GET    | `/`              | index.html 반환           |
| GET    | `/favicon.ico`   | 파비콘 반환               |

---

### 📄 Interview 관련 (prefix 없음)

| Method | Endpoint               | 설명                                |
|--------|------------------------|-------------------------------------|
| POST   | `/parse_resume`        | PDF 이력서를 업로드하여 JSON으로 변환 |
| POST   | `/generate_questions`  | 이력서를 기반으로 면접 질문 생성   |
| POST   | `/transcribe`          | 음성 파일 STT 변환                  |
| POST   | `/audio_metrics`       | 오디오 총 길이 및 침묵 구간 분석    |
| POST   | `/evaluate`            | 질문/답변 + 오디오 평가 및 결과 저장 |
| POST   | `/video/start`         | 서버 웹캠 영상 녹화 시작            |
| POST   | `/video/stop`          | 녹화 종료 및 경로 반환              |
| POST   | `/audio/record_and_transcribe` | 마이크로 녹음 + STT 분석    |

---

### 🧍 Pose 분석 관련 (`/pose` prefix)

| Method | Endpoint                  | 설명                               |
|--------|---------------------------|------------------------------------|
| POST   | `/pose/analyze`           | 면접 영상 업로드 및 자세 분석      |
| GET    | `/pose/log/{video_id}`    | 분석 로그 다운로드 (텍스트 파일)   |

---

## 🔒 보안

- 업로드 파일명 검증: `..` 또는 경로 분리자 포함 시 거부
- `uuid` 기반 파일명으로 충돌 방지
- 임시/업로드/로그 디렉터리는 자동 생성됨

---

## 💡 개선 예정 기능

- WebSocket 기반 실시간 분석
- 자세 분석 결과의 시각화
- 사용자 히스토리 저장
- LLM 기반 질문 평가 점수화

---

## 🛠️ 사용 기술

- **FastAPI** - Python 기반 비동기 웹 프레임워크
- **MediaPipe** - 자세 인식 (Google)
- **OpenCV** - 비디오 녹화 및 처리
- **STTClient** - 음성 인식 모듈 (ex. Whisper 등)
- **LLM (예: Ollama)** - 이력서 기반 질문 생성
- **Uvicorn** - ASGI 서버

---

## 🙋‍♂️ 문의

이 프로젝트는 연구 및 데모 목적이며,  
기능 제안 또는 오류 제보는 GitHub Issues를 통해 환영합니다.
