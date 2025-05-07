import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_upload_resume_text_returns_static_questions():
    # 이력서 내용
    sample_resume_text = {
        "text": "저는 컴퓨터공학을 전공하고 다양한 프로젝트 경험이 있습니다."
    }

    response = client.post("/upload_resume_text", json=sample_resume_text)

    # 응답 상태 확인
    assert response.status_code == 200

    data = response.json()

    # 필드 존재 확인
    assert "resume_id" in data
    assert "questions" in data
    assert isinstance(data["questions"], list)
    assert len(data["questions"]) > 0

    # 질문 내용 확인 (기본 질문 중 일부 예시 포함)
    expected_keywords = ["자기소개", "프로젝트", "협업", "기술", "동기"]
    matched = any(any(keyword in q for keyword in expected_keywords) for q in data["questions"])
    assert matched, "기본 질문 세트가 포함되어 있어야 함"
