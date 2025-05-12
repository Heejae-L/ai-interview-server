import requests

SERVER_URL = "http://localhost:8000"  # FastAPI 서버 주소
RESUME_ID = "test-resume-123"  # 임의의 resume_id 사용
VIDEO_FILE_PATH = "app/tests/interview.mp4" # 테스트할 영상 파일 경로

# 1. 영상 업로드 요청
def upload_video():
    files = {
        "file": open(VIDEO_FILE_PATH, "rb")
    }
    data = {
        "resume_id": RESUME_ID
    }

    response = requests.post(f"{SERVER_URL}/upload_video", files=files, data=data)
    print("Upload Response:", response.json())

# 2. 분석 결과 확인 요청
def get_result():
    response = requests.get(f"{SERVER_URL}/get_result/{RESUME_ID}")
    print("Result Response:", response.json())

if __name__ == "__main__":
    upload_video()
    print("분석 중... (잠시 후 결과 확인)")
    import time
    time.sleep(5)  # 분석 시간 대기 (스크립트에 따라 조정)
    get_result()
