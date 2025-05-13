import requests
import time
import os

SERVER_URL = "http://localhost:8000"
USER_ID = "test-user-001"
RESUME_ID_BASE = "test-vid-resume-" # 각 테스트 실행 시 고유 ID를 위해 timestamp 추가 가능

# !!! 중요 !!!
# 아래 VIDEO_FILE_PATH에 실제 유효한 MP4 파일의 경로를 입력하세요.
# 예: VIDEO_FILE_PATH = "sample.mp4" (스크립트와 같은 폴더에 sample.mp4가 있는 경우)
# 예: VIDEO_FILE_PATH = "/path/to/your/video/sample.mp4" (절대 경로)
VIDEO_FILE_PATH = "app/tests/test_interview.mp4" # 이 파일을 실제 MP4 파일로 교체하거나 경로 수정

# 0. (선택 사항) 이력서 텍스트 업로드 및 resume_id 확보
def register_resume():
    # 실행 시마다 다른 resume_id를 사용하도록 간단히 timestamp 추가
    current_resume_id = f"{RESUME_ID_BASE}{int(time.time())}"
    payload = {"text": f"Test resume for {current_resume_id}"}
    response = requests.post(f"{SERVER_URL}/upload_resume_text", json=payload)
    if response.status_code == 200:
        response_data = response.json()
        server_resume_id = response_data.get("resume_id")
        if server_resume_id:
            print(f"Resume registered. Server RESUME_ID: {server_resume_id}")
            return server_resume_id # 서버에서 생성된 ID 사용
        else:
            print("Error: resume_id not found in /upload_resume_text response.")
            return None
    else:
        print(f"Resume registration failed. Status: {response.status_code}, Response: {response.text}")
        return None

# 1. 영상 업로드 요청
def upload_video(current_resume_id: str):
    if not current_resume_id:
        print("Cannot upload video without a valid resume_id.")
        return None

    # VIDEO_FILE_PATH 유효성 검사
    if not os.path.exists(VIDEO_FILE_PATH):
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"Error: Test video file NOT FOUND at '{os.path.abspath(VIDEO_FILE_PATH)}'")
        print(f"Please update 'VIDEO_FILE_PATH' in the script with a valid MP4 file path.")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return None
    if os.path.getsize(VIDEO_FILE_PATH) < 1000: # 매우 작은 파일 (아마도 유효하지 않음)
         print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
         print(f"Warning: Test video file '{os.path.abspath(VIDEO_FILE_PATH)}' is very small.")
         print(f"Please ensure it is a valid MP4 video file.")
         print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


    try:
        with open(VIDEO_FILE_PATH, "rb") as video_file_object:
            files = {
                "file": (os.path.basename(VIDEO_FILE_PATH), video_file_object, "video/mp4")
            }
            data = {
                "user_id": USER_ID,
                "resume_id": current_resume_id
            }

            print(f"\nUploading video '{VIDEO_FILE_PATH}' for USER_ID: {USER_ID}, RESUME_ID: {current_resume_id}...")
            response = requests.post(f"{SERVER_URL}/upload_video", files=files, data=data)

        print(f"Upload Response Status: {response.status_code}")
        try:
            print(f"Upload Response JSON: {response.json()}")
            return response.json()
        except requests.exceptions.JSONDecodeError:
            print(f"Upload Response Text (Non-JSON): {response.text}")
            return {"detail": response.text, "status_code": response.status_code} # 에러 시에도 유사한 구조로 반환

    except FileNotFoundError: # 이 부분은 위 os.path.exists로 이미 처리됨
        print(f"Error: Video file '{VIDEO_FILE_PATH}' not found during upload attempt.")
        return None
    except Exception as e:
        print(f"An error occurred during video upload: {e}")
        return None

# 2. 분석 로그 파일 확인 요청
def get_log(current_resume_id: str):
    if not current_resume_id:
        print("Cannot get log without a valid resume_id.")
        return

    print(f"\nRequesting log for RESUME_ID: {current_resume_id}...")
    response = requests.get(f"{SERVER_URL}/get_log/{current_resume_id}")
    print(f"Get Log Response Status: {response.status_code}")

    if response.status_code == 200:
        print("Log Content (first 500 chars):")
        print(response.text)
        # with open(f"{current_resume_id}_log.txt", "w", encoding="utf-8") as f:
        #     f.write(response.text)
        # print(f"Log saved to {current_resume_id}_log.txt")
    elif response.status_code == 404:
        try:
            print(f"Log not found (404): {response.json()}")
        except requests.exceptions.JSONDecodeError:
            print(f"Log not found (404) and non-JSON response: {response.text}")
    else:
        print(f"Error getting log. Status: {response.status_code}, Response: {response.text}")

if __name__ == "__main__":
    # --- 가장 먼저, 실제 비디오 파일이 있는지 확인 ---
    if not os.path.exists(VIDEO_FILE_PATH):
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"CRITICAL ERROR: Test video file NOT FOUND at '{os.path.abspath(VIDEO_FILE_PATH)}'")
        print(f"Please set the 'VIDEO_FILE_PATH' variable in this script to a real .mp4 file.")
        print(f"Aborting test.")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        exit(1) # 실제 파일 없으면 테스트 중단

    print("--- Test: Register Resume ---")
    current_test_resume_id = register_resume()

    if current_test_resume_id:
        print(f"\n--- Test: Upload Video (RESUME_ID: {current_test_resume_id}) ---")
        upload_response = upload_video(current_test_resume_id)

        # 업로드 응답이 있고, 상태 코드가 200 OK인지 확인
        if upload_response and upload_response.get("status") == "ok": # main.py의 성공 응답 형식에 따름
            analysis_time_seconds = 20
            print(f"\nAnalysis likely in progress. Waiting for {analysis_time_seconds} seconds before fetching the log...")
            time.sleep(analysis_time_seconds)

            print(f"\n--- Test: Get Log (RESUME_ID: {current_test_resume_id}) ---")
            get_log(current_test_resume_id)
        elif upload_response:
            print(f"Video upload or analysis initiation failed. Server message: {upload_response.get('detail', 'No specific detail')}")
        else:
            print("Video upload function returned None or an unexpected error, skipping log retrieval.")
    else:
        print("Resume registration failed. Skipping video upload and log retrieval.")

    print("\n--- Test Script Finished ---")