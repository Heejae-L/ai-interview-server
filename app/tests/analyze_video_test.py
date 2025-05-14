import requests
import time
import os

# --- 설정 변수 ---
# 실제 운영 중인 FastAPI 서버 주소로 변경하세요.
# 로컬 테스트 시: "http://localhost:8000"
# Render 배포 시: "https://your-render-app-name.onrender.com"
SERVER_URL = "http://localhost:8000" # 예시 URL, 실제 서버 주소로 변경

# 테스트에 사용할 사용자 ID (임의 지정)
TEST_USER_ID = "testuser_for_reanalysis_001"

CLIENT_VIDEO_FILE_PATH = "test_interview.mp4" # 실제 파일 경로로 수정

# --- 도우미 함수 ---
def create_dummy_sample_video_if_not_exists(filepath):
    """테스트용 샘플 비디오 파일이 없으면 간단한 내용을 가진 파일을 생성합니다."""
    if not os.path.exists(filepath):
        print(f"경고: 테스트 비디오 파일 '{filepath}'을(를) 찾을 수 없습니다. 더미 파일을 생성합니다.")
        try:
            # 실제 비디오 내용은 아니지만, 파일은 존재하도록 만듭니다.
            # 제대로 된 테스트를 위해서는 실제 비디오 파일을 사용해야 합니다.
            with open(filepath, "wb") as f:
                f.write(b"This is a dummy file for testing video upload.")
            print(f"더미 비디오 파일 '{filepath}'이(가) 생성되었습니다. 실제 비디오 파일로 교체하는 것이 좋습니다.")
        except IOError as e:
            print(f"오류: 더미 비디오 파일 '{filepath}' 생성 실패: {e}")
            print("테스트를 진행하려면 유효한 비디오 파일이 필요하거나 파일 권한을 확인하세요.")
            exit(1)

def step_print(message):
    print(f"\n--- 단계: {message} ---")

# --- 테스트 실행 ---
if __name__ == "__main__":
    # 0. 테스트용 샘플 비디오 파일 준비 (없으면 더미 생성)
    create_dummy_sample_video_if_not_exists(CLIENT_VIDEO_FILE_PATH)

    current_resume_id = None

    try:
        # 1. 이력서 제출 및 resume_id 확보 (세션 생성 목적)
        step_print("이력서 제출 및 resume_id 확보")
        resume_payload = {"text": "이것은 기존 영상 분석 테스트를 위한 샘플 이력서입니다."}
        response = requests.post(f"{SERVER_URL}/upload_resume_text", json=resume_payload)
        response.raise_for_status() # 오류 발생 시 예외 발생
        resume_data = response.json()
        current_resume_id = resume_data.get("resume_id")
        if not current_resume_id:
            raise ValueError("/upload_resume_text 응답에서 resume_id를 찾을 수 없습니다.")
        print(f"성공: Resume ID '{current_resume_id}' 발급받음.")
        print(f"질문: {resume_data.get('questions')}")

        # 2. 테스트 영상 업로드 (서버에 분석 대상 영상 파일 생성 목적)
        #    이 부분은 /analyze_video/{resume_id} 테스트의 사전 조건입니다.
        #    main.py에 /upload_video 엔드포인트가 있어야 합니다.
        step_print(f"테스트 영상 업로드 (User ID: {TEST_USER_ID}, Resume ID: {current_resume_id})")
        if not os.path.exists(CLIENT_VIDEO_FILE_PATH):
             print(f"오류: 클라이언트 비디오 파일 '{CLIENT_VIDEO_FILE_PATH}'을 찾을 수 없어 업로드를 건너뜁니다.")
             print("서버에 수동으로 '{TEST_USER_ID}_{current_resume_id}.mp4' (또는 .webm 등) 파일을 준비해야 합니다.")
        else:
            files = {'file': (os.path.basename(CLIENT_VIDEO_FILE_PATH), open(CLIENT_VIDEO_FILE_PATH, 'rb'), 'video/mp4')} # MIME 타입은 파일에 맞게 조정
            video_upload_payload = {
                'user_id': TEST_USER_ID,
                'resume_id': current_resume_id
            }
            response = requests.post(f"{SERVER_URL}/upload_video", files=files, data=video_upload_payload)
            response.raise_for_status()
            upload_data = response.json()
            print(f"성공: 영상 업로드 완료. 서버 메시지: {upload_data.get('message')}")
            # 초기 분석이 있을 수 있으므로 잠시 대기 (선택적)
            print("초기 분석을 위해 잠시 대기 (10초)...")
            time.sleep(10)


        # 3. 저장된 영상 분석 요청 (/analyze_video/{resume_id} 테스트)
        step_print(f"저장된 영상 분석 요청 (Resume ID: {current_resume_id})")
        # 이 시점에는 서버 VIDEO_DIR에 TEST_USER_ID_current_resume_id.mp4 (또는 업로드된 확장자) 파일이 있어야 함
        analysis_response = requests.post(f"{SERVER_URL}/analyze_video/{current_resume_id}")
        analysis_response.raise_for_status()
        analysis_data = analysis_response.json()
        print(f"성공: 재분석 요청 완료. 서버 메시지: {analysis_data.get('message')}")
        print(f"분석된 비디오 (서버 유추): {analysis_data.get('analyzed_video')}")
        print(f"로그 파일 식별자: {analysis_data.get('log_file_identifier')}")

        # 4. 분석 결과 로그 확인 (재분석된 로그)
        step_print(f"재분석된 로그 파일 확인 (Resume ID: {current_resume_id})")
        # 분석 스크립트 실행 시간에 따라 충분히 대기해야 함
        log_wait_time = 30 # 초 단위, 영상 길이에 따라 조절 필요
        print(f"재분석 완료를 위해 {log_wait_time}초 대기...")
        time.sleep(log_wait_time)

        log_response = requests.get(f"{SERVER_URL}/get_log/{current_resume_id}")
        log_response.raise_for_status() # 200 OK가 아니면 예외 발생
        
        print(f"성공: 로그 파일 수신 완료 (HTTP Status: {log_response.status_code})")
        print("=" * 20 + " 로그 내용 시작 " + "=" * 20)
        print(log_response.text) # 로그 내용 전체 출력
        print("=" * 20 + "  로그 내용 끝  " + "=" * 20)
        
        if "--- 분석 결과 요약 ---" in log_response.text:
            print("\n테스트 확인: 로그 파일에 요약 정보가 포함되어 있습니다.")
        else:
            print("\n테스트 확인: 로그 파일에 요약 정보가 누락되었거나 다른 형식일 수 있습니다.")

    except requests.exceptions.HTTPError as http_err:
        print(f"\nHTTP 오류 발생: {http_err}")
        try:
            # 서버에서 JSON 형태의 상세 오류 메시지를 보냈을 경우 출력
            error_details = http_err.response.json()
            print(f"서버 상세 오류: {error_details}")
        except ValueError:
            # JSON 형태가 아닐 경우 일반 텍스트로 출력
            print(f"서버 응답 (텍스트): {http_err.response.text}")
    except Exception as e:
        print(f"\n예상치 못한 오류 발생: {e}")
    finally:
        print("\n--- 테스트 스크립트 종료 ---")