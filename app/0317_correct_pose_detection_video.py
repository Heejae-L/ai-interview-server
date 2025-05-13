import cv2
import mediapipe as mp
import numpy as np
import sys
import os
from collections import Counter

# MediaPipe Pose 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 흔들림 판단을 위한 기준 값 저장
previous_positions = None

# 잘못된 자세 로그 저장 함수
def log_mistakes_to_txt(mistakes, timestamp, log_file_path):
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir): # log_dir이 빈 문자열이 아니고, 존재하지 않을 때
        os.makedirs(log_dir, exist_ok=True)

    with open(log_file_path, "a", encoding="utf-8") as file:
        for mistake in mistakes:
            file.write(f"{timestamp:.2f} sec: {mistake}\n")

# 몸의 흔들림 판단 함수
def check_body_stability(landmarks, threshold=0.05):
    global previous_positions
    
    required_landmarks = [
        mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP
    ]
    for lm_idx in required_landmarks:
        if not (0 <= lm_idx < len(landmarks) and landmarks[lm_idx].visibility > 0.1):
            return None 

    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    
    current_positions = np.array([
        [left_shoulder.x, left_shoulder.y],
        [right_shoulder.x, right_shoulder.y],
        [left_hip.x, left_hip.y],
        [right_hip.x, right_hip.y]
    ])
    
    movement_detected = False
    if previous_positions is not None:
        if current_positions.shape == previous_positions.shape:
            movement = np.linalg.norm(current_positions - previous_positions, axis=1).mean()
            if movement > threshold:
                movement_detected = True
        else:
            pass 
    
    previous_positions = current_positions 
    
    if movement_detected:
        return "몸을 흔들고 있습니다."
    return None

# 다리 벌어짐 계산 함수
def check_knee_position(landmarks):
    required_landmarks = [
        mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE,
        mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER
    ]
    for lm_idx in required_landmarks:
        if not (0 <= lm_idx < len(landmarks) and landmarks[lm_idx].visibility > 0.1):
            return None

    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
    right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    
    if not (left_shoulder.visibility > 0.1 and right_shoulder.visibility > 0.1 and \
            left_knee.visibility > 0.1 and right_knee.visibility > 0.1):
        return None

    shoulder_width = np.abs(left_shoulder.x - right_shoulder.x)
    knee_distance = np.abs(left_knee.x - right_knee.x)
    
    if knee_distance > shoulder_width * 1.2: 
        return "다리를 너무 많이 벌리고 있습니다."
    return None

# 허리 기울어짐 계산 함수
def check_back_straightness(landmarks):
    required_landmarks = [
        mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP
    ]
    for lm_idx in required_landmarks:
        if not (0 <= lm_idx < len(landmarks) and landmarks[lm_idx].visibility > 0.1):
            return None

    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    
    shoulder_y_diff = abs(left_shoulder.y - right_shoulder.y)
    
    shoulder_mid_x = (left_shoulder.x + right_shoulder.x) / 2
    hip_mid_x = (left_hip.x + right_hip.x) / 2
    body_lean = abs(shoulder_mid_x - hip_mid_x)

    if shoulder_y_diff > 0.04 or body_lean > 0.06: 
        return "허리를 곧게 펴주세요." # 또는 "허리가 한쪽으로 기울어져 있습니다."
    return None

# 고개 좌우 기울어짐 계산 함수 (Roll)
def check_head_tilt(landmarks):
    required_landmarks = [
        mp_pose.PoseLandmark.LEFT_EAR, mp_pose.PoseLandmark.RIGHT_EAR,
    ]
    for lm_idx in required_landmarks:
        if not (0 <= lm_idx < len(landmarks) and landmarks[lm_idx].visibility > 0.1):
            return None
            
    left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR]
    right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR]

    ear_y_diff = abs(left_ear.y - right_ear.y)
    
    if ear_y_diff > 0.03: 
         return "고개가 옆으로 기울어져 있습니다."
    return None

# 시선 방향 추정 함수 (머리 방향 기반)
def estimate_gaze_direction(landmarks, image_width, image_height):
    required_landmarks = [
        mp_pose.PoseLandmark.NOSE,
        mp_pose.PoseLandmark.LEFT_EYE, mp_pose.PoseLandmark.RIGHT_EYE,
    ]
    for lm_idx in required_landmarks:
        if not (0 <= lm_idx < len(landmarks) and landmarks[lm_idx].visibility > 0.2):
             return "시선: 알 수 없음" 

    nose = landmarks[mp_pose.PoseLandmark.NOSE]
    left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE]
    right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE]

    eye_mid_x_norm = (left_eye.x + right_eye.x) / 2
    eye_mid_y_norm = (left_eye.y + right_eye.y) / 2
    nose_x_norm = nose.x
    nose_y_norm = nose.y
    
    # 임계값 (정규화된 좌표 기준, 실험으로 조정 필요)
    # 이 값은 얼굴 크기, 카메라와의 거리 등에 따라 민감할 수 있습니다.
    horizontal_threshold_norm = 0.04 # 코가 눈 중심에서 좌우로 벗어나는 정도
    vertical_threshold_norm = 0.03   # 코가 눈 중심에서 상하로 벗어나는 정도

    gaze_h_direction = "정면(좌우)"
    # 코가 눈 중심보다 (카메라 기준) 왼쪽에 있다면, 사용자는 오른쪽을 보고 있는 것
    if nose_x_norm < eye_mid_x_norm - horizontal_threshold_norm:
        gaze_h_direction = "오른쪽" 
    # 코가 눈 중심보다 (카메라 기준) 오른쪽에 있다면, 사용자는 왼쪽을 보고 있는 것
    elif nose_x_norm > eye_mid_x_norm + horizontal_threshold_norm:
        gaze_h_direction = "왼쪽"   

    gaze_v_direction = "정면(상하)"
    # 코가 눈 중심보다 (카메라 기준) 위쪽에 있다면(y값이 작음), 사용자는 위쪽을 보고 있는 것
    if nose_y_norm < eye_mid_y_norm - vertical_threshold_norm:
        gaze_v_direction = "위쪽"
    # 코가 눈 중심보다 (카메라 기준) 아래쪽에 있다면(y값이 큼), 사용자는 아래쪽을 보고 있는 것
    elif nose_y_norm > eye_mid_y_norm + vertical_threshold_norm:
        gaze_v_direction = "아래쪽"
    
    if gaze_h_direction == "정면(좌우)" and gaze_v_direction == "정면(상하)":
        return "시선: 정면"
    elif gaze_h_direction != "정면(좌우)" and gaze_v_direction == "정면(상하)":
        return f"시선: {gaze_h_direction}"
    elif gaze_h_direction == "정면(좌우)" and gaze_v_direction != "정면(상하)":
        return f"시선: {gaze_v_direction}"
    else: 
        return f"시선: {gaze_v_direction}-{gaze_h_direction}"


# 메인 실행 함수
def main(video_path, output_log_path):
    global previous_positions 
    previous_positions = None 

    print(f"[INFO] 분석 대상 영상: {video_path}")
    print(f"[INFO] 로그 저장 경로: {output_log_path}")

    # 새 분석 시작 시 이전 로그 파일 삭제
    if os.path.exists(output_log_path):
        os.remove(output_log_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] 영상을 열 수 없습니다: {video_path}", file=sys.stderr)
        sys.exit(1) 

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: 
        print("[WARNING] FPS가 0입니다. 기본값 30으로 설정합니다.", file=sys.stderr)
        fps = 30.0 # 부동소수점형으로
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = 0
    
    mistake_type_counts = Counter()
    gaze_direction_counts = Counter()
    total_frames_with_landmarks = 0 # 랜드마크가 감지된 총 프레임 수

    # GUI 창을 보여줄지 여부 (서버 환경에서는 False로)
    show_gui = os.environ.get("SHOW_POSE_GUI", "False").lower() == "true"
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        timestamp = frame_count / fps
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(image_rgb)
        
        current_frame_mistakes = [] # 현재 프레임에서 발생한 모든 문제점

        if result.pose_landmarks:
            total_frames_with_landmarks +=1
            landmarks = result.pose_landmarks.landmark
            
            if len(landmarks) < 33: # MediaPipe Pose는 33개의 랜드마크를 제공
                if show_gui: 
                    cv2.imshow("Pose Detection", frame)
                    if cv2.waitKey(1) & 0xFF == 27: break
                continue

            # --- 자세 문제점 분석 ---
            posture_checks_results = {
                "몸 흔들림": check_body_stability(landmarks),
                "무릎 벌어짐": check_knee_position(landmarks),
                "허리 굽음/기울어짐": check_back_straightness(landmarks),
                "고개 옆으로 기울어짐": check_head_tilt(landmarks)
            }

            for problem_description, message in posture_checks_results.items():
                if message: # 메시지가 있다면 문제 발생
                    current_frame_mistakes.append(message)
                    mistake_type_counts[message] += 1 
            
            # --- 시선 방향 분석 ---
            gaze_direction_str = estimate_gaze_direction(landmarks, frame_width, frame_height)
            if gaze_direction_str:
                gaze_direction_counts[gaze_direction_str] += 1
                # current_frame_mistakes.append(gaze_direction_str) # 시선도 자세 문제처럼 화면에 표시하려면 주석 해제

            # 현재 프레임의 문제점들 로그 기록 (시간별)
            if current_frame_mistakes:
                log_mistakes_to_txt(current_frame_mistakes, timestamp, output_log_path)
            
            if show_gui:
                annotated_image = frame.copy()
                mp_drawing.draw_landmarks(annotated_image, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                y_offset = 30
                # 현재 프레임의 자세 문제점만 화면에 표시
                temp_display_mistakes = []
                for msg in posture_checks_results.values():
                    if msg: temp_display_mistakes.append(msg)
                
                for message_to_display in temp_display_mistakes:
                    cv2.putText(annotated_image, message_to_display, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    y_offset += 30
                
                if gaze_direction_str: # 시선 방향도 화면에 표시
                     cv2.putText(annotated_image, gaze_direction_str, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,100,0),2)

                cv2.imshow("Pose Detection", annotated_image)
                if cv2.waitKey(1) & 0xFF == 27: # ESC 키로 종료
                    break
        elif show_gui: # 랜드마크가 없더라도 GUI를 표시하는 경우 원본 프레임 표시
            cv2.imshow("Pose Detection", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    if show_gui:
        cv2.destroyAllWindows()
    
    # --- 분석 완료 후 로그 파일에 요약 정보 추가 ---
    with open(output_log_path, "a", encoding="utf-8") as file:
        file.write("\n\n--- 분석 결과 요약 ---\n")
        
        file.write("\n[자세 문제점별 발생 횟수 (프레임 기준)]\n")
        if mistake_type_counts:
            for mistake_msg, count in mistake_type_counts.items():
                duration_seconds = count / fps # 각 문제 유형이 감지된 총 프레임 수 기반 시간
                file.write(f"- {mistake_msg}: {count}회 (약 {duration_seconds:.2f}초 동안 감지됨)\n")
        else:
            file.write("특별한 자세 문제점이 발견되지 않았습니다.\n")
        
        file.write("\n[시선 분석 (추정)]\n")
        if gaze_direction_counts and total_frames_with_landmarks > 0:
            file.write(f"총 유효 프레임 (랜드마크 감지): {total_frames_with_landmarks} 프레임\n")
            for direction, count in sorted(gaze_direction_counts.items()): # 보기 좋게 정렬
                duration_seconds = count / fps
                percentage = (count / total_frames_with_landmarks) * 100
                file.write(f"- {direction}: 약 {duration_seconds:.2f}초 ({percentage:.1f}%)\n")
        else:
            file.write("시선 분석 데이터를 충분히 수집하지 못했거나, 얼굴이 감지되지 않았습니다.\n")
        
        total_video_duration = frame_count / fps
        file.write(f"\n총 영상 분석 시간: {total_video_duration:.2f}초\n")

    print(f"[INFO] 영상 분석 완료: {video_path}")
    if os.path.exists(output_log_path) and os.path.getsize(output_log_path) > 0:
        print(f"[INFO] 로그가 성공적으로 '{output_log_path}'에 저장되었습니다.")
    elif os.path.exists(output_log_path):
        print(f"[INFO] 로그 파일은 생성되었으나 내용이 없습니다: '{output_log_path}'. 자세가 양호했을 수 있습니다.")
    else:
        print(f"[WARNING] 로그 파일이 생성되지 않았습니다: '{output_log_path}'.", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("사용법: python3 0317_correct_pose_detection_video.py <video_path> <output_log_path>", file=sys.stderr)
        sys.exit(1)

    video_path_arg = sys.argv[1]
    output_log_path_arg = sys.argv[2]
    
    main(video_path_arg, output_log_path_arg)