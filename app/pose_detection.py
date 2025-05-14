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


def analyze_video(video_path: str, output_log_path: str) -> None:
    """
    1) video_path 영상을 열어서 MediaPipe 분석
    2) 문제점별 타임스탬프 기록 → output_log_path에 저장
    3) 마지막에 요약(횟수, 퍼센트 등)도 같은 파일에 덧붙임
    """
    global previous_positions
    previous_positions = None

    # 기존 main()의 초기화 부분
    if os.path.exists(output_log_path):
        os.remove(output_log_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = 0

    mistake_counts = Counter()
    gaze_counts    = Counter()
    valid_frames   = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        timestamp = frame_count / fps

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        if res.pose_landmarks:
            valid_frames += 1
            lm = res.pose_landmarks.landmark

            # 자세 체크
            mistakes = []
            for fn in (check_body_stability, check_knee_position,
                       check_back_straightness, check_head_tilt):
                msg = fn(lm)
                if msg:
                    mistakes.append(msg)
                    mistake_counts[msg] += 1

            # 시선 체크
            gaze = estimate_gaze_direction(lm, frame_w, frame_h)
            gaze_counts[gaze] += 1

            # 프레임별 문제 로그
            if mistakes:
                log_mistakes_to_txt(mistakes, timestamp, output_log_path)

    cap.release()

    # 요약 기록
    with open(output_log_path, "a", encoding="utf-8") as f:
        f.write("\n\n--- 분석 결과 요약 ---\n")
        f.write("[자세 문제점별 감지 프레임 수]\n")
        for msg, cnt in mistake_counts.items():
            sec = cnt / fps
            f.write(f"- {msg}: {cnt}회 ({sec:.2f}초)\n")
        f.write("\n[시선 분석]\n")
        if valid_frames:
            for g, cnt in gaze_counts.items():
                pct = cnt / valid_frames * 100
                f.write(f"- {g}: {cnt}프레임 ({pct:.1f}%)\n")
        else:
            f.write("랜드마크가 감지된 프레임이 없습니다.\n")
        f.write(f"\n[총 영상 길이] {frame_count/fps:.2f}초\n")