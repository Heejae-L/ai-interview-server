import cv2
import mediapipe as mp
import numpy as np

# MediaPipe Pose 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 흔들림 판단을 위한 기준 값 저장
previous_positions = None

# 잘못된 자세 로그 저장 함수
def log_mistakes_to_txt(mistakes, timestamp):
    with open("/mistakes_log.txt", "a") as file:
        for mistake in mistakes:
            file.write(f"{timestamp:.2f} sec: {mistake}\n")

# 몸의 흔들림 판단 함수
def check_body_stability(landmarks, threshold=0.05):
    global previous_positions
    
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
    
    if previous_positions is not None:
        movement = np.linalg.norm(current_positions - previous_positions, axis=1).mean()
        if movement > threshold:
            previous_positions = current_positions  # 새로운 위치 업데이트
            return "Your body is shaking."
    
    previous_positions = current_positions  # 초기화
    return None
# 다리 벌어짐 계산 함수
def check_knee_position(landmarks):
    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
    right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
    hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    
    knee_distance = np.abs(left_knee.x - right_knee.x)
    hip_width = np.abs(left_knee.y - hip.y)
    
    if knee_distance > hip_width * 1.2:
        return "Don't spread your knees too much."
    return None

# 허리 기울어짐 계산 함수
def check_back_straightness(landmarks):
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    
    shoulder_slope = abs(left_shoulder.y - right_shoulder.y)
    hip_slope = abs(left_hip.y - right_hip.y)
    
    if shoulder_slope > 0.05 or hip_slope > 0.05:
        return "Keep your back straight."
    return None

# 고개 기울어짐 계산 함수
def check_head_tilt(landmarks):
    nose = landmarks[mp_pose.PoseLandmark.NOSE]
    left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR]
    right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR]
    
    tilt = abs(left_ear.y - right_ear.y)
    if tilt > 0.01:
        return "Your head is tilted."
    return None

# 정면 여부 판단 함수
def check_facing_forward(landmarks):
    nose = landmarks[mp_pose.PoseLandmark.NOSE]
    left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE]
    right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE]
    
    eye_balance = abs(left_eye.x - right_eye.x)
    if eye_balance < 0.1:
        return None
    return "Face forward."

# 메인 실행 함수
def main():
    video_path = "app/interview.mp4"  # 분석할 비디오 파일 경로
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("영상을 열 수 없습니다.")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        timestamp = frame_count / fps  # 현재 시간 (초 단위)
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if result.pose_landmarks:
            mp_drawing.draw_landmarks(image, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = result.pose_landmarks.landmark
            
            mistakes = []
            knee_message = check_knee_position(landmarks)
            back_message = check_back_straightness(landmarks)
            head_message = check_head_tilt(landmarks)
            forward_message = check_facing_forward(landmarks)
            shake_message = check_body_stability(landmarks)
            
            for msg in [knee_message, back_message, head_message, forward_message, shake_message]:
                if msg:
                    mistakes.append(msg)
            
            if mistakes:
                log_mistakes_to_txt(mistakes, timestamp)
                
            y_offset = 50
            for message in mistakes:
                cv2.putText(image, message, (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (225, 225, 255), 2)
                y_offset += 30

        cv2.imshow("Pose Detection", image)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
