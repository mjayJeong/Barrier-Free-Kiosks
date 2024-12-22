import cv2
import mediapipe as mp
import numpy as np
import time

# MediaPipe 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

# 화면 크기 설정 (1080 x 1920)
SCREEN_WIDTH = 1080
SCREEN_HEIGHT = 1920

# 캘리브레이션 지점 (화면의 네 구석)
calibration_points = [
    (50, 50), (1030, 50), (50, 1870), (1030, 1870)
]

# 웹캠 설정
cap = cv2.VideoCapture(0)

def get_eye_direction(landmarks, eye_indices, eye_center):
    # 눈의 특정 랜드마크들로 방향 추정
    eye_left = landmarks[eye_indices[0]]
    eye_right = landmarks[eye_indices[3]]

    # 화면 크기에 맞게 변환
    eye_left = np.array([eye_left.x, eye_left.y])
    eye_right = np.array([eye_right.x, eye_right.y])

    # 벡터 계산 (오른쪽 -> 왼쪽 방향)
    eye_direction = eye_right - eye_left
    eye_direction = eye_direction / np.linalg.norm(eye_direction)  # 단위 벡터화

    # 눈의 중심에서 방향으로 벡터 확장
    gaze_point = eye_center + eye_direction * 50  # 50 픽셀 만큼 벡터 확장

    return gaze_point

# 캘리브레이션 루프
def calibrate(cap, face_mesh, calibration_points, duration=5):
    calibration_data = []
    for point_index, point in enumerate(calibration_points):
        print(f"Point {point_index + 1}을(를) 바라보세요.")
        pupils_list = []
        start_time = time.time()

        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret:
                print("프레임을 가져오는 데 실패했습니다.")
                return None

            # 캘리브레이션 지점 지속적으로 표시
            cv2.circle(frame, point, 10, (0, 255, 0), -1)
            cv2.putText(frame, f'Point {point_index + 1}', (point[0] + 10, point[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 얼굴 랜드마크 감지
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # 왼쪽 눈과 오른쪽 눈 랜드마크 인덱스
                    left_eye_indices = [33, 160, 158, 133]  # 왼쪽 눈
                    right_eye_indices = [263, 387, 385, 362]  # 오른쪽 눈

                    # 눈의 중심 계산
                    h, w, _ = frame.shape
                    left_eye_center = np.mean([[face_landmarks.landmark[i].x * w,
                                                face_landmarks.landmark[i].y * h] for i in left_eye_indices], axis=0)
                    right_eye_center = np.mean([[face_landmarks.landmark[i].x * w,
                                                 face_landmarks.landmark[i].y * h] for i in right_eye_indices], axis=0)

                    # 중앙 눈 위치 계산
                    eye_center = (left_eye_center + right_eye_center) / 2
                    pupils_list.append(eye_center)

            cv2.imshow('Calibration', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # 캘리브레이션 평균값 저장
        calibration_data.append(np.mean(pupils_list, axis=0))
        print(f"Point {point_index + 1} 캘리브레이션 좌표: {calibration_data[-1]}")
        print(f"Point {point_index + 1} 캘리브레이션 완료! 2초 대기")
        time.sleep(2)

    return calibration_data

# 시선 추적 루프
def gaze_tracking(cap, face_mesh, calibration_data):
    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 가져오는 데 실패했습니다.")
            break

        # 얼굴 랜드마크 감지
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # 왼쪽 눈과 오른쪽 눈 랜드마크 인덱스
                left_eye_indices = [33, 160, 158, 133]  # 왼쪽 눈
                right_eye_indices = [263, 387, 385, 362]  # 오른쪽 눈

                # 눈의 중심 계산
                h, w, _ = frame.shape
                left_eye_center = np.mean([[face_landmarks.landmark[i].x * w,
                                            face_landmarks.landmark[i].y * h] for i in left_eye_indices], axis=0)
                right_eye_center = np.mean([[face_landmarks.landmark[i].x * w,
                                             face_landmarks.landmark[i].y * h] for i in right_eye_indices], axis=0)

                # 왼쪽과 오른쪽 눈 방향 추정
                left_gaze_point = get_eye_direction(face_landmarks.landmark, left_eye_indices, left_eye_center)
                right_gaze_point = get_eye_direction(face_landmarks.landmark, right_eye_indices, right_eye_center)

                # 왼쪽과 오른쪽 시선 방향의 평균을 계산하여 하나의 시선 방향으로 설정
                gaze_point = (left_gaze_point + right_gaze_point) / 2

                # 시선 좌표를 화면에 맞게 매핑 (캘리브레이션 데이터 이용)
                x_ratio = (gaze_point[0] - calibration_data[0][0]) / (calibration_data[3][0] - calibration_data[0][0])
                y_ratio = (gaze_point[1] - calibration_data[0][1]) / (calibration_data[3][1] - calibration_data[0][1])
                mapped_x = max(0, min(SCREEN_WIDTH, int(SCREEN_WIDTH * x_ratio)))
                mapped_y = max(0, min(SCREEN_HEIGHT, int(SCREEN_HEIGHT * y_ratio)))

                # 눈 중심과 시선의 끝점을 연결하여 시선 방향 표시
                eye_center = (left_eye_center + right_eye_center) / 2
                cv2.line(frame, (int(eye_center[0]), int(eye_center[1])), (mapped_x, mapped_y), (0, 0, 255), 2)

                # 시선이 가리키는 지점 좌표 출력
                print(f"시선 좌표: ({mapped_x}, {mapped_y})")

        # 결과 프레임을 화면에 표시
        cv2.imshow("Gaze Tracker", frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 메인 실행부
if __name__ == "__main__":
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    # 캘리브레이션 수행
    calibration_data = calibrate(cap, face_mesh, calibration_points, duration=5)
    if calibration_data:
        print("캘리브레이션 완료! 시선 추적을 시작합니다.")
        gaze_tracking(cap, face_mesh, calibration_data)

    # 리소스 해제
    cap.release()
    cv2.destroyAllWindows()
