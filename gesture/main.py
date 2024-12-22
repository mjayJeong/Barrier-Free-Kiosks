import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import models
import mediapipe as mp



# 모델 불러오기
model = models.load_model('6-3_model.h5')

# MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# 카메라 연동
cap = cv2.VideoCapture(0)

# 손 감지하고 잘라내서 반환
def detect_hands(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    hand_images = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, _ = image.shape
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)
            
            hand_crop = image[y_min:y_max, x_min:x_max]
            hand_images.append(hand_crop)
            
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    return hand_images


# 평균 밝기를 계산하는 함수
def calculate_brightness(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    average_brightness = np.mean(gray_image)  # 평균 밝기 계산
    lux_value = average_brightness * (100 / 255)  # 0-255 범위를 0-100 lux로 변환 (단순 예시)
    return lux_value



if not cap.isOpened():
    exit()
    
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 손 인식하여 hand_images 리스트에 저장
    hand_images = detect_hands(frame)

    brightness = calculate_brightness(frame)
    
    # 다른 예측
    if hand_images:
        hand_images_resized = []
        for img in hand_images:
            if img.size > 0:  # 이미지가 비어있지 않은지 확인
                resized_img = cv2.resize(img, (64, 64))  # 모델 입력 크기로 리사이즈
                hand_images_resized.append(resized_img)
        
        if hand_images_resized:  # 리사이즈된 이미지가 있는 경우
            hand_images_np = np.array(hand_images_resized) / 255.0  # 정규화
            predictions = model.predict(hand_images_np)
            if predicted_class == 0:
                predicted_class = 1
            else:
                predicted_class = np.argmax(predictions, axis=1)[0]
    else:
        predicted_class = "No hand detected"  # 손이 감지되지 않았을 경우 기본 메시지 설정
    
    cv2.putText(frame, f'Brightness: {brightness:.2f} lux', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # 밝기 표시
    cv2.putText(frame, f'Predicted: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)  # 예측 클래스 표시

    
    # # 간단 예측        
    # predictions = model.predict(hand_images)
    # predicted_class = np.argmax(predictions, axis=1)[0]
 
    # 프레임 띄우기
    cv2.imshow('Hand Recognition', frame)
        
    # q키로 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 해제
cap.release()
cv2.destroyAllWindows()
