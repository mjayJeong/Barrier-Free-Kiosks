from deepface import DeepFace
import cv2
import numpy as np

# AgeNet 모델 파일 경로 설정
AGE_PROTOTXT = "deploy_age.prototxt"  # 다운로드한 prototxt 파일 경로
AGE_MODEL = "age_net.caffemodel"      # 다운로드한 caffemodel 파일 경로
AGE_BUCKETS = ["0-2", "4-6", "8-12", "15-20", "25-32", "38-43", "48-53", "60-"]

# AgeNet 초기화
age_net = cv2.dnn.readNetFromCaffe(AGE_PROTOTXT, AGE_MODEL)

# DeepFace 가중치 설정
DEEPFACE_WEIGHT = 0.7
AGENET_WEIGHT = 0.3

# AgeNet 나이 예측
def predict_age_agenet(image):
    try:
        blob = cv2.dnn.blobFromImage(image, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
        age_net.setInput(blob)
        preds = age_net.forward()
        age_bucket = AGE_BUCKETS[preds[0].argmax()]
        # AgeNet 나이 중간값 반환
        age_map = {"0-2": 1, "4-6": 5, "8-12": 10, "15-20": 18, "25-32": 28, "38-43": 40, "48-53": 50, "60-": 65}
        return age_map[age_bucket]
    except Exception as e:
        print(f"AgeNet 오류: {e}")
        return None

# DeepFace 나이 예측
def predict_age_deepface(image):
    try:
        result = DeepFace.analyze(img_path=image, actions=["age"], enforce_detection=False, detector_backend="retinaface")
        return result[0].get("age", None)
    except Exception as e:
        print(f"DeepFace 오류: {e}")
        return None

# 연령대 분류
def classify_age(age):
    if age < 20:
        return "10대"
    elif 20 <= age < 30:
        return "20대"
    elif 30 <= age < 40:
        return "30대"
    elif 40 <= age < 50:
        return "40대"
    elif 50 <= age < 60:
        return "50대"
    else:
        return "60대 이상"

# 앙상블 예측
def ensemble_age_prediction(image):
    predictions = []

    # DeepFace 예측
    deepface_age = predict_age_deepface(image)
    if deepface_age is not None:
        predictions.append((deepface_age, DEEPFACE_WEIGHT))

    # AgeNet 예측
    agenet_age = predict_age_agenet(image)
    if agenet_age is not None:
        predictions.append((agenet_age, AGENET_WEIGHT))

    if predictions:
        # 가중치를 적용한 평균 나이 계산
        weighted_sum = sum(age * weight for age, weight in predictions)
        total_weight = sum(weight for _, weight in predictions)
        avg_age = weighted_sum / total_weight
        return classify_age(avg_age)
    else:
        return "나이 예측 실패"

# 웹캠 입력을 통한 실시간 나이 추정
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
else:
    print("웹캠에서 실시간 나이 추정을 시작합니다. 종료하려면 'q'를 누르세요.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("프레임을 가져올 수 없습니다.")
            break

        # 얼굴 검출
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # 얼굴별로 나이 예측 및 출력
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w].copy()
            age_group = ensemble_age_prediction(face_img)

            # 결과를 이미지에 출력
            label = f"{age_group}"
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # 프레임을 창에 표시
        cv2.imshow('Real-Time Age Prediction', frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
