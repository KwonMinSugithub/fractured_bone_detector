import cv2
import numpy as np
from keras.models import load_model

# 훈련된 모델 로드
model_path = r"C:\Users\user\Desktop\fractured_bone_detector\test.keras"
model = load_model(model_path)


# 입력 이미지 전처리 함수 정의
def preprocess_image(image):
    desired_size = (224, 224)
    # 그레이스케일로 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 원하는 크기로 조절
    resized = cv2.resize(gray, desired_size)
    # 픽셀 값 정규화
    normalized = resized / 255.0
    # 모델 입력 형태에 맞게 차원 확장
    processed_image = np.expand_dims(normalized, axis=-1)
    processed_image = np.expand_dims(processed_image, axis=0)  # 배치 차원 추가
    return processed_image


# 바운딩 박스 함수 정의
def draw_bounding_box(frame):
    height, width, _ = frame.shape
    box_size = 224
    x = int((width - box_size) / 2)
    y = int((height - box_size) / 2)
    cv2.rectangle(frame, (x, y), (x + box_size, y + box_size), (255, 0, 0), 2)
    return frame, x, y


# 골절 여부를 예측하는 함수 정의
def predict_fracture(image):
    # 입력 이미지 전처리
    processed_image = preprocess_image(image)
    # 골절 확률 예측
    prediction = model.predict(processed_image)
    # 확률이 0.5보다 크면 골절로 분류, 그렇지 않으면 골절이 없음으로 분류
    probability = prediction[0][0]
    result = "Fractured" if probability > 0.5 else "Non-Fractured"
    return result, probability


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from camera.")
        break

    frame_with_box, box_x, box_y = draw_bounding_box(frame)
    # 바운딩 박스 내의 이미지 자르기
    cropped_image = frame[box_y : box_y + 224, box_x : box_x + 224]
    # 잘린 이미지에서 골절 예측
    result, probability = predict_fracture(cropped_image)
    # 프레임에 예측과 확률을 표시
    text = f"{result} ({probability:.2f})"
    cv2.putText(
        frame_with_box, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
    )
    print(f"Result : {result}, Probability : {probability : .2f}")
    cv2.imshow("Fracture Detection", frame_with_box)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
