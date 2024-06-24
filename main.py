import os
import cv2
import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


# 이미지를 로드 및 전처리 함수 정의
def load_images(positive_path, negative_path, desired_size=(224, 224)):
    images = []
    labels = []

    # 긍정 이미지 처리
    for filename in os.listdir(positive_path):
        if filename.endswith(".jpg"):
            img_path = os.path.join(positive_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, desired_size)
                images.append(np.expand_dims(img, axis=-1))
                labels.append(0)  # 긍정 라벨

    # 부정 이미지 처리
    for filename in os.listdir(negative_path):
        if filename.endswith(".jpg"):
            img_path = os.path.join(negative_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, desired_size)
                images.append(np.expand_dims(img, axis=-1))
                labels.append(1)  # 부정 라벨

    return np.array(images), np.array(labels)


# 긍정과 부정 이미지 디렉토리 경로
positive_path = r"C:\Users\user\Desktop\fractured_bone_detector\binary_image\positive"
negative_path = r"C:\Users\user\Desktop\fractured_bone_detector\binary_image\negative"

# 이미지 로드 및 전처리
images, labels = load_images(positive_path, negative_path)
images, labels = shuffle(images, labels, random_state=42)

# 데이터를 훈련, 검증, 테스트 세트로 분할
train_images, test_images, train_labels, test_labels = train_test_split(
    images, labels, test_size=0.2, random_state=42
)
train_images, val_images, train_labels, val_labels = train_test_split(
    train_images, train_labels, test_size=0.2, random_state=42
)

# 픽셀 값 정규화 [0, 1]
train_images = train_images / 255.0
val_images = val_images / 255.0
test_images = test_images / 255.0


# 모델 생성 함수 정의
def create_model(input_shape):
    model = models.Sequential(
        [
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss=keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )

    return model


# 입력 형태 정의
input_shape = train_images[0].shape

# 모델 생성
model = create_model(input_shape)

# 모델 콜백
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    filepath=r"C:\Users\user\Desktop\fractured_bone_detector\test.keras",
    save_best_only=True,
)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

# 모델 훈련
history = model.fit(
    train_images,
    train_labels,
    epochs=20,
    validation_data=(val_images, val_labels),
    callbacks=[checkpoint_cb, early_stopping_cb],
)

# 테스트 세트에서 모델 평가
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")


# 훈련 및 검증 손실 가져오기
train_loss = history.history["loss"]
val_loss = history.history["val_loss"]

# 훈련 및 검증 정확도 가져오기
train_accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]

# 손실 그래프 그리기
plt.plot(train_loss, label="Training Loss")
plt.plot(val_loss, label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# 정확도 그래프 그리기
plt.plot(train_accuracy, label="Training Accuracy")
plt.plot(val_accuracy, label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
