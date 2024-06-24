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

# 이미지 처리 함수
def handle_image(positive_path, negative_path):
    img = []
    labels = []

    desired_width = 224
    desired_height = 224

    # 긍정 이미지 처리
    for i in os.listdir(positive_path):
        if i.endswith(".jpg"):  # ".jpg" 확장자를 가진 파일만 처리
            imgurl = os.path.join(positive_path, i)
            imgfile = cv2.imread(imgurl, cv2.IMREAD_GRAYSCALE)
            if imgfile is None:
                print(f"Failed to load image at {imgurl}")
                continue
            imgfile = cv2.resize(imgfile, (desired_width, desired_height))
            imgfile = np.expand_dims(imgfile, axis=-1)
            img.append(imgfile)
            labels.append(0)  # 긍정 이미지에 대한 라벨 추가

    # 부정 이미지 처리
    for i in os.listdir(negative_path):
        if i.endswith(".jpg"):  # ".jpg" 확장자를 가진 파일만 처리
            imgurl = os.path.join(negative_path, i)
            imgfile = cv2.imread(imgurl, cv2.IMREAD_GRAYSCALE)
            if imgfile is None:
                print(f"Failed to load image at {imgurl}")
                continue
            imgfile = cv2.resize(imgfile, (desired_width, desired_height))
            imgfile = np.expand_dims(imgfile, axis=-1)
            img.append(imgfile)
            labels.append(1)  # 부정 이미지에 대한 라벨 추가

    # label_positive = np.zeros((len(os.listdir(positive_path))), dtype=int)
    # label_negative = np.ones((len(os.listdir(negative_path))), dtype=int)
    # labels.extend(label_positive)
    # labels.extend(label_negative)

    return np.array(img), np.array(labels)


positive_path = r"C:\Users\user\Desktop\fractured_bone_detector\binary_image\positive"
negative_path = r"C:\Users\user\Desktop\fractured_bone_detector\binary_image\negative"

image_data, labels = handle_image(positive_path, negative_path)


image_data = np.array(image_data)
labels = np.array(labels)

X, Y = shuffle(image_data, labels, random_state=42)

train_input, test_input, train_target, test_target = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

scaled_input = train_input / 255.0

train_scaled, val_scaled, train_target, val_target = train_test_split(
    scaled_input, train_target, test_size=0.2, random_state=42
)


def create_model(input_shape):
    # num_classes = 2
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))

    model.add(layers.Flatten())

    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation="sigmoid"))

    # if num_classes > 1:
    #     output_layer = layers.Dense(num_classes)
    #     output_activation = "softmax"
    # else:
    #     output_layer = layers.Dense(1)
    #     output_activation = "sigmoid"

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )

    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        r"C:\Users\user\Desktop\fractured_bone_detector\test.keras", save_best_only=True
    )
    early_stopping_cb = keras.callbacks.EarlyStopping(
        patience=5, restore_best_weights=True
    )

    return model, [checkpoint_cb, early_stopping_cb]


input_shape = (224, 224, 1)
model, callbacks = create_model(input_shape)

history = model.fit(
    train_scaled,
    train_target,
    epochs=70,
    validation_data=(val_scaled, val_target),
    callbacks=callbacks,
)

test_scaled = test_input / 255.0
test_loss, test_accuracy = model.evaluate(test_scaled, test_target)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
