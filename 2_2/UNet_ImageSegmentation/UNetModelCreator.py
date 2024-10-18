import os
import json
import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tensorflow.python.client import device_lib
import tensorflow as tf
from keras.utils import plot_model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import LearningRateScheduler
    
print("텐서플로우 버전:", tf.__version__)
print(device_lib.list_local_devices())

# Intersection over Union
def iou_metric(y_true, y_pred):
    y_true = K.one_hot(tf.cast(y_true, tf.int32), num_classes)
    y_pred = K.cast(y_pred > 0.5, 'float32')
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=[1, 2, 3])
    iou = intersection / (sum_ - intersection + K.epsilon())
    return iou

# f1 스코어 함수 (정밀도와 재현율의 조화)
def f1_score(y_true, y_pred):
    y_true = K.one_hot(tf.cast(y_true, tf.int32), num_classes)
    y_pred = K.cast(y_pred > 0.5, 'float32')
    tp = K.sum(y_true * y_pred, axis=[1, 2, 3])
    fp = K.sum((1 - y_true) * y_pred, axis=[1, 2, 3])
    fn = K.sum(y_true * (1 - y_pred), axis=[1, 2, 3])

    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    
    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1

# LR 스케쥴링
def step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=10):
    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch/step_size))
    
    return LearningRateScheduler(schedule)

# U-Net 모델 정의
def unet_model(input_shape, num_classes):
    inputs = Input(input_shape)
    
    # 인코더 부분
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # 디코더 부분
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    up1 = UpSampling2D(size=(2, 2))(conv4)
    up1 = Concatenate()([up1, conv3])

    conv5 = Conv2D(256, 3, activation='relu', padding='same')(up1)
    conv5 = Conv2D(256, 3, activation='relu', padding='same')(conv5)
    up2 = UpSampling2D(size=(2, 2))(conv5)
    up2 = Concatenate()([up2, conv2])

    conv6 = Conv2D(128, 3, activation='relu', padding='same')(up2)
    conv6 = Conv2D(128, 3, activation='relu', padding='same')(conv6)
    up3 = UpSampling2D(size=(2, 2))(conv6) 
    up3 = Concatenate()([up3, conv1])

    conv7 = Conv2D(64, 3, activation='relu', padding='same')(up3)
    conv7 = Conv2D(64, 3, activation='relu', padding='same')(conv7)

    dropout1 = Dropout(0.5)(conv7)

    # 출력 레이어
    output = Conv2D(num_classes, 1, activation='softmax')(dropout1)

    model = Model(inputs=inputs, outputs=output)
    
    # 모델 컴파일
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy', iou_metric, f1_score])
    return model

# 데이터 불러오기 및 전처리
def load_data(data_folder):
    images = []
    masks = []
    i = 0
    # JSON 폴더에서 모든 JSON 파일 목록을 가져옵니다.
    json_files = [f for f in os.listdir(data_folder) if f.endswith('.json')]

    for json_file in json_files:
        # JSON 파일의 경로
        json_path = os.path.join(data_folder, json_file)

        with open(json_path, 'r') as file:
            data = json.load(file)
            image_path = data['asset']['path'].replace('file:', '')  # 경로에서 'file:' 부분을 제거
            image = cv2.imread(image_path)
            # 이미지 크기 조정
            image = cv2.resize(image, (256, 256))
            images.append(image)
            
            original_width = int(data['asset']['size']['width'])
            original_height = int(data['asset']['size']['height'])
            width_ratio = 256 / original_width
            height_ratio = 256 / original_height

            labels = np.zeros((256, 256), dtype=np.uint8)
            print(str(i) + ":" + image_path)
            for region in data['regions']:
                label = region['tags'][0]
                # 아무것도 할당하지 않은 부분은 0으로 할당되기 때문에, class가 존재하면 0 이상의 값으로 할당
                if label == 'theShoulder':
                    class_id = 1
                elif label == 'carRoad':
                    class_id = 2
                # 보행자 도로도 학습 시키는 경우에만 활성화
                #elif label == 'pedestrianRoad':
                    #class_id = 3
                else:
                    class_id = 0

                if class_id >= 0:
                    points = [[int(point['x'] * width_ratio), int(point['y'] * height_ratio)] for point in region['points']]
                    points = np.array(points, dtype=np.int32)
                    cv2.fillPoly(labels, [points], class_id)
            masks.append(labels)
            i = i + 1
    return images, masks

# JSON 데이터 폴더 경로 설정
data_folder = 'dataset/img'

# 클래스 수 (3개 클래스: UnKnown, theShoulder, carRoad, (선택적)pedestrianRoad)
num_classes = 3

# 이미지 크기 및 입력 형태 설정
input_shape = (256, 256, 3)

# U-Net 모델 생성
model = unet_model(input_shape, num_classes)

plot_model(model, to_file='model.png')
# 데이터 로딩
images, masks = load_data(data_folder)

# 사용자 컬러맵 정의
colors = ['red', 'green', 'blue']
cmap = ListedColormap(colors)

imageIndex = 426

plt.subplot(1, 2, 1)
plt.imshow(images[imageIndex])
plt.title(f"Image {imageIndex}")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(masks[imageIndex], cmap=cmap)
plt.title("Mask")
plt.axis('off')

plt.show()

# 데이터 분할 (학습 데이터와 테스트 데이터로 분할)
X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=0.2, random_state=42)

# 데이터 전처리 (0~1 범위로 정규화)
X_train = np.array(X_train) / 255.0
X_test = np.array(X_test) / 255.0
y_train = np.array(y_train)
y_test = np.array(y_test)

# 학습률 스케줄러 정의
lr_sched = step_decay_schedule(initial_lr=1e-4, decay_factor=0.85, step_size=5)

# 최상의 모델을 저장하는 콜백 정의
model_checkpoint = keras.callbacks.ModelCheckpoint('unet_model_keras.h5', save_best_only=True)

# 콜백 리스트 생성
callbacks = [lr_sched, model_checkpoint]

# 모델 학습
history = model.fit(X_train, y_train, batch_size=2, epochs=50, validation_split=0.1, callbacks = callbacks)

# 모델 평가
loss, accuracy, iou, f1 = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')
print(f'Test IoU: {iou:.4f}')
print(f'Test F1-Score: {f1:.4f}')


loss = history.history['loss']
val_loss = history.history['val_loss']
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

# 그래프 그리기
epochs = range(1, len(loss) + 1)

# 손실 그래프
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# 정확도 그래프
plt.subplot(1, 2, 2)
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()