import torch
from tensorflow import keras
from keras import layers, models, regularizers  # regularizers 추가
from keras import optimizers
import h5py
import pickle

# Device 설정 (여기서는 CPU만 사용)
# device = torch.device("cuda")

## 1. 모델 만들기

# Sequential 클래스 객체
classifier = keras.models.Sequential()

# 1st Convolution Layer - 수정된 부분
classifier.add(keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=(64, 64, 3),
                                    kernel_regularizer=regularizers.l2(0.01)))  # L2 정규화
classifier.add(keras.layers.BatchNormalization())  # Batch Normalization을 먼저 적용
classifier.add(keras.layers.Activation('gelu'))  # 활성화 함수 적용
classifier.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
classifier.add(keras.layers.Dropout(0.3))  # Dropout 추가

# 2nd convolution layer
# 2nd Convolution Layer - 수정된 부분
classifier.add(keras.layers.Conv2D(32, (3, 3), padding='same', 
                                    kernel_regularizer=regularizers.l2(0.01)))  # L2 정규화 추가
classifier.add(keras.layers.BatchNormalization())  # Batch Normalization을 먼저 적용
classifier.add(keras.layers.Activation('gelu'))  # 활성화 함수 적용
classifier.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
classifier.add(keras.layers.Dropout(0.3))  # Dropout 추가

# 3rd Convolution Layer
classifier.add(keras.layers.Conv2D(64, (3, 3), padding='same', 
                                    kernel_regularizer=regularizers.l2(0.01)))  # L2 정규화 추가
classifier.add(keras.layers.BatchNormalization())  # Batch Normalization을 먼저 적용
classifier.add(keras.layers.Activation('gelu'))  # 활성화 함수 적용
classifier.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
classifier.add(keras.layers.Dropout(0.3))  # Dropout 추가


# 4th Convolution Layer
classifier.add(keras.layers.Convolution2D(128, (3, 3), activation='gelu', padding='same'))
classifier.add(keras.layers.BatchNormalization())  # Batch Normalization
classifier.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

# Flattening
classifier.add(keras.layers.Flatten())

# Fully connected layer
classifier.add(keras.layers.Dense(256, kernel_regularizer=regularizers.l2(0.01)))  # L2 정규화 추가
classifier.add(keras.layers.BatchNormalization())  # Batch Normalization을 먼저 적용
classifier.add(keras.layers.Activation('gelu'))  # 활성화 함수 적용
classifier.add(keras.layers.Dropout(0.3))  # Dropout 추가

# Output layer
classifier.add(keras.layers.Dense(10, activation='softmax'))

# Optimizer를 Adam으로 하고, learning rate를 약간 줄임
opt = keras.optimizers.Adam(learning_rate=0.0005)

# Compiling
classifier.compile(
    optimizer=opt,  # Adam
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

## 2. 모델 훈련
train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    brightness_range=(0.8, 1.2),
    horizontal_flip=True
)

test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    '/raid/co_show05/Gesture/training_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)


# 조기 종료(Early Stopping)
early_stopping = keras.callbacks.EarlyStopping(
    monitor='loss',
    patience=5,
    restore_best_weights=True
)

# 모델 학습
model = classifier.fit(
    training_set,
    steps_per_epoch=1300,
    epochs=40,
    callbacks=[early_stopping]  # 조기 종료
)



## 3. 모델 저장
classifier.save('Gesture/6_6_model.h5')

with open('Gesture/6_6_history.pkl', 'wb') as f:
    pickle.dump(model.history, f)



## 4. 성능 확인
import matplotlib.pyplot as plt

# 정확도
plt.plot(model.history['accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train'], loc='upper left')
plt.show()

# 손실
plt.plot(model.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
plt.show()


## 5. 모델 평가 (테스트 데이터)
# test_set 생성 및 모델 평가
test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

test_set = test_datagen.flow_from_directory(
    '/raid/co_show05/Gesture/test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

# 테스트 데이터에 대한 모델 평가
test_loss, test_accuracy = classifier.evaluate(test_set)

print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")






#### gpt 4o도 좋은데, copilot gpt 써보면서 변경사항 업데이트 해가면서 물어봐도
#### training set 정확도 높이기, test 정확도 줄어도 상관없음 

#### filter, layer수를 줄여보거나

