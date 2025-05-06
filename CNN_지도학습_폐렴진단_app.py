from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# 1. 경로 설정
train_dir = "chest_xray/train"
val_dir = "chest_xray/val"

# 2. 데이터 불러오기
train_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    train_dir, target_size=(150, 150), batch_size=16, class_mode='binary')

val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    val_dir, target_size=(150, 150), batch_size=16, class_mode='binary')

# 3. 모델 구성
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# 4. 컴파일 & 학습
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(train_gen, epochs=3, validation_data=val_gen)
print("학습 정확도:", history.history['accuracy'])
print("검증 정확도:", history.history['val_accuracy'])

import numpy as np
from tensorflow.keras.preprocessing import image

# 테스트 이미지 경로
img_path = 'chest_xray/test/NORMAL/IM-0001-0001.jpeg'  # 예시 경로

# 이미지 전처리
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# 예측 실행
prediction = model.predict(img_array)

# 결과 출력
if prediction[0][0] > 0.5:
    print(f"예측 결과: 폐렴일 확률은 {prediction[0][0]*100:.2f}% 입니다.")
else:
    print(f"예측 결과: 정상일 확률은 {(1 - prediction[0][0])*100:.2f}% 입니다.")

import matplotlib.pyplot as plt

# 정확도 시각화
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# 손실값 시각화
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


import os
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image

# 테스트 폴더에서 이미지 몇 개만 불러오기
test_dir = 'chest_xray/test/NORMAL'  # 또는 PNEUMONIA로 바꿔서 확인 가능
img_files = os.listdir(test_dir)[:5]  # 앞에서 5개만 예시로 사용

plt.figure(figsize=(15, 8))

for i, fname in enumerate(img_files):
    img_path = os.path.join(test_dir, fname)
    
    # 이미지 불러오기
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # 예측
    prediction = model.predict(img_array)[0][0]

    # 예측 결과 해석
    if prediction > 0.5:
        label = f"폐렴 가능성 {prediction*100:.2f}%"
    else:
        label = f"정상 가능성 {(1 - prediction)*100:.2f}%"

    # 시각화
    plt.subplot(1, 5, i+1)
    plt.imshow(img)
    plt.title(label, fontsize=10)
    plt.axis('off')

plt.tight_layout()
plt.savefig("cnn_training_results.png")
plt.show()




