import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0: DEBUG, 1: INFO, 2: WARNING, 3: ERROR
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tf_keras.models import Sequential, load_model
from tf_keras.layers import  Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tf_keras.preprocessing import image
from tf_keras.callbacks import EarlyStopping
from tf_keras.optimizers import Adam

# 모델 저장 경로
model_path = "C:/Users/Jun/Desktop/anaconda/3.10python/final/bolt_ok_ng_model.h5"


# 이미지 크기 및 파라미터 설정
img_size = (128, 128)

# 📌 기존 모델이 존재하면 학습을 건너뜀
if os.path.exists(model_path):
    print(f"✅ 기존 모델이 이미 존재합니다. ({model_path})")
    exit()  # 학습을 다시 수행하지 않음


print("📌 저장된 모델이 없으므로 새로운 모델을 학습합니다.")

# 데이터 로드 함수
def load_images_from_folder(folder, label):
    images, labels = [], []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        try:
            img = image.load_img(img_path, target_size=img_size)
            img_array = image.img_to_array(img) / 255.0
            images.append(img_array)
            labels.append(label)
        except Exception as e:
            print(f"⚠ 이미지 로드 오류: {img_path}, {e}")
    return images, labels

# 데이터셋 경로 설정 (원본 + 증강 이미지 포함)
ok_dir_1 = "C:/Users/Jun/Desktop/anaconda/3.10python/final/train/OK"
ok_dir_2 = "C:/Users/Jun/Desktop/anaconda/3.10python/final/train/OK_augmented"
ng_dir_1 = "C:/Users/Jun/Desktop/anaconda/3.10python/final/train/NG"
ng_dir_2 = "C:/Users/Jun/Desktop/anaconda/3.10python/final/train/NG_augmented"


# 데이터 로드
ok_images_1, ok_labels_1 = load_images_from_folder(ok_dir_1, 0)
ok_images_2, ok_labels_2 = load_images_from_folder(ok_dir_2, 0)
ng_images_1, ng_labels_1 = load_images_from_folder(ng_dir_1, 1)
ng_images_2, ng_labels_2 = load_images_from_folder(ng_dir_2, 1)

# 합치기
ok_images = ok_images_1 + ok_images_2
ok_labels = ok_labels_1 + ok_labels_2
ng_images = ng_images_1 + ng_images_2
ng_labels = ng_labels_1 + ng_labels_2

# 최종 데이터 병합
X = np.array(ok_images + ng_images)
y = np.array(ok_labels + ng_labels)

# 훈련/테스트 데이터 분할 (80:20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# CNN 모델 정의
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.6),                                       #과적합 방지 조절
    Dense(1, activation='sigmoid')
])

# 모델 컴파일
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 📌 Early Stopping 설정 (자동으로 최적 `epochs` 찾기)
early_stopping = EarlyStopping(
    monitor='val_loss',  # 검증 데이터의 손실 값 기준으로 모니터링
    patience=5,          # 손실 값이 5번 연속 개선되지 않으면 학습 중단
    restore_best_weights=True  # 가장 좋은 가중치 복원
)

# 학습 진행 (Early Stopping 적용)
print("📌 모델 학습을 시작합니다...")
history = model.fit(
    X_train, y_train,
    epochs=100,  # 너무 크게 설정해도 Early Stopping이 자동으로 중단
    validation_data=(X_test, y_test),
    callbacks=[early_stopping]
)

# 모델 저장
model.save(model_path)
print(f"✅ 학습 완료! 모델이 저장되었습니다: {model_path}")

# 📌 학습 과정 시각화 (손실 값 및 정확도 그래프 출력)
plt.figure(figsize=(12, 5))

# 손실 값 그래프
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Loss Graph")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

# 정확도 그래프
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Accuracy Graph")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.show()

#검증용 파일
np.save("C:/Users/Jun/Desktop/anaconda/3.10python/final/X_test.npy", X_test)
np.save("C:/Users/Jun/Desktop/anaconda/3.10python/final/Y_test.npy", y_test)




