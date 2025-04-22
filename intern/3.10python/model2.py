import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from tf_keras.models import Sequential
from tf_keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tf_keras.preprocessing import image
from tf_keras.callbacks import EarlyStopping

# 모델 저장 경로
model_path = "C:/Users/Jun/Desktop/anaconda/3.10python/final/screw_defect_model.h5"

# 📌 기존 모델이 존재하면 학습을 건너뜀
if os.path.exists(model_path):
    print(f"✅ 기존 모델이 이미 존재합니다. ({model_path})")
    exit()  # 학습을 다시 수행하지 않음

print("📌 저장된 모델이 없으므로 새로운 모델을 학습합니다.")

# 데이터 로드 함수
def load_images_from_folder(folder, label):
    images, labels = [], []
    label_map = {
        '2회전조': 0,
        '무나사': 1,
        '제자리나사': 2,
        '찍힘': 3,
        '나사산 문제': 4
    }

    if label not in label_map:
        raise ValueError(f"Invalid label: {label}. Valid labels are {list(label_map.keys())}.")

    label_value = label_map[label]

    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        try:
            img = image.load_img(img_path, target_size=(128, 128))
            img_array = image.img_to_array(img) / 255.0
            images.append(img_array)
            labels.append(label_value)
        except Exception as e:
            print(f"⚠ 이미지 로드 오류: {img_path}, {e}")
    return images, labels

# 데이터셋 경로 설정 (NG 이미지와 각 불량 원인 라벨 포함)
ng_dir_a = "C:/Users/Jun/Desktop/anaconda/3.10python/final/train/분류/2회전조"
ng_dir_b = "C:/Users/Jun/Desktop/anaconda/3.10python/final/train/분류/무나사"
ng_dir_c = "C:/Users/Jun/Desktop/anaconda/3.10python/final/train/분류/제자리나사"
ng_dir_d = "C:/Users/Jun/Desktop/anaconda/3.10python/final/train/분류/찍힘"
ng_dir_e = "C:/Users/Jun/Desktop/anaconda/3.10python/final/train/분류/나사산"
ng_dir_a_augmented = "C:/Users/Jun/Desktop/anaconda/3.10python/final/train/분류/2회전조_agumented"
ng_dir_b_augmented = "C:/Users/Jun/Desktop/anaconda/3.10python/final/train/분류/무나사_agumented"
ng_dir_c_augmented = "C:/Users/Jun/Desktop/anaconda/3.10python/final/train/분류/제자리나사_agumented"
ng_dir_d_augmented = "C:/Users/Jun/Desktop/anaconda/3.10python/final/train/분류/찍힘_agumented"
ng_dir_e_augmented = "C:/Users/Jun/Desktop/anaconda/3.10python/final/train/분류/나사산_agumented"

# 불량 원인별 데이터 로드
ng_images_a, ng_labels_a = load_images_from_folder(ng_dir_a, '2회전조')
ng_images_b, ng_labels_b = load_images_from_folder(ng_dir_b, '무나사')
ng_images_c, ng_labels_c = load_images_from_folder(ng_dir_c, '제자리나사')
ng_images_d, ng_labels_d = load_images_from_folder(ng_dir_d, '찍힘')
ng_images_e, ng_labels_e = load_images_from_folder(ng_dir_e, '나사산 문제')
ng_images_a_aug, ng_labels_a_aug = load_images_from_folder(ng_dir_a_augmented, '2회전조')
ng_images_b_aug, ng_labels_b_aug = load_images_from_folder(ng_dir_b_augmented, '무나사')
ng_images_c_aug, ng_labels_c_aug = load_images_from_folder(ng_dir_c_augmented, '제자리나사')
ng_images_d_aug, ng_labels_d_aug = load_images_from_folder(ng_dir_d_augmented, '찍힘')
ng_images_e_aug, ng_labels_e_aug = load_images_from_folder(ng_dir_e_augmented, '나사산 문제')

# 데이터 병합 (원본 + 증강)
ng_images = ng_images_a + ng_images_b + ng_images_c + ng_images_d + ng_images_e
ng_labels = ng_labels_a + ng_labels_b + ng_labels_c + ng_labels_d + ng_labels_e

# 증강 이미지 병합
ng_images += ng_images_a_aug + ng_images_b_aug + ng_images_c_aug + ng_images_d_aug + ng_images_e_aug
ng_labels += ng_labels_a_aug + ng_labels_b_aug + ng_labels_c_aug + ng_labels_d_aug + ng_labels_e_aug

# 훈련/테스트 데이터 분할 (80:20)
X_train, X_test, y_train, y_test = train_test_split(ng_images, ng_labels, test_size=0.2, random_state=42)

# 데이터 크기 확인
print(f"X_train shape: {len(X_train)}")
print(f"y_train shape: {len(y_train)}")

# CNN 모델 정의 (불량 원인 분류 모델)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(5, activation='softmax')  # 5개의 불량 원인으로 분류
])

# 모델 컴파일
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Early Stopping 설정
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 모델 학습
history = model.fit(np.array(X_train), np.array(y_train), epochs=50, validation_data=(np.array(X_test), np.array(y_test)), callbacks=[early_stopping])

# 모델 저장
model.save(model_path)

plt.rc('font', family='Malgun Gothic')
# 학습 과정 시각화
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Loss Graph")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Accuracy Graph")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.show()

# 모델 평가 후 confusion matrix 출력
from sklearn.metrics import confusion_matrix, classification_report

# 예측 실행
y_pred = model.predict(np.array(X_test))
y_pred_labels = np.argmax(y_pred, axis=1)  # 가장 높은 확률을 가진 클래스를 선택

# Confusion Matrix 계산 및 시각화
cm = confusion_matrix(y_test, y_pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["2회전조", "무나사", "제자리나사", "찍힘", "나사산 문제"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# Precision / Recall / F1-score 리포트 출력
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred_labels, target_names=["2회전조", "무나사", "제자리나사", "찍힘", "나사산 문제"]))
