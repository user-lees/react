import os
import subprocess
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tf_keras.models import Sequential, load_model
from tf_keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tf_keras.preprocessing import image
from tf_keras.callbacks import EarlyStopping

# 가상환경의 Python 경로
python_path = "C:/Users/Jun/Desktop/anaconda/3.10python/.venv/Scripts/python.exe"

# 모델 저장 경로
model_path1 = "C:/Users/Jun/Desktop/anaconda/3.10python/final/bolt_ok_ng_model.h5"
model_path2 = "C:/Users/Jun/Desktop/anaconda/3.10python/final/screw_defect_model.h5"

# 첫 번째 파일 실행 (모델이 없으면 학습)
if not os.path.exists(model_path1):
    print("📌 1단계: 모델 학습 시작 (train.py 실행 중)...")
    result = subprocess.run([python_path, "model1.py"], capture_output=True, text=True)
    print(f"Train.py 출력: {result.stdout}")
    if result.returncode != 0:
        print(f"train.py 오류: {result.stderr}")
else:
    print(f"✅ 모델1이 이미 존재합니다. ({model_path1}) 1단계 건너뛰기.")

# 두 번째 파일 실행 (모델이 없으면 학습)
if not os.path.exists(model_path2):
    print("📌 2단계: 분류 모델 학습 시작 (classification_train.py 실행 중)...")
    result = subprocess.run([python_path, "model2.py"], capture_output=True, text=True)
    print(f"Classification_train.py 출력: {result.stdout}")
    if result.returncode != 0:
        print(f"classification_train.py 오류: {result.stderr}")
else:
    print(f"✅ 모델2이 이미 존재합니다. ({model_path2}) 2단계 건너뛰기.")

# 세 번째 파일 실행 (모델이 존재하면 실행)
print("📌 3단계: 테스트 및 결과 저장 시작 (test.py 실행 중)...")
result = subprocess.run([python_path, "test.py"], capture_output=True, text=True)
print(f"Test.py 출력: {result.stdout}")
if result.returncode != 0:
    print(f"test.py 오류: {result.stderr}")

print("📌 모든 작업이 완료되었습니다.")


