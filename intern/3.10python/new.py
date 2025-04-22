import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
import numpy as np
import pandas as pd
from tf_keras.models import load_model

# 모델 로드 (OK/NG 분류 모델)
ok_ng_model_path = "C:/Users/Jun/Desktop/anaconda/3.10python/final/bolt_ok_ng_model.h5"
ok_ng_model = load_model(ok_ng_model_path)

# 불량 원인 분류 모델 로드
defect_model_path = "C:/Users/Jun/Desktop/anaconda/3.10python/final/screw_defect_model.h5"
defect_model = load_model(defect_model_path)

# 고정된 좌/우 나사 영역 (최신 기준)
LEFT_BOX = (190, 155, 210, 170)
RIGHT_BOX = (420, 155, 210, 170)

# 이미지 크기 (모델 입력 사이즈)
img_size = (128, 128)

# 불량 원인 분류 함수
def classify_defect(cropped_img):
    cropped_img_resized = cv2.resize(cropped_img, img_size)
    cropped_img_array = np.array(cropped_img_resized) / 255.0
    cropped_img_array = np.expand_dims(cropped_img_array, axis=0)

    defect_prediction = defect_model.predict(cropped_img_array, verbose=0)
    defect_label = np.argmax(defect_prediction, axis=1)[0]

    defect_labels = ["2회전조", "무나사", "제자리나사", "찍힘", "나사산 문제"]
    return defect_labels[defect_label], defect_prediction[0][defect_label]

# 개별 이미지에서 좌/우 나사 예측 함수
def predict_fixed_screw_boxes(model, img_path):
    bgr_img = cv2.imread(img_path)
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    results = []
    for label, (x, y, w, h) in zip(["LEFT", "RIGHT"], [LEFT_BOX, RIGHT_BOX]):
        screw_img = rgb_img[y:y + h, x:x + w]
        screw_img_resized = cv2.resize(screw_img, img_size)
        screw_img_array = np.array(screw_img_resized) / 255.0
        screw_img_array = np.expand_dims(screw_img_array, axis=0)

        prediction = model.predict(screw_img_array, verbose=0)[0][0]
        result = "OK" if prediction < 0.3 else "NG"
        confidence = 1 - prediction if prediction < 0.3 else prediction

        # NG인 경우 불량 원인 분류 실행
        if result == "NG":
            defect_type, defect_confidence = classify_defect(screw_img)
            results.append((os.path.basename(img_path), label, result, round(float(confidence), 4), defect_type, round(float(defect_confidence), 4)))
        else:
            results.append((os.path.basename(img_path), label, result, round(float(confidence), 4), "None", 0.0))

    return results

# 폴더 내 모든 이미지 검사 실행
def predict_folder_images(model, folder_path):
    all_results = []
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    if not image_files:
        print("⚠ 폴더에 이미지가 없습니다.")
        return []

    print(f"\n📂 검사 시작: {len(image_files)}개 이미지 분석 중...\n")

    for file in image_files:
        img_path = os.path.join(folder_path, file)
        results = predict_fixed_screw_boxes(model, img_path)
        all_results.extend(results)

        for r in results:
            print(f"[{r[0]} - {r[1]}] → {r[2]} (신뢰도: {r[3]:.2f}), 불량 원인: {r[4]} (신뢰도: {r[5]:.2f})")

    return all_results


import pyodbc
from datetime import datetime  # 날짜 정보를 추가하기 위해 import

# 결과 저장 함수 (DB에 저장)
def save_results_to_db(results, db_connection_str):
    if results:
        conn = pyodbc.connect(db_connection_str)
        cursor = conn.cursor()

        # 현재 날짜 추가
        current_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # 결과를 DB에 삽입
        insert_data = [(current_date, r[0], r[1], r[2], r[3], r[4], r[5]) for r in results]
        cursor.executemany("""
        INSERT INTO TSCNN (DATE, FileName, Position, Result, Reliability, BadType, BadReliability)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, insert_data)

        # 커밋하고 연결 종료
        conn.commit()
        conn.close()
        print(f"\n📁 결과 DB 저장 완료 → {db_connection_str}")
    else:
        print("⚠ 저장할 결과가 없습니다.")

# 실행 경로 설정
test_folder = "C:/Users/Jun/Desktop/anaconda/3.10python/final/test"  # 검사할 이미지 폴더
db_connection_str = "DRIVER={ODBC Driver 17 for SQL Server};SERVER=yhpro.tsst.kr;DATABASE=YHPRODB;UID=yhpro;PWD=yhpro1234#@!"

# 실행
results = predict_folder_images(ok_ng_model, test_folder)
save_results_to_db(results, db_connection_str)
