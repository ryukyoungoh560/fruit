import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# ==========================================================
# 1️⃣ 경로 설정
# ==========================================================
base_dir = os.path.dirname(os.path.dirname(__file__))  # 프로젝트 루트
rgb_csv = os.path.join(base_dir, "output", "rgb_data_summerking_gray.csv")

# 품질 데이터 엑셀 경로 (사용자 지정)
quality_excel = "C:/Users/FORYOUCOM/Desktop/스마트팜 창의 설계/윤여은 사과/(특허)썸머킹 품질예측/썸머킹 데이터-이미지-84제외.xlsx"

# ==========================================================
# 2️⃣ 데이터 불러오기
# ==========================================================
rgb_df = pd.read_csv(rgb_csv)
quality_df = pd.read_excel(quality_excel)

# --- 컬럼 이름 소문자 + 공백/기호 제거
quality_df.columns = [c.strip().lower().replace(" ", "").replace("\xa0", "") for c in quality_df.columns]

# --- 'no' 컬럼 자동 탐색
no_col = None
for c in quality_df.columns:
    if c in ["no", "no.", "번호", "id", "image_no", "num"]:
        no_col = c
        break

if no_col is None:
    raise KeyError(f"⚠️ 엑셀에서 'NO.' 또는 '번호' 컬럼을 찾을 수 없습니다. 현재 컬럼들: {quality_df.columns.tolist()}")

# --- 파일 이름 생성
quality_df["file_name"] = quality_df[no_col].astype(str) + ".jpg"

# ==========================================================
# 3️⃣ 데이터 병합
# ==========================================================
merged = pd.merge(rgb_df, quality_df, on="file_name", how="inner")
print(f"✅ 병합 완료: {len(merged)}개 데이터 남음\n")
print(merged.head())

# ==========================================================
# 4️⃣ 입력 / 출력 설정
# ==========================================================
features = ["r_mean", "g_mean", "b_mean", "storageperiod"]
targets = ["weightloss", "ciel", "ciea", "cieb"]

# 컬럼명도 공백 제거되어 있으므로 소문자 버전으로 변환
merged.columns = [c.strip().lower().replace(" ", "") for c in merged.columns]

# ==========================================================
# 5️⃣ 모델 평가 함수
# ==========================================================
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    return r2, rmse

# ==========================================================
# 6️⃣ 타깃별로 모델 실행
# ==========================================================
results = []

for target in targets:
    print(f"\n🎯 Target: {target}")

    df = merged.dropna(subset=features + [target])
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
        "LinearRegression": LinearRegression()
    }

    for name, model in models.items():
        r2, rmse = evaluate_model(model, X_train, X_test, y_train, y_test)
        results.append({
            "Target": target,
            "Model": name,
            "R2": round(r2, 4),
            "RMSE": round(rmse, 4)
        })
        print(f"  ▶ {name:<16} | R² = {r2:.4f} | RMSE = {rmse:.4f}")

# ==========================================================
# 7️⃣ 결과 저장
# ==========================================================
result_df = pd.DataFrame(results)
output_dir = os.path.join(base_dir, "output")
os.makedirs(output_dir, exist_ok=True)

save_path = os.path.join(output_dir, "model_results_summerking2.csv")
result_df.to_csv(save_path, index=False, encoding="utf-8-sig")

print(f"\n✅ 모든 모델 평가 완료! 결과 저장됨: {save_path}")
print(result_df)
