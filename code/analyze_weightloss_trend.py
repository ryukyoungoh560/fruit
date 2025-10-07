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
rgb_csv = os.path.join(base_dir, "output", "rgb_data_summerking1.csv")

# 품질 데이터 엑셀 경로 (사용자 지정)
quality_excel = "C:/Users/FORYOUCOM/Desktop/스마트팜 창의 설계/윤여은 사과/(특허)썸머킹 품질예측/썸머킹 데이터-이미지-84제외.xlsx"

# ==========================================================
# 2️⃣ 데이터 불러오기 및 전처리
# ==========================================================
rgb_df = pd.read_csv(rgb_csv)
quality_df = pd.read_excel(quality_excel)

# 컬럼명 통일
quality_df.columns = [c.strip().lower().replace(" ", "").replace("\xa0", "") for c in quality_df.columns]

# 'no.' 또는 '번호' 컬럼 탐색
no_col = None
for c in quality_df.columns:
    if c in ["no", "no.", "번호", "id", "image_no", "num"]:
        no_col = c
        break
if no_col is None:
    raise KeyError(f"⚠️ 엑셀에서 'NO.' 또는 '번호' 컬럼을 찾을 수 없습니다. 현재 컬럼들: {quality_df.columns.tolist()}")

quality_df["file_name"] = quality_df[no_col].astype(str) + ".jpg"

# 병합
merged = pd.merge(rgb_df, quality_df, on="file_name", how="inner")
merged.columns = [c.strip().lower().replace(" ", "") for c in merged.columns]

# 필요한 컬럼만 추출
cols_needed = ["file_name", "r_mean", "g_mean", "b_mean", "storageperiod", "weightloss"]
df = merged[cols_needed].dropna().copy()

# ==========================================================
# 3️⃣ storage period별 RGB 평균 및 ΔRGB 계산
# ==========================================================
grouped = df.groupby("storageperiod")[["r_mean", "g_mean", "b_mean", "weightloss"]].mean().reset_index()
grouped = grouped.sort_values("storageperiod").reset_index(drop=True)

# ΔRGB 계산
grouped["delta_r"] = grouped["r_mean"].diff().fillna(0)
grouped["delta_g"] = grouped["g_mean"].diff().fillna(0)
grouped["delta_b"] = grouped["b_mean"].diff().fillna(0)

# ΔRGB 포함한 최종 데이터 확인
print("\n📊 ΔRGB 기반 평균 데이터")
print(grouped)

# ==========================================================
# 4️⃣ 학습 데이터 구성
# ==========================================================
X = grouped[["delta_r", "delta_g", "delta_b", "storageperiod"]]
y = grouped["weightloss"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

models = {
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
    "LinearRegression": LinearRegression()
}

# ==========================================================
# 5️⃣ 모델 학습 및 평가
# ==========================================================
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)

    results.append({
        "Model": name,
        "R2": round(r2, 4),
        "RMSE": round(rmse, 4)
    })
    print(f"\n🎯 {name} 결과")
    print(f"R² = {r2:.4f}, RMSE = {rmse:.4f}")

# ==========================================================
# 6️⃣ 결과 저장
# ==========================================================
result_df = pd.DataFrame(results)
output_dir = os.path.join(base_dir, "output")
os.makedirs(output_dir, exist_ok=True)

save_path = os.path.join(output_dir, "model_weightloss_trend.csv")
result_df.to_csv(save_path, index=False, encoding="utf-8-sig")

print(f"\n✅ ΔRGB 기반 weight loss 예측 완료! 결과 저장됨: {save_path}")
print(result_df)
