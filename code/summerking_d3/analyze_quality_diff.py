import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, root_mean_squared_error
import numpy as np

# ==========================================================
# 1️⃣ 경로 설정
# ==========================================================
rgb_csv = "C:/Users/FORYOUCOM/Documents/GitHub/fruit/output/rgb_summerking_d3.csv"
quality_excel = "C:/Users/FORYOUCOM/Desktop/스마트팜 창의 설계/윤여은 사과/(특허)썸머킹 품질예측/썸머킹 데이터-이미지-84제외.xlsx"

# 자동으로 base_dir 설정 (rgb_csv 기준)
base_dir = os.path.dirname(rgb_csv)

# ==========================================================
# 2️⃣ 데이터 불러오기 및 전처리
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
targets = ["weightloss", "ciel", "ciea", "cieb"]  # ✅ 여러 품질 변수 예측

# 컬럼명도 공백 제거되어 있으므로 소문자 버전으로 변환
merged.columns = [c.strip().lower().replace(" ", "") for c in merged.columns]

# ==========================================================
# 5️⃣ storage period별 RGB 평균 및 ΔRGB 계산 (diff 방식만)
# ==========================================================
df = merged.dropna(subset=features + targets).copy()

# storage period별 평균 계산
grouped = (
    df.groupby("storageperiod")[["r_mean", "g_mean", "b_mean"] + targets]
    .mean()
    .reset_index()
    .sort_values("storageperiod")
    .reset_index(drop=True)
)

# --- (A) 구간별 변화량 (diff 방식)
grouped["delta_r_diff"] = grouped["r_mean"].diff().fillna(0)
grouped["delta_g_diff"] = grouped["g_mean"].diff().fillna(0)
grouped["delta_b_diff"] = grouped["b_mean"].diff().fillna(0)

print("\n📊 ΔRGB 기반 평균 데이터 (diff 방식)")
print(grouped)

# ==========================================================
# 6️⃣ 학습 데이터 구성
# ==========================================================
X = grouped[["delta_r_diff", "delta_g_diff", "delta_b_diff", "storageperiod"]]

results = []

# ==========================================================
# 7️⃣ 타깃별 모델 학습 및 평가
# ==========================================================
for target in targets:
    print(f"\n🎯 Target: {target}")

    df_target = grouped.dropna(subset=X.columns.tolist() + [target]).copy()
    X_train, X_test, y_train, y_test = train_test_split(
        df_target[X.columns], df_target[target], test_size=0.3, random_state=42
    )

    # 머신러닝 모델 5종 구성
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
        "SVR": SVR(kernel="rbf", C=10, epsilon=0.1),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "GradientBoosting": GradientBoostingRegressor(random_state=42)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)
        rmse = root_mean_squared_error(y_test, preds)

        results.append({
            "Target": target,
            "Model": name,
            "R2": round(r2, 4),
            "RMSE": round(rmse, 4)
        })
        print(f"  ▶ {name:<16} | R² = {r2:.4f} | RMSE = {rmse:.4f}")

# ==========================================================
# 8️⃣ 결과 저장
# ==========================================================
result_df = pd.DataFrame(results)
output_dir = os.path.join(base_dir, "output")
os.makedirs(output_dir, exist_ok=True)

save_path = os.path.join(output_dir, "diff_summerking.csv")
result_df.to_csv(save_path, index=False, encoding="utf-8-sig")

print(f"\n✅ ΔRGB(diff) 기반 품질 예측 완료! 결과 저장됨: {save_path}")
print(result_df)
