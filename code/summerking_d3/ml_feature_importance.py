import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================================
# 1️⃣ 경로 설정
# ==========================================================
rgb_csv = "C:/Users/FORYOUCOM/Documents/GitHub/fruit/output/rgb_summerking_d3.csv"
quality_excel = "C:/Users/FORYOUCOM/Desktop/스마트팜 창의 설계/윤여은 사과/(특허)썸머킹 품질예측/썸머킹 데이터-이미지-84제외.xlsx"

base_dir = os.path.dirname(rgb_csv)

# ==========================================================
# 2️⃣ 데이터 불러오기 및 전처리
# ==========================================================
rgb_df = pd.read_csv(rgb_csv)
quality_df = pd.read_excel(quality_excel)
quality_df.columns = [c.strip().lower().replace(" ", "").replace("\xa0", "") for c in quality_df.columns]

# 'no' 컬럼 탐색
no_col = None
for c in quality_df.columns:
    if c in ["no", "no.", "번호", "id", "image_no", "num"]:
        no_col = c
        break
if no_col is None:
    raise KeyError("⚠️ 엑셀에서 'NO.' 또는 '번호' 컬럼을 찾을 수 없습니다.")

quality_df["file_name"] = quality_df[no_col].astype(str) + ".jpg"

# ==========================================================
# 3️⃣ 데이터 병합
# ==========================================================
merged = pd.merge(rgb_df, quality_df, on="file_name", how="inner")
print(f"✅ 병합 완료: {len(merged)}개 데이터 남음")

features = ["r_mean", "g_mean", "b_mean", "storageperiod"]
target = "weightloss"  # ✅ weightloss만 남김

merged.columns = [c.strip().lower().replace(" ", "") for c in merged.columns]

# ==========================================================
# 4️⃣ RGB 변화량 계산
# ==========================================================
df = merged.dropna(subset=features + [target]).copy()
grouped = (
    df.groupby("storageperiod")[["r_mean", "g_mean", "b_mean", target]]
    .mean()
    .reset_index()
    .sort_values("storageperiod")
    .reset_index(drop=True)
)

# ΔRGB (3일 간격)
grouped["delta_r_diff"] = grouped["r_mean"].diff().fillna(0)
grouped["delta_g_diff"] = grouped["g_mean"].diff().fillna(0)
grouped["delta_b_diff"] = grouped["b_mean"].diff().fillna(0)

# 누적 ΔRGB (0일 대비)
base_r, base_g, base_b = grouped.loc[0, ["r_mean", "g_mean", "b_mean"]]
grouped["delta_r_total"] = grouped["r_mean"] - base_r
grouped["delta_g_total"] = grouped["g_mean"] - base_g
grouped["delta_b_total"] = grouped["b_mean"] - base_b

# ==========================================================
# 5️⃣ Feature Importance 계산
# ==========================================================
feature_importance_list = []

# 입력 방식 3종
modes = {
    "RGB": ["r_mean", "g_mean", "b_mean", "storageperiod"],
    "Diff": ["delta_r_diff", "delta_g_diff", "delta_b_diff", "storageperiod"],
    "Total": ["delta_r_total", "delta_g_total", "delta_b_total", "storageperiod"],
}

for mode_name, feat_cols in modes.items():
    print(f"\n🔹 Feature Importance 계산 중: {mode_name}")

    df_target = grouped.dropna(subset=feat_cols + [target]).copy()
    X_train, X_test, y_train, y_test = train_test_split(
        df_target[feat_cols], df_target[target], test_size=0.3, random_state=42
    )

    # 트리 기반 모델만 사용
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        importance = model.feature_importances_
        for f, imp in zip(feat_cols, importance):
            feature_importance_list.append({
                "Mode": mode_name,
                "Model": name,
                "Feature": f,
                "Importance": round(float(imp), 4)
            })
        print(f"  ▶ {mode_name:<6} | {name:<16} 완료")

# ==========================================================
# 6️⃣ 결과 저장 + 시각화
# ==========================================================
output_dir = os.path.join(base_dir, "output")
os.makedirs(output_dir, exist_ok=True)

fi_df = pd.DataFrame(feature_importance_list)
save_path_fi = os.path.join(output_dir, "feature_importance_weightloss.csv")
fi_df.to_csv(save_path_fi, index=False, encoding="utf-8-sig")

print(f"\n✅ Feature Importance 저장 완료: {save_path_fi}")

# Feature 이름 보기 좋게 변경
fi_df["Feature"] = fi_df["Feature"].replace({
    "r_mean": "R_mean", "g_mean": "G_mean", "b_mean": "B_mean",
    "delta_r_diff": "ΔR_diff", "delta_g_diff": "ΔG_diff", "delta_b_diff": "ΔB_diff",
    "delta_r_total": "ΔR_total", "delta_g_total": "ΔG_total", "delta_b_total": "ΔB_total",
    "storageperiod": "Storage Period"
})

# ==========================================================
# 7️⃣ 시각화 (모드별 그래프 3개로 분리)
# ==========================================================
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", font="Arial", font_scale=1.1)

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
modes = ["RGB", "Diff", "Total"]
colors = ["#4DB6AC", "#FFB74D", "#64B5F6"]

for ax, mode, color in zip(axes, modes, colors):
    sub = fi_df[fi_df["Mode"] == mode]
    sns.barplot(
        data=sub,
        x="Feature",
        y="Importance",
        hue="Model",   # RandomForest vs GradientBoosting 구분
        palette="muted",
        ax=ax
    )
    ax.set_title(f"{mode} Model", fontsize=13, weight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Importance" if mode == "RGB" else "")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="Model", loc="upper right")

fig.suptitle("Feature Importance for Weight Loss Prediction (by Input Mode)", fontsize=15, weight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.95])

# 저장
save_path_fig3 = os.path.join(output_dir, "feature_importance_weightloss_by_mode.png")
plt.savefig(save_path_fig3, dpi=300)
plt.show()

print(f"✅ 3분할 그래프 저장 완료: {save_path_fig3}")

