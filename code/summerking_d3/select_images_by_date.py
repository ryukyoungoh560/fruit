import os
import shutil
import pandas as pd
import math

# ===========================================
# 1️⃣ 경로 설정
# ===========================================
base_dir = "C:/Users/FORYOUCOM/Desktop/스마트팜 창의 설계/윤여은 사과/(특허)썸머킹 품질예측"

excel_path = os.path.join(base_dir, "썸머킹 데이터-이미지-84제외.xlsx")
img_dir = os.path.join(base_dir, "사과 품질예측 사진(썸머킹)")
output_dir = os.path.join(base_dir, "3일간격 사과 품질 예측 사진")

os.makedirs(output_dir, exist_ok=True)

# ===========================================
# 2️⃣ 엑셀 불러오기
# ===========================================
df = pd.read_excel(excel_path)

# 컬럼명 통일
df.columns = [c.strip().lower().replace(" ", "").replace("\xa0", "") for c in df.columns]

# 'no' 또는 'no.' 컬럼 탐색
no_col = None
for c in df.columns:
    if c in ["no", "no.", "번호"]:
        no_col = c
        break

if no_col is None:
    raise ValueError("❌ 엑셀에 'NO.' 또는 '번호' 컬럼이 없습니다.")

# 파일명 컬럼 생성
df["file_name"] = df[no_col].astype(str) + ".jpg"

# ===========================================
# 3️⃣ 3일 간격 그룹 계산
# ===========================================
if "storageperiod" not in df.columns:
    raise ValueError("❌ 엑셀에 'storage period' 컬럼이 없습니다. 이름을 확인하세요.")

# 각 행을 3일 단위로 그룹핑
df["period_group"] = df["storageperiod"].apply(lambda x: int(math.floor(x / 3)) * 3)

# ===========================================
# 4️⃣ 각 그룹별로 상위 5개씩 선택
# ===========================================
selected_files = []

for group, sub_df in df.groupby("period_group"):
    sub_sorted = sub_df.sort_values(by=no_col)
    subset = sub_sorted.head(5)  # 상위 5개만 선택
    selected_files.extend(subset["file_name"].tolist())

    # 출력 폴더 생성
    group_folder = os.path.join(output_dir, f"{group}일")
    os.makedirs(group_folder, exist_ok=True)

    # 이미지 복사
    for file_name in subset["file_name"]:
        src = os.path.join(img_dir, file_name)
        dst = os.path.join(group_folder, file_name)

        if os.path.exists(src):
            shutil.copy(src, dst)
        else:
            print(f"⚠️ 이미지 없음: {src}")

print(f"\n✅ 총 {len(selected_files)}개의 이미지가 3일 간격으로 복사되었습니다.")
print(f"💾 저장 경로: {output_dir}")
