import json
import numpy as np
import cv2
from pycocotools import mask as maskUtils
import pandas as pd
import os
from tqdm import tqdm

# ========================================
# 1️⃣ 경로 설정
# ========================================
base_dir = os.path.dirname(os.path.dirname(__file__))  # 프로젝트 루트 기준
json_path = os.path.join(base_dir, "json", "instances_default.json")
img_dir = os.path.join(base_dir, "image")
output_dir = os.path.join(base_dir, "output")
os.makedirs(output_dir, exist_ok=True)

# ========================================
# 2️⃣ JSON 파일 불러오기
# ========================================
with open(json_path, "r", encoding="utf-8") as f:
    coco = json.load(f)

# 이미지 정보 매핑
images = {img["id"]: img["file_name"] for img in coco["images"]}

# ========================================
# 3️⃣ fruit 라벨(category_id) 찾기
# ========================================
fruit_cat_id = None
for cat in coco["categories"]:
    if cat["name"].lower() == "fruit":
        fruit_cat_id = cat["id"]
        break

if fruit_cat_id is None:
    raise ValueError("⚠️ 'fruit' 라벨을 JSON에서 찾을 수 없습니다. 카테고리 이름 확인 필요.")

# ========================================
# 4️⃣ 이미지별 RGB 계산
# ========================================
data = []

print(f"\n🍎 총 {len(images)}장의 이미지에서 fruit 영역 RGB를 추출합니다...\n")

for img_id, file_name in tqdm(images.items()):
    # JSON에 포함된 상대경로 그대로 사용
    img_path = os.path.join(img_dir, file_name)  

    if not os.path.exists(img_path):
        print(f"⚠️ 이미지 파일을 찾을 수 없습니다: {img_path}")
        continue

    # 이미지 읽기
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ 이미지 로드 실패: {img_path}")
        continue

    height, width = img.shape[:2]

    # fruit annotation만 필터링
    anns = [ann for ann in coco["annotations"]
            if ann["image_id"] == img_id and ann["category_id"] == fruit_cat_id]

    if len(anns) == 0:
        print(f"⚠️ fruit 라벨 annotation 없음: {file_name}")
        continue

    # 마스크 합성
    mask_total = np.zeros((height, width), dtype=np.uint8)
    for ann in anns:
        rle = maskUtils.frPyObjects(ann["segmentation"], height, width)
        m = maskUtils.decode(rle)
        mask_total = np.maximum(mask_total, m)

    # fruit 영역 픽셀 추출
    fruit_pixels = img[mask_total.astype(bool)]

    if fruit_pixels.size == 0:
        print(f"⚠️ fruit 영역 픽셀 없음: {file_name}")
        continue

    # 평균 RGB 계산 (OpenCV는 BGR → RGB 순서 변환)
    mean_bgr = np.mean(fruit_pixels, axis=0)
    mean_rgb = mean_bgr[::-1]

    data.append({
        "file_name": os.path.basename(file_name),
        "R_mean": round(mean_rgb[0], 3),
        "G_mean": round(mean_rgb[1], 3),
        "B_mean": round(mean_rgb[2], 3)
    })

# ========================================
# 5️⃣ 결과 저장
# ========================================
df = pd.DataFrame(data)
csv_path = os.path.join(output_dir, "rgb_data_summerking1.csv")
df.to_csv(csv_path, index=False, encoding="utf-8-sig")

print(f"\n✅ RGB 추출 완료! 총 {len(df)}개 이미지의 fruit 평균 RGB를 계산했습니다.")
print(f"💾 저장 경로: {csv_path}")
