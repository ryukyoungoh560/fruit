import json
import numpy as np
import cv2
from pycocotools import mask as maskUtils
import pandas as pd
import os
from tqdm import tqdm

# ========================================
# 1️⃣ 경로 설정 (A 방식: base_dir을 한 단계 위로)
# ========================================
# 현재 파일 경로: fruit/code/summerking_d3/extract_rgb_cali.py
# 목표 base_dir: fruit/
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  

json_path = os.path.join(base_dir, "json", "instances_default_d3.json")
img_dir = os.path.join(base_dir, "image", "summerking_d3")
output_dir = os.path.join(base_dir, "output")
os.makedirs(output_dir, exist_ok=True)

print(f"📁 base_dir: {base_dir}")
print(f"📄 JSON 경로: {json_path}")
print(f"🖼️ 이미지 폴더: {img_dir}")
print(f"💾 출력 폴더: {output_dir}\n")

# ========================================
# 2️⃣ JSON 파일 불러오기
# ========================================
with open(json_path, "r", encoding="utf-8") as f:
    coco = json.load(f)

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
# 🌈 4️⃣ Gray World 보정 함수
# ========================================
def gray_world_correction(img: np.ndarray) -> np.ndarray:
    """
    Gray World Algorithm:
    전체 이미지의 평균 R,G,B를 같게 맞추어 색온도(조명 영향)를 보정합니다.
    """
    if img is None or img.size == 0:
        return img

    img = img.astype(np.float32)
    mean_per_channel = np.mean(img, axis=(0, 1))  # [B, G, R]
    gray_mean = np.mean(mean_per_channel)

    gain = gray_mean / mean_per_channel
    img_corrected = img * gain
    img_corrected = np.clip(img_corrected, 0, 255).astype(np.uint8)
    return img_corrected

# ========================================
# 5️⃣ 이미지별 RGB 계산
# ========================================
data = []
print(f"\n🍎 총 {len(images)}장의 이미지에서 Gray World 보정 후 fruit 영역 RGB를 추출합니다...\n")

annotations = coco.get("annotations", [])
if not annotations:
    raise ValueError("⚠️ JSON 파일에 'annotations'가 비어 있습니다. segmentation 포함 여부 확인 필요.")

for img_id, file_name in tqdm(images.items(), desc="Processing images"):
    img_path = os.path.join(img_dir, file_name)
    if not os.path.exists(img_path):
        print(f"⚠️ 이미지 파일을 찾을 수 없습니다: {img_path}")
        continue

    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ 이미지 로드 실패: {img_path}")
        continue

    height, width = img.shape[:2]

    # --- Gray World 색상 보정 적용
    img_corrected = gray_world_correction(img)

    # --- fruit annotation만 필터링
    anns = [ann for ann in annotations if ann["image_id"] == img_id and ann["category_id"] == fruit_cat_id]
    if not anns:
        print(f"⚠️ fruit 라벨 annotation 없음: {file_name}")
        continue

    # --- 마스크 합성
    mask_total = np.zeros((height, width), dtype=np.uint8)
    for ann in anns:
        seg = ann.get("segmentation", [])
        if not seg:
            continue
        if isinstance(seg, dict) and "counts" in seg:
            m = maskUtils.decode(seg)
        else:
            rle = maskUtils.frPyObjects(seg, height, width)
            m = maskUtils.decode(rle)
        if m.ndim == 3:
            m = np.sum(m, axis=2)
        mask_total = np.maximum(mask_total, m.astype(np.uint8))


    # --- fruit 영역 픽셀 추출
    fruit_pixels = img_corrected[mask_total.astype(bool)]
    if fruit_pixels.size == 0:
        print(f"⚠️ fruit 영역 픽셀 없음: {file_name}")
        continue

    # --- 평균 RGB 계산 (OpenCV는 BGR → RGB 변환)
    mean_bgr = np.mean(fruit_pixels, axis=0)
    mean_rgb = mean_bgr[::-1]

    data.append({
        "file_name": os.path.basename(file_name),
        "R_mean": round(float(mean_rgb[0]), 3),
        "G_mean": round(float(mean_rgb[1]), 3),
        "B_mean": round(float(mean_rgb[2]), 3)
    })

# ========================================
# 6️⃣ 결과 저장
# ========================================
df = pd.DataFrame(data)
csv_path = os.path.join(output_dir, "rgb_summerking_d3_gray.csv")
df.to_csv(csv_path, index=False, encoding="utf-8-sig")

print(f"\n✅ Gray World 색상 보정 + RGB 추출 완료!")
print(f"📊 총 {len(df)}개 이미지의 fruit 평균 RGB 계산 완료.")
print(f"💾 저장 경로: {csv_path}")
