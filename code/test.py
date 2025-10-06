import json
import numpy as np
import cv2
from pycocotools import mask as maskUtils
import matplotlib.pyplot as plt
import os

# ====== 🔹 1. 경로 설정 ======
base_dir = os.path.dirname(os.path.dirname(__file__))  # project_root 기준
json_path = os.path.join(base_dir, "json", "instances_default.json")
img_dir = os.path.join(base_dir, "image")

# ====== 🔹 2. JSON 파일 불러오기 ======
with open(json_path, "r", encoding="utf-8") as f:
    coco = json.load(f)

# ====== 🔹 3. 이미지 목록 ======
images = {img["id"]: img["file_name"] for img in coco["images"]}

# ====== 🔹 4. 테스트할 이미지 선택 (첫 번째 예시)
img_id = list(images.keys())[0]  # 첫 번째 이미지
file_name = images[img_id]
print(f"현재 확인 중: {file_name}")

# ====== 🔹 5. 해당 이미지의 annotation 불러오기 ======
anns = [ann for ann in coco["annotations"] if ann["image_id"] == img_id]

# ====== 🔹 6. 원본 이미지 불러오기 ======
img_path = os.path.join(img_dir, file_name)
img = cv2.imread(img_path)

if img is None:
    raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {img_path}")

mask_total = np.zeros(img.shape[:2], dtype=np.uint8)

# ====== 🔹 7. annotation 마스크 합성 ======
for ann in anns:
    rle = maskUtils.frPyObjects(ann["segmentation"], img.shape[0], img.shape[1])
    m = maskUtils.decode(rle)
    mask_total = np.maximum(mask_total, m * 255)

# ====== 🔹 8. 시각화 ======
plt.figure(figsize=(10,4))

# (1) 마스크만 표시
plt.subplot(1,2,1)
plt.title("Mask only")
plt.imshow(mask_total, cmap="gray")
plt.axis("off")

# (2) 원본 위에 마스크 오버레이
overlay = img.copy()
overlay[mask_total > 0] = (0, 255, 0)  # 마스크 부분 초록색 표시
plt.subplot(1,2,2)
plt.title("Overlay on original")
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.tight_layout()
plt.show()
