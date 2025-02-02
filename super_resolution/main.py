import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 1. 이미지 로드 및 전처리
image_path = "grayscale_mammogram_x_ray_image_showing_breast_tissue_with_moderate_contrast_suitable_for_medical_analysis.png"
low_res = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Grayscale 로드
low_res = low_res.astype(np.float32) / 255.0  # 정규화
low_res_rgb = np.stack((low_res,) * 3, axis=-1)  # Grayscale → RGB 변환
low_res_expanded = np.expand_dims(low_res_rgb, axis=0)  # 배치 차원 추가

# 2. EDSR 모델 로드 (TensorFlow Hub)
model_url = "https://tfhub.dev/captain-pool/esrgan-tf2/1"
sr_model = hub.load(model_url)

# 3. Super Resolution 수행
sr_output = sr_model(tf.convert_to_tensor(low_res_expanded))
sr_image = tf.squeeze(sr_output).numpy()  # 배치 차원 제거
sr_image = np.clip(sr_image, 0, 1)  # 값 범위 유지

# ✅ Grayscale 변환 (R, G, B 채널 평균)
sr_grayscale = np.mean(sr_image, axis=-1)  # RGB → Grayscale

# 4. 결과 시각화
plt.figure(figsize=(15, 8))

# 원본 (저해상도)
plt.subplot(1, 2, 1)
plt.imshow(low_res, cmap="gray")  # Grayscale로 표시
plt.title("Low Resolution (Original)")
plt.axis("off")

# Super Resolution (회색조)
plt.subplot(1, 2, 2)
plt.imshow(sr_grayscale, cmap="gray")  # Grayscale로 표시
plt.title("Super Resolution (Enhanced)")
plt.axis("off")

plt.tight_layout()
plt.show()
