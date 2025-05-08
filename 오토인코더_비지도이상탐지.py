import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# 🔤 한글 깨짐 방지 설정
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Cropping2D
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# 이미지 경로 설정 (현재 실행 디렉토리 기준)
img_dir = './train/NORMAL'
img_size = (150, 150)

# ✅ 이미지 불러오기
print("이미지 불러오는 중...")
img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpeg')]
x_data = []

for fname in img_files:
    img_path = os.path.join(img_dir, fname)
    img = load_img(img_path, target_size=img_size)
    img_array = img_to_array(img) / 255.0  # 정규화
    x_data.append(img_array)

x_data = np.array(x_data)
print(f"총 이미지 수: {len(x_data)}")

# ✅ 오토인코더 모델 정의
input_img = Input(shape=(150, 150, 3))

# 인코더
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# 디코더
x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Cropping2D(((1, 1), (1, 1)))(x)  # 크기 맞추기: 152→150

decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

# 모델 컴파일
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# ✅ 모델 학습
print("모델 학습 중...")
autoencoder.fit(x_data, x_data, epochs=10, batch_size=32, shuffle=True)
print("✅ 학습 완료!")

# ✅ 복원 이미지 생성
decoded_imgs = autoencoder.predict(x_data)

def plot_xai_overlay(original_imgs, reconstructed_imgs, n=5):
    

    plt.figure(figsize=(12, 6))
    for i in range(n):
        orig = original_imgs[i]
        recon = reconstructed_imgs[i]

        # 복원 차이 계산 (heatmap용)
        diff = np.abs(orig - recon)
        diff_mean = np.mean(diff, axis=-1)

        # 원본 이미지
        ax = plt.subplot(2, n, i + 1)
        ax.imshow(orig)
        ax.axis("off")
        ax.set_title("원본")

        # 히트맵 오버레이
        ax = plt.subplot(2, n, i + 1 + n)
        ax.imshow(orig, cmap='gray')
        ax.imshow(diff_mean, cmap='hot', alpha=0.6)
        ax.axis("off")
        ax.set_title("복원 실패 Heatmap")

    plt.tight_layout()
    plt.show()

# 실행
plot_xai_overlay(x_data, decoded_imgs, n=5)




# 각 이미지별 재구성 오류(MSE) 계산
mse_errors = np.mean((x_data - decoded_imgs) ** 2, axis=(1,2,3))

# MSE값 출력 및 시각화
print("재구성 오류(MSE) 샘플:", mse_errors[:10])

# MSE 값 시각화
plt.figure(figsize=(8,4))
plt.hist(mse_errors, bins=50, color='skyblue')
plt.title("재구성 오류 (MSE) 분포")
plt.xlabel("MSE")
plt.ylabel("이미지 개수")
plt.grid(True)
plt.show()


# 상위 5% 이상치를 기준으로 설정
threshold = np.percentile(mse_errors, 95)
print("이상치 기준 threshold:", threshold)


# MSE가 threshold보다 큰 이미지 인덱스 찾기
anomaly_idx = np.where(mse_errors > threshold)[0]
print("이상치 개수:", len(anomaly_idx))


# 이상치 중 5개만 시각화
plt.figure(figsize=(15, 4))
for i, idx in enumerate(anomaly_idx[:5]):
    ax = plt.subplot(1, 5, i+1)
    plt.imshow(x_data[idx])
    plt.title(f"MSE: {mse_errors[idx]:.4f}")
    plt.axis("off")
plt.tight_layout()
plt.show()


# 폐렴 이미지 경로 설정
test_dir = './train/PNEUMONIA'  # 또는 './test/PNEUMONIA'

# 이미지 전처리
img_files = [f for f in os.listdir(test_dir) if f.endswith('.jpeg')]
test_data = []

for fname in img_files[:10]:  # 실험용으로 10장만 불러오기
    img_path = os.path.join(test_dir, fname)
    img = load_img(img_path, target_size=(150, 150))
    img_array = img_to_array(img) / 255.0
    test_data.append(img_array)

test_data = np.array(test_data)


# 폐렴 이미지를 복원시켜봄
reconstructed = autoencoder.predict(test_data)



# 재구성 오차 계산 (MSE)
mse_errors = np.mean((test_data - reconstructed) ** 2, axis=(1,2,3))

# 오차 출력
for i, error in enumerate(mse_errors):
    print(f"{i+1}번 폐렴 이미지 MSE 오차: {error:.4f}")



plt.figure(figsize=(15, 4))
for i in range(5):
    # 원본
    ax = plt.subplot(2, 5, i + 1)
    plt.imshow(test_data[i])
    plt.title("원본 폐렴")
    plt.axis("off")

    # 복원
    ax = plt.subplot(2, 5, i + 6)
    plt.imshow(reconstructed[i])
    plt.title(f"복원 (MSE: {mse_errors[i]:.4f})")
    plt.axis("off")

plt.tight_layout()
plt.show()


print("NaN 포함 여부:", np.isnan(x_data).any())






def load_images(folder_path, img_size=(150, 150)):
    images = []
    for fname in os.listdir(folder_path):
        if fname.endswith(".jpeg"):
            path = os.path.join(folder_path, fname)
            img = load_img(path, target_size=img_size)
            img_array = img_to_array(img) / 255.0
            images.append(img_array)
    return np.array(images)

# 경로 설정
normal_dir = './test/NORMAL'
pneumonia_dir = './test/PNEUMONIA'

# 이미지 불러오기
normal_x = load_images(normal_dir)
pneumonia_x = load_images(pneumonia_dir)

def calculate_mse(original, reconstructed):
    return np.mean((original - reconstructed) ** 2, axis=(1,2,3))


# 복원 이미지 생성
normal_recon = autoencoder.predict(normal_x)
pneumonia_recon = autoencoder.predict(pneumonia_x)

# MSE 계산
normal_mse_errors = calculate_mse(normal_x, normal_recon)
pneumonia_mse_errors = calculate_mse(pneumonia_x, pneumonia_recon)


print(f"정상 평균 MSE: {np.mean(normal_mse_errors):.6f}")
print(f"폐렴 평균 MSE: {np.mean(pneumonia_mse_errors):.6f}")




plt.figure(figsize=(10, 5))
plt.hist(normal_mse_errors, bins=50, alpha=0.7, label="정상")
plt.hist(pneumonia_mse_errors, bins=50, alpha=0.7, label="폐렴")
plt.axvline(np.percentile(normal_mse_errors, 95), color='red', linestyle='--', label="정상 기준 threshold")
plt.xlabel("MSE")
plt.ylabel("이미지 수")
plt.legend()
plt.title("정상 vs 폐렴 MSE 분포 비교")
plt.grid(True)
plt.show()

pneumonia_recon = autoencoder.predict(pneumonia_x)




# ✅ MSE 분포 시각화
plt.figure(figsize=(10, 5))
plt.hist(normal_mse_errors, bins=50, alpha=0.7, label='정상', color='skyblue')
plt.hist(pneumonia_mse_errors, bins=50, alpha=0.7, label='폐렴', color='orange')
plt.axvline(np.percentile(normal_mse_errors, 95), color='red', linestyle='--', label='정상 기준 threshold')

plt.xlabel("MSE")
plt.ylabel("이미지 수")
plt.title("정상 vs 폐렴 MSE 분포 비교")
plt.legend()
plt.grid(True)
plt.show()



threshold = np.percentile(normal_mse_errors, 95)
pneumonia_anomaly_count = np.sum(pneumonia_mse_errors > threshold)
total_pneumonia = len(pneumonia_mse_errors)

print(f"폐렴 중 이상치로 감지된 수: {pneumonia_anomaly_count} / {total_pneumonia}")
print(f"감지율: {pneumonia_anomaly_count / total_pneumonia * 100:.2f}%")





def plot_xai_heatmap(original_imgs, reconstructed_imgs, mse_errors, n=5):
    plt.figure(figsize=(15, 6))

    for i in range(n):
        orig = original_imgs[i]
        recon = reconstructed_imgs[i]
        mse = mse_errors[i]

        # 복원 오류 계산
        diff = np.abs(orig - recon)
        diff_mean = np.mean(diff, axis=-1)

        # 원본
        ax = plt.subplot(n, 3, i*3 + 1)
        plt.imshow(orig)
        plt.title("원본")
        plt.axis("off")

        # 복원
        ax = plt.subplot(n, 3, i*3 + 2)
        plt.imshow(recon)
        plt.title("복원")
        plt.axis("off")

        # 복원 실패 영역 heatmap
        ax = plt.subplot(n, 3, i*3 + 3)
        plt.imshow(diff_mean, cmap='hot')
        plt.title(f"복원 실패 Heatmap\nMSE: {mse:.4f}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

    plot_xai_heatmap(pneumonia_x, pneumonia_recon, pneumonia_mse_errors)


def plot_xai_overlay(original_imgs, reconstructed_imgs, n=5):
    
    

    plt.figure(figsize=(12, 6))

    for i in range(n):
        orig = original_imgs[i]
        recon = reconstructed_imgs[i]

        # 복원 차이 계산 (heatmap용)
        diff = np.abs(orig - recon)
        diff_mean = np.mean(diff, axis=-1)

        # 히트맵 오버레이를 위해 alpha 투명도와 cmap 설정
        ax = plt.subplot(2, n, i + 1)
        ax.imshow(orig)
        ax.axis('off')
        ax.set_title("원본")

        ax = plt.subplot(2, n, i + 1 + n)
        ax.imshow(orig, cmap='gray')
        ax.imshow(diff_mean, cmap='hot', alpha=0.6)  # 원본 위에 heatmap 덮기
        ax.axis('off')
        ax.set_title("복원 실패 Heatmap")

    plt.tight_layout()
    plt.show()

plot_xai_overlay(pneumonia_x, pneumonia_recon, n=5)

