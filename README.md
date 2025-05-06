# 폐렴 이상 탐지 프로젝트 (Autoencoder + XAI)

<p align="center">
  <img src="https://github.com/soyomii/pneumonia-anomaly-xai/assets/autoencoder_xai_heatmap_overlay_top5.png" width="600"/>
  <br/>
  <i>Autoencoder 기반 복원 실패 영역을 Heatmap으로 시각화한 예시</i>
</p>



# 폐렴 이상 탐지 프로젝트 (Autoencoder + XAI)

이 프로젝트는 정상 흉부 X-ray 이미지만을 학습한 오토인코더(Autoencoder)를 사용하여,  
폐렴 환자 이미지를 복원하면서 생기는 오류(MSE)를 이용해 비정상 이미지를 감지합니다.  
복원 실패한 부분을 **Heatmap**으로 시각화하여, 어떤 영역이 이상인지를 쉽게 볼 수 있도록 했습니다.

---

## 사용한 기술
- Python
- TensorFlow, Keras, NumPy, Matplotlib
- VSCode, Git, GitHub

---

## 프로젝트 목적
- **정상 이미지만 학습**하고, 학습한 내용에서 벗어난 이미지를 **비정상으로 판단**합니다. (이게 **비지도 학습**)
- 모델이 왜 그런 결정을 내렸는지 **설명할 수 있도록** **XAI**를 적용해, 결과를 쉽게 해석할 수 있게 했습니다.

---

## 진행한 작업
- **정상 데이터만** 학습하는 오토인코더 구성
- 폐렴 이미지를 복원하고, **재구성 오류(MSE)**를 계산하여 비정상 이미지를 탐지
- **복원 실패 영역을 Heatmap으로** 시각화하여 이상 부위를 직관적으로 확인
- **정상**과 **폐렴** 이미지를 비교하여 재구성 오류 분포 확인

---

## 폴더 구성
- `오토인코더_비지도이상탐지.py` : 모델 학습 및 이상 탐지 전체 코드
- `project/train/` : 학습용 X-ray 이미지
- `project/test/` : 테스트용 X-ray 이미지

---

## 추가 개선 아이디어
- 다양한 오토인코더 구조를 시도해 더 좋은 성능을 얻을 수 있습니다. (예: 더 깊은 네트워크 사용)
- **Threshold**를 자동화하는 방법을 추가하면, 이상 탐지가 더 객관적으로 이뤄질 수 있습니다.
- **데이터셋 확대** 및 **데이터 증강**을 통해 모델 성능을 더욱 향상시킬 수 있습니다.

---

## 기존 CNN 모델 vs XAI 모델
- **기존 CNN 모델**은 **라벨링된 데이터**를 학습하여 폐렴을 **분류**하는 **지도 학습** 방식이었습니다.
- **XAI 모델**은 **정상 이미지**만 학습하고, 이를 기준으로 **비정상 이미지를 탐지**하는 **비지도 학습** 방식입니다.
- **XAI 모델**은 **Heatmap**을 사용해 **모델의 판단 근거**를 시각적으로 **설명 가능한** 방식으로 제공합니다.  
  (즉, 모델이 어떤 부분을 이상으로 판단했는지 쉽게 알 수 있습니다.)

---

## 한계 및 고려사항
- **정상 이미지만** 학습하기 때문에, **정상 이미지와 유사한 비정상 이미지**를 잘못 탐지할 수 있습니다.
- **Threshold 설정**이 **수작업**으로 이루어져 있어, 자동화할 방법을 고민할 필요가 있습니다.
- 오토인코더 모델이 **세밀한 병변**까지 정확히 복원하는 데는 한계가 있을 수 있습니다.
- **Heatmap**은 복원 오류가 큰 부분을 시각화하지만, **모든 병변을 정확히 표현하는 것**은 아닙니다.

---


---

## 📊 시각화 결과

### 🔍 재구성 오류(MSE) 분포
<img src="https://github.com/soyomii/pneumonia-anomaly-xai/assets/autoencoder_mse_distribution_histogram.png" width="600"/>

### 🎯 정상 vs 폐렴 MSE 비교
<img src="https://github.com/soyomii/pneumonia-anomaly-xai/assets/mse_distribution_comparison_threshold.png" width="600"/>

### 🔥 복원 실패 Heatmap 예시 (정상 / 폐렴)
<img src="https://github.com/soyomii/pneumonia-anomaly-xai/assets/mse_distribution_threshold_refined.png" width="700"/>

### 📸 복원 비교: 원본 vs Autoencoder
<img src="https://github.com/soyomii/pneumonia-anomaly-xai/assets/pneumonia_reconstruction_comparison.png" width="750"/>

### ⚠️ 높은 MSE 순서 Top5 샘플
<img src="https://github.com/soyomii/pneumonia-anomaly-xai/assets/top5_high_mse_pneumonia_samples.png" width="750"/>



