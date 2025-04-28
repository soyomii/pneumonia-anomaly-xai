# 폐렴 이상 탐지 프로젝트 (Autoencoder + XAI)

정상 흉부 X-ray 영상을 오토인코더(Autoencoder)로 학습한 뒤,  
폐렴 환자 이미지를 복원하면서 생기는 오류(MSE)를 기반으로 이상을 감지하는 프로젝트입니다.  
복원 실패한 부분을 Heatmap으로 시각화해, 이상 영역을 직관적으로 볼 수 있도록 했습니다.

---

## 사용한 기술
- Python
- TensorFlow, Keras, NumPy, Matplotlib
- VSCode, Git, GitHub

---

## 진행한 작업

- 정상 이미지만 학습하는 오토인코더 구성
- 폐렴 이미지 복원 → 재구성 오류(MSE) 계산
- 오류가 큰 영역을 Heatmap으로 시각화
- 정상과 폐렴 환자의 MSE 분포 비교

---

## 폴더 구성
- `오토인코더_비지도이상탐지.py` : 모델 학습 및 이상 탐지 전체 코드
- `project/train/` : 학습용 X-ray 이미지
- `project/test/` : 테스트용 X-ray 이미지

---

## 추가 개선 아이디어
- 다양한 오토인코더 구조 실험 (예: 딥러닝 레이어 추가)
- Threshold를 자동화하는 방법 검토
- 데이터셋 확대 및 증강 적용

---

