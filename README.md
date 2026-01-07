# 🛰️ Super-Resolution Model Comparison for Satellite Imagery

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

위성영상 도메인에서 4가지 Super-Resolution 모델의 성능을 비교 분석한 프로젝트입니다.

---

##  핵심 발견 (Key Findings)

> **"SR 모델의 성능은 모델 구조보다 열화 가정(Degradation Assumption)에 훨씬 민감하다"**
- 이미지 복원 문제에서, 모델의 아키텍처보다 학습할 때 사용한 열화모델(이미지가 어떻게 손상되는지에 대한 가정)이 결과 성능에 보다 큰 영향을 준다는 뜻입니다.
- 모델은 주어진 열화 분포에 맞춰 특정한 보정 방법을 학습합니다. 학습 시 사용한 열화와 인풋 데이터의 열화방식이 일치해야 잘 동작합니다.
- 따라서, 위성분야의 다운스트림 태스크에서 사용할 모델 선정시 **도메인 데이터의 열화방식을 고려해 모델을 선정해야 한다는 결론을 얻었습니다**.

###  실험 결과

| 데이터 열화 타입 | Classical SR (EDSR, SwinIR-M) | Real-SR (Real-ESRGAN, HAT) |
|----------|------------------------------|---------------------------|
| **Bicubic 열화** | **28.4 dB** ✅ | 23.2 dB | 
| **센서/현실 열화** | 9.4 dB  | **18.0 dB** ✅ | 

**핵심 인사이트:**
- Classical SR 은 모델 학습시 주로 Bicubic 다운샘플링으로 학습된 모델 (예: EDSR, SwinIR-M)
- Real-SR: 실제 세계의 복합 열화(블러, 센서 노이즈, 압축 등)를 모사하거나 강건성(robustness)을 고려해 학습된 모델 (예: Real-ESRGAN, HAT)
- **성능 역전**: 열화 타입에 따라 완전히 뒤바뀜

**학습 가정의 차이 **
- Classical SR
    - 가정: 저해상도 이미지는 고해상도를 단순히 bicubic으로 다운샘플한 결과.
    - 결과: 입력 노이즈/블러가 단순하고 규칙적 → 고주파 복원에 최적화.
- Real-SR
    - 가정: 입력은 센서 PSF, 대기 산란, 노이즈, 압축 등 복합적·비선형적 열화를 겪음.
    - 결과: 불확실성·잡음·구조적 손상에 강건하도록 학습됨.
###  실험 결과 시각화

![Result Chart](result.png)

**차트 해석:**
- **파란색 막대**: UC Merced Land Use DataSet + Bicubic Degradation
- **주황색 막대**: Synthetic (합성 데이터 + Bicubic Degradation)
- **초록색 빗금 막대**: UC Merced Land Use DataSet + Real-world complex Degradation 

**관찰:**
1. Classicar SR 모델은 Bicubic Degradation이 사용된 데이터셋에서 강세를 보이고, Resl SR 모델은 Real-world complex Degradation 이 사용된 데이터셋에서 성능 우위 확인

---

##  연구 배경

### 문제 정의

기존 SR 연구는 주로 **Bicubic downsampling** 환경에서 평가되지만, 실제 위성영상은 다음과 같은 **복합 열화**를 겪습니다:

-  대기 산란 (Atmospheric Scattering)
-  센서 PSF 블러 (Point Spread Function)
-  센서 노이즈 (Sensor Noise)
-  JPEG 압축 (Compression Artifacts)

**의문점**: 
- super-resolution을 위한 모델 선정시 단순히 벤치마크성능이 우수한 모델을 고르면 되는걸까?
- Bicubic 환경에서 우수한 모델이 실제 센서 열화 환경에서도 우수할까?

**결론**:
- 벤치마크 성능위주로 모델 선정하면, 열화 타입 불일치 시 *성능 급락**을 겪는다
- super-resolution 모델 선정은 데이터의 Degradation type과 이어질 다운스트림 태스크를 고려해야한다 

---

##  실험 설계

### 테스트 모델 (4개)

#### Classical SR (Bicubic 학습)
1. **EDSR** - CNN 기반, DIV2K (bicubic)
2. **SwinIR-M** - Transformer 기반, DIV2K (bicubic)

#### Real-SR (복합 열화 학습)
3. **Real-ESRGAN** - CNN+GAN, DF2K (real degradation)
4. **HAT-L** - Transformer, ImageNet (real degradation)

> **주의**: 모델 분류는 **아키텍처가 아닌 학습 데이터의 열화 타입**을 기준으로 합니다.

### 테스트 데이터셋 (3개)

#### 1️⃣ UC Merced Land Use + Bicubic Degradation
- 원본: UC Merced Land Use Dataset
- 열화: Bicubic downsampling (256×256 → 64×64 → 256×256)
- 목적: 이상적인 환경에서의 성능 측정

#### 2️⃣ Synthetic (합성 데이터 + Bicubic Degradation)
- 원본: 랜덤 생성된 합성 위성영상
- 열화: Bicubic downsampling
- 목적: 단순 패턴에서의 성능 측정

#### 3️⃣ UC Merced Land Use + Real-world complex Degradation
- 원본: UC Merced Land Use Dataset
- 열화: **Sentinel-2 센서 시뮬레이션**
  - 대기 산란 (Rayleigh scattering)
  - PSF 블러 (Gaussian kernel)
  - 센서 노이즈 (Gaussian + Poisson)
- 목적: 실제 위성 센서 환경 재현

> **⚠️ 중요**: Real-world complex Degradation은 물리적 센서 특성을 코드로 시뮬레이션 한 것을 말합니다.

---


### 모델 선택 가이드

| 데이터 유형 | 열화 패턴 | 추천 모델 | 이유 |
|-----------|----------|----------|------|
| **벤치마크/연구** | Bicubic | Classical SR (EDSR, SwinIR-M) | 학습 데이터 일치, 최고 PSNR |
| **실제 위성 Raw** | 센서 노이즈 | Real-SR (Real-ESRGAN, HAT) | 복합 열화 처리 |
| **CCTV/카메라** | 압축, 노이즈 | Real-SR | 실제 환경 최적화 |
| **열화 타입 불명** | 알 수 없음 | Real-SR | 안전한 선택 |

### 벤치마크의 함정

⚠️ **주의사항**:
- DIV2K(bicubic) 1등 모델 ≠ 실제 환경 1등
- 벤치마크 순위는 제한된 환경에서만 유효
- 실제 배포 시 도메인 특성 고려 필수

---

### 평가 지표
- **PSNR** (Peak Signal-to-Noise Ratio): 화질 측정
- **추론 시간**: CPU 기준 처리 속도

---

## 📦 설치 및 실행

### 1. 환경 설정

```bash
# 저장소 클론
git clone https://github.com/WhiteTree93/super_resolution_model_comp.git
cd super_resolution_model_comp

# Conda 환경 생성
conda create -n sr_models python=3.10
conda activate sr_models

# 필수 패키지 설치
pip install torch torchvision
pip install super-image realesrgan basicsr timm einops
pip install opencv-python-headless scipy matplotlib pandas
pip install jupyter notebook
```

### 2. 체크포인트 준비

저장소에 포함된 체크포인트 (총 285MB):
- ✅ `edsr-base_x4.pt` (5.8MB) - EDSR Classical SR
- ✅ `RealESRGAN_x4plus.pth` (64MB) - Real-ESRGAN Real-SR
- ✅ `swinir_classical_x4.pth` (57MB) - SwinIR-M Classical SR
- ✅ `HAT-L_SRx4_ImageNet-pretrain.pth` (158MB) - HAT-L Real-SR


### 3. 데이터셋 다운로드

UC Merced Land Use Dataset (필수):
```bash
# 수동 다운로드 필요
# http://weegee.vision.ucmerced.edu/datasets/landuse.html
# UCMerced_LandUse.zip 압축 해제 후 프로젝트 루트에 배치
```

### 4. 노트북 실행

```bash
jupyter notebook SR_Model_Comparison.ipynb
```



## 🔬 재현 가능성

### 실험 환경
- **OS**: macOS
- **CPU**: Apple Silicon / Intel (CUDA 불필요)
- **Python**: 3.10
- **PyTorch**: 2.0.1
- **메모리**: 8GB 이상 권장

### 재현 단계
1. 환경 설정 (위 설치 가이드 참조)
2. 체크포인트 확인 (저장소 포함)
3. UC Merced 데이터셋 다운로드
4. 노트북 순차 실행


## 📚 참고 문헌

### 모델 논문
1. **EDSR**: Lim et al., ["Enhanced Deep Residual Networks for Single Image Super-Resolution"](https://arxiv.org/abs/1707.02921), CVPRW 2017
2. **Real-ESRGAN**: Wang et al., ["Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data"](https://arxiv.org/abs/2107.10833), ICCVW 2021
3. **SwinIR**: Liang et al., ["SwinIR: Image Restoration Using Swin Transformer"](https://arxiv.org/abs/2108.10257), ICCVW 2021
4. **HAT**: Chen et al., ["Activating More Pixels in Image Super-Resolution Transformer"](https://arxiv.org/abs/2205.04437), CVPR 2023

### 데이터셋
- **UC Merced Land Use Dataset**: Yang & Newsam, 2010 ([Link](http://weegee.vision.ucmerced.edu/datasets/landuse.html))

## 🔗 관련 링크

- [Jupyter Notebook 보기](SR_Model_Comparison.ipynb)
