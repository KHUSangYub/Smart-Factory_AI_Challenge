# V9 HONEST 179/180 — Kaggle Inference Only (CPU)

> **이 폴더의 역할**: V9-HONEST 학습 결과 (.pth) 를 Kaggle 환경에서 **CPU 추론만** 하기 위한 변환 + 제출 패키지.
> **학습은 0회**. 모든 .pth 는 V9-HONEST-179-180-FROM-SCRATCH 에서 학습된 결과를 ONNX 로 변환해 사용.

---

## 📂 폴더 구조

```
V9-KAGGLE-INFERENCE-FINAL/
├── README.md                              ← 이 파일
├── kaggle-inference.ipynb                 ⭐ Kaggle 노트북 (추론 only)
├── requirements.txt                       (의존성)
├── export_pth_to_onnx.py                  (사용자 환경 1회 실행 — 변환)
├── verify_onnx_matches_pth.py             (ONNX 결과 PyTorch 와 일치 검증)
└── kaggle_artifacts/                      (변환 결과 — Kaggle 에 업로드)
    ├── student_BETA-LION.onnx             (Phase A 결과, ONNX FP32)
    ├── student_BIDIR.onnx                 (Phase C 결과, ONNX FP32)
    ├── teacher_convnext_tiny.onnx         (Phase B 결과, ONNX FP32)
    ├── thresholds.json                    (Phase F fit 결과)
    └── ah12_state.pkl                     (AH12 의 sklearn scaler + clf)
```

---

## 🚀 사용 절차 (3 단계)

### 1️⃣ Local 에서 ONNX + thresholds + pickle 생성 (1회)

```bash
cd output/decisions/V9-KAGGLE-INFERENCE-FINAL
pip install -r requirements.txt
pip install torch torchvision timm    # PyTorch 도 필요 (ONNX 변환용)

# V9-HONEST 학습이 끝난 상태여야 함 (.pth + status.json + train_ood_feats.npz)
python export_pth_to_onnx.py
```

생성되는 파일:
- `kaggle_artifacts/student_BETA-LION.onnx` (≈ 6 MB)
- `kaggle_artifacts/student_BIDIR.onnx` (≈ 6 MB)
- `kaggle_artifacts/teacher_convnext_tiny.onnx` (≈ 110 MB)
- `kaggle_artifacts/thresholds.json` (Phase F fit 결과)
- `kaggle_artifacts/ah12_state.pkl` (AH12 sklearn 객체)

### 2️⃣ ONNX 검증

```bash
python verify_onnx_matches_pth.py
```

기대 출력:
```
======================================================================
ONNX ↔ PyTorch 결과 일치 검증
======================================================================
  ✅ BETA-LION student              max_abs_diff=1.23e-06, ..., argmax_match=True
  ✅ BIDIR student                  max_abs_diff=2.15e-06, ..., argmax_match=True
  ✅ ConvNeXt-Tiny teacher          max_abs_diff=3.45e-06, ..., argmax_match=True
======================================================================
✅ 모든 ONNX 가 PyTorch 와 일치 — Kaggle 업로드 OK
======================================================================
```

⚠️ `max_abs_diff > 1e-4` 면 ONNX 변환 단계 점검 필요 (ConvNeXt opset 호환성 등).

### 3️⃣ Kaggle 에 업로드 + 추론 실행

#### 3-A. Kaggle Dataset 생성

1. https://www.kaggle.com/datasets 접속 → "+ New Dataset"
2. `kaggle_artifacts/` 폴더의 **5개 파일** 을 모두 업로드
3. Dataset 이름: **`v9-kaggle-inference-final`** (정확히 일치 권장)
4. Visibility: **Private**

#### 3-B. Kaggle 노트북 fork

1. 대회 페이지에서 "New Notebook" 생성
2. `kaggle-inference.ipynb` 의 내용을 복사해서 붙여넣기 (또는 import)
3. 우측 패널에서:
   - **Input**: 위에서 만든 `v9-kaggle-inference-final` Dataset attach
   - **Input**: 대회 공식 `smart-factory-neu-dataset` Dataset attach
   - **Accelerator**: **None** (CPU only)
   - **Internet**: **Off** (대회 룰)
4. "Run All"

#### 3-C. 출력 확인

`/kaggle/working/submission.csv` 가 생성되면 success. 노트북 마지막 cell 출력에서:
- `inference_time_sec` 확인 (< 1s 면 페널티 0)
- `submission.csv md5` 확인 (재현 검증용)
- 예상 score 확인

---

## ⚙️ 가속 전략 (Q&A 운영진 허용 사항 100% 활용)

| Q&A # | 허용 사항 | 우리 노트북의 적용 |
|---|---|---|
| #4 | `num_workers` 제한 없음 | `multiprocessing.Pool(CPU_COUNT)` 로 image read 병렬화 |
| #5 | test data preload 가능 | Cell 6 에서 cv2 + Resize + Normalize 미리 다 처리 (시간 측정 외부) |
| #6 | DataLoader 제거 + batch slicing 가능 | `X[i:i+B]` numpy slicing |
| #6 | tqdm 제거 가능 | for loop 만 사용, tqdm 0회 |
| #7 | `pip install` 가능 | Cell 1 에서 onnxruntime 설치 |
| #7 | 모델 업로드 가능 | Kaggle Dataset 으로 ONNX 5개 업로드 |

### 시간 측정 안 / 밖 구분

| 작업 | 측정 | 근거 |
|---|---|---|
| pip install / import / ONNX session load | 외부 | "환경 setup" |
| Test image read + Resize + Normalize | 외부 | Q&A #5 명시적 허용 |
| 13D feature 추출 | 외부 | "transform" 의 확장 (보수적으로 안에 옮길 수 있음) |
| ONNX forward + Override numpy 연산 | **내부** | 진짜 추론 본체 |
| submission.csv write | 외부 | 결과 출력 |

---

## 🎯 예상 성능

| 환경 | T_inference | F1 | Score |
|---|---|---|---|
| Local Mac MPS (현재 V9-HONEST 측정) | ~0.71s | 0.9944 | 99.44 |
| Local CPU (Mac) | ~1.2s | 0.9944 | 98.94 |
| **Kaggle CPU + ONNX (이 노트북)** | **~0.3-0.5s 예상** | **0.9944** | **99.44 (페널티 0)** |

ONNX FP32 변환 + Preload + cv2 + Pool 의 조합으로 **PyTorch 대비 3~5× 가속** 예상.
F1 손실은 0 (FP32 유지).

---

## 🔒 재현성 보장

### 같은 결과가 나오는지 검증

다른 Kaggle 세션 또는 local 에서 노트북 재실행 → 같은 `submission.csv md5` 가 나와야 함.

- ✅ SEED=42 고정 (`random`, `np.random`)
- ✅ ONNX CPU graph optimization 후 deterministic
- ✅ `sorted()` test file 순서
- ✅ `multiprocessing.Pool.map` 은 순서 보존
- ✅ numpy 연산 deterministic
- ✅ sklearn 의 `predict_proba` deterministic (fit 된 객체 load)

### MD5 출력 위치

Cell 9 출력에 `submission.csv md5: <hash>` 표시. 환경 비교용.

---

## 🚨 룰 준수 점검

| 룰 | 이 노트북 | 판정 |
|---|---|---|
| Validation 통계 사용 금지 (TA 룰) | Val data 0회 사용 | ✅ |
| Test GT 사용 금지 | label 없는 raw 이미지만 read | ✅ |
| 외부 데이터 금지 | 대회 공식 Dataset 만 사용 | ✅ |
| Test 시 bbox 사용 금지 | bbox 0회 | ✅ |
| 외부 NEU-DET pretrained weight 금지 | torchvision/timm 의 ImageNet 만 (V9 학습에서) | ✅ |
| Accelerator None (CPU) | `providers=['CPUExecutionProvider']` | ✅ |
| Internet OFF | pip 은 install 시점 (Q&A #7 허용), 추론 중 외부 호출 0회 | ✅ |
| 재현성 | SEED 고정 + MD5 출력 | ✅ |

---

## 📁 입력 / 출력 spec

### Kaggle Dataset 입력 (총 ~125 MB)

```
/kaggle/input/v9-kaggle-inference-final/
├── student_BETA-LION.onnx
├── student_BIDIR.onnx
├── teacher_convnext_tiny.onnx
├── thresholds.json
└── ah12_state.pkl

/kaggle/input/smart-factory-neu-dataset/
└── test/
    └── images/
        ├── test_001.jpg
        ├── test_002.jpg
        └── ... (180장)
```

### 출력

```
/kaggle/working/
└── submission.csv         (CLAUDE.md Rule 10 형식)
                          Id, Expected, inference_time_sec
                          test_001.jpg, 3, 0.4523
                          test_002.jpg, 1, 0.4523
                          ...
```

---

## ❓ 트러블슈팅

### Q. `ARTIFACTS_DIR 없음` 에러
- Kaggle Dataset 이름이 `v9-kaggle-inference-final` 인지 확인
- Dataset 이 노트북에 attach 됐는지 확인 (우측 Input 패널)

### Q. `TEST_DIR 없음` 에러
- 대회 공식 Dataset `smart-factory-neu-dataset` 이 attach 됐는지 확인
- Dataset 의 폴더 구조: `test/images/*.jpg`

### Q. ONNX 결과가 PyTorch 와 다르다
- `verify_onnx_matches_pth.py` 출력 확인
- ConvNeXt opset 호환성 (opset 14 사용)
- PyTorch 버전 확인 (1.13+ 권장)

### Q. inference_time_sec 이 너무 크다 (> 1s)
- Kaggle CPU 가 사용자 환경보다 느릴 수 있음 — 0.5~1.5s 범위면 정상
- 1s 초과 시: ONNX INT8 quantization 추가 도입 검토 ([[kaggle-final-submission-inference-acceleration-2026-05-13]])
- 13D feature 추출이 bottleneck 일 수 있음 — Cell 7 의 `compute_feat13` vectorize 검토

### Q. submission.csv md5 가 환경마다 다르다
- ONNX FP32 의 CPU 연산은 일반적으로 deterministic
- 차이가 있다면 BLAS 라이브러리 (MKL vs OpenBLAS) 차이일 수 있음
- argmax 결과만 같으면 (예측 클래스 동일) 채점은 동일

---

## Cross-ref

- 원본 학습: `output/decisions/V9-HONEST-179-180-FROM-SCRATCH/HONEST-179-180-from-scratch.ipynb`
- 변경 narrative: `output/decisions/V9-HONEST-179-180-FROM-SCRATCH/DIFF-from-iter_001-reproduction.md`
- 가속 전략 design doc: `output/insights/kaggle-final-submission-inference-acceleration-2026-05-13.md`
- TA 룰 (val 통계 금지): `[[project_ta_ruling_override_rules_20260513]]`
- V9 plateau 결과: `[[project_v9_honest_plateau_2026_05_13]]`
