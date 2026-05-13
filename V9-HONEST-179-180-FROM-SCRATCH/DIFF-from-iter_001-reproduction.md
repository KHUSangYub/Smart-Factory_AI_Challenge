# 변경 사항: `iter_001-vanilla-seed42/reproduction.ipynb` → `HONEST-179-180-from-scratch.ipynb`

> **TL;DR** — iter_001 은 V5-Q2-BIDIR (score 96.67, 174/180) 의 **순수 학습 재현**만 했고, 2 model (BETA-LION student + BIDIR student) 만 산출. V9 는 거기에 **(1) ConvNeXt-Tiny teacher 추가**, **(2) 추론 + ensemble + class_bias 보정**, **(3) honest override stack (R3 + R4 + AH12 + 2 vetos)** 까지 단일 노트북으로 통합. 결과 score 96.67 → **99.44 (179/180)**, 추론 시간 < 1s 유지.

---

## 0. 한눈에 보는 차이

| 항목 | `iter_001/reproduction.ipynb` | `V9/HONEST-179-180-from-scratch.ipynb` |
|---|---|---|
| **목적** | V5-Q2-BIDIR 학습만 재현 (2 student .pth 생성) | 학습 + 추론 + override + submission.csv 까지 전부 |
| **Cell 수** | 12 | 26 |
| **Phase 수** | A, B (학습만) | A → G (학습 3개 + 추론 + 룰 fit + override + submission) |
| **학습 모델** | 2 teacher + 2 student = 4 | 3 teacher + 2 student = 5 |
| **Test 추론** | ❌ 없음 | ✅ Phase E 에서 모든 test 이미지에 대해 ensemble 추론 |
| **Override rule** | ❌ 없음 | ✅ R3 + R4 + AH12 + P_VETO + BIDIR_ROL_VETO |
| **OOD cache 빌드** | ❌ 없음 | ✅ train 6-class × 240 × 5 augs = 7,200 sample multi-OOD cache |
| **추가 dependency** | torch, torchvision, cv2, PIL | + `timm`, `sklearn`, `skimage.feature` |
| **출력** | 4 × `.pth` | 5 × `.pth` + `submission.csv` + `status.json` + OOD `.npz` 캐시 2개 |
| **Local score** | (학습만 — 직접 측정 X. 외부 추론 노트북에서 96.67) | **99.44 (179/180)** |

---

## 1. 모델 (Architecture) 차이

### 1.1 iter_001 — 2 teacher + 2 student

| Role | Architecture | 학습 augmentation | Epochs |
|---|---|---|---|
| Teacher A | ResNet50 (IN1K_V2 pretrained) | H-blur k=15 p=0.7 + HFlip | 8 |
| Student A | MobileNetV3-Small (IN1K_V1) | (KD from Teacher A, 같은 aug) | 12 |
| Teacher B | ResNet50 (IN1K_V2 pretrained) | **BIDIR-blur** k=15 p=0.7 + HFlip | 8 |
| Student B | MobileNetV3-Small (IN1K_V1) | (KD from Teacher B, 같은 aug) | 8 |

### 1.2 V9 — 3 teacher + 2 student

| Role | Architecture | 학습 augmentation | Epochs | 추가 여부 |
|---|---|---|---|:-:|
| Teacher A (BETA-LION) | ResNet50 | H-blur **k=15** p=0.7 + HFlip | 8 | (동일) |
| Student A (BETA-LION) | MobileNetV3-Small | KD, Lion betas (0.95, 0.99) | 12 | (동일) |
| **Teacher CNXT (NEW)** | **ConvNeXt-Tiny** (timm `convnext_tiny.fb_in1k`) | H-blur **k=21** p=0.7 + HFlip | 8 | ⭐ 추가 |
| Teacher B (BIDIR) | ResNet50 | BIDIR-blur k=15 p=0.7 + HFlip | 8 | (동일) |
| Student B (BIDIR) | MobileNetV3-Small | KD, default AdamW betas | 8 | (동일) |

**핵심 신규 모델**: `teacher_convnext_tiny.pth` — student 가 아닌 **inference-time 3rd vote** 로 사용됨 (R3 의 3-way agreement: champ vs beta_argmax vs **cnxt_argmax**). KD source 가 아니라 **앙상블 멤버**.

> ConvNeXt-T 는 학습 시 **dummy student 생성 → 즉시 del** 로 RNG state 를 V3 원본 recipe 와 일치시킴 (bit-exact 재현 위해).

---

## 2. 하이퍼파라미터 (Hyperparameters) 차이

### 2.1 공통 (변경 없음)

| 항목 | 값 |
|---|---|
| `IMG_SIZE` | 192 |
| `BATCH` | 32 |
| `WEIGHT_DECAY` | 1e-4 |
| `TEACHER_LR / STUDENT_LR` | 1e-3 |
| `ALPHA (KD)` | 0.5 |
| `TEMPERATURE (KD)` | 3.0 |
| `MOTION_BLUR_K` (Phase A/C) | 15 |
| `MOTION_BLUR_P` | 0.7 |
| `STUDENT_BETAS_A` (Lion-like) | (0.95, 0.99) |
| `STUDENT_EPOCHS_A / EPOCHS_B` | 12 / 8 |
| `TEACHER_EPOCHS_A / EPOCHS_B` | 8 / 8 |
| `SEED` | 42 |

### 2.2 V9 에서 신규 추가된 하이퍼파라미터

| 항목 | 값 | 용도 |
|---|---|---|
| **ConvNeXt-T `CNXT_BLUR_K`** | **21** (Phase A/C 와 다름) | V3 recipe — 더 큰 blur 로 학습 |
| **ConvNeXt-T epochs** | 8 (`TEACHER_EPOCHS_A` 재사용) | 8 epoch + h-blur k=21 |
| **`wA` (ensemble weight)** | **0.7** | `champ = 0.7·P_BETA + 0.3·P_BIDIR` |
| **`class_bias`** | `[+2.0, -1.1, +1.9, -0.1, +1.2, -0.7]` | logit shift, train multi-OOD 에서 fit |
| **`N_AUGS`** | 5 | OOD cache 빌드 — 이미지당 5번 aug |
| **`MOTION_K` (OOD)** | 15 | OOD aug kernel (test 분포 모방) |
| **`MOTION_ANGLE`** | (-30°, +30°) | OOD aug 회전 각도 |
| **`LBP_P`, `LBP_R`, `LBP_BINS`** | 8, 1, 10 | AH12 13D feature 의 LBP 파라미터 |
| **`AUG_SEED`** | `SEED` (42) | OOD aug RNG seed (deterministic) |

> `wA`, `class_bias` 는 **V9-Q2-BIDIR-MULTI-OOD-2026-05-15** 분석에서 train multi-OOD blur (k random ∈ {25-31, 29-35, 33-39}, 5 augs × 1440 = 7,200 샘플) 로 grid sweep + coord descent 해서 derive 한 canonical 값. **val 통계 사용 안 함 (honest)**.

### 2.3 Override rule 의 임계값 (Phase F 에서 fit)

| 임계값 | Fit 방법 | 데이터 source |
|---|---|---|
| `PREC_THRESHOLD` (R3) | 0.95 (상수) | — |
| `MIN_GAIN` (R3) | 3 (상수) | — |
| `cnxt_floor` (R3, per-target) | grid `[0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99]` 중 prec≥0.95, gain≥3 최대인 값 | **train OOD cache** |
| `P_VETO` (pit→inc 차단) | R3 inc-target fire 에서 max p_pit + 0.01 | **train OOD** |
| `PIT_CNXT_FLOOR / BETA_FLOOR / PFLOOR` (R4) | 3D grid sweep, prec≥0.95, gain≥3 | **train OOD cache** |
| `BIDIR_ROL_VETO` (R4 차단) | TP (beta_pit≥0.99, GT=pit) 의 max BIDIR_rol + 0.02 | **train OOD cache** |
| `A_r1_fft`, `A_r1_coh` (AH12) | inc class 의 5%ile / 95%ile | **train inclusion 1,200 sample** |
| `A_r2_thr`, `A_r3_thr`, `A_r4_thr` (AH12) | 99%ile (각각 inc-vs-pit LR / Mahalanobis / LR prob) | **train inc + pit feats** |

> 모든 임계값이 **train 데이터에서만** derive 됨 — `validation/` 폴더 사용 0회 (TA 룰 준수, [[project_ta_ruling_override_rules_20260513]]).

---

## 3. 데이터 증강 (Augmentation) 차이

### 3.1 학습 시 (Train-time) augmentation

| 모델 | iter_001 | V9 |
|---|---|---|
| Teacher A / Student A | `HorizontalMotionBlur(k=15, p=0.7)` + HFlip | (동일) |
| Teacher B / Student B | `BidirectionalMotionBlur(k=15, p=0.7)` + HFlip | (동일) |
| **Teacher CNXT** | (해당 없음) | **`HorizontalMotionBlur(k=21, p=0.7)`** + HFlip ⭐ |

### 3.2 추론 시 (Inference) — V9 에서 신규 도입

iter_001 은 추론 단계가 없음. V9 는 다음 두 종류의 추론을 함:

1. **Test inference** (180장): augmentation 없음, val_transform (Resize 192 + Normalize) 만.
2. **Train OOD cache 빌드** ⭐ — honest threshold fit 용:
   ```python
   motion_blur(img_bgr, k=15, rng=rng_aug)
     # angle ∈ rng.uniform(-30°, +30°)
     # 1D kernel (mid-row=1.0) → rotation by angle → normalize → filter2D
   ```
   - Train 의 모든 class × 모든 image × 5 augs = **6 × 240 × 5 = 7,200 sample**
   - cv2 기반 (PIL augmentation 과 다른 pipeline) — test 분포 (5 m/s 컨베이어 motion blur) 를 시뮬레이션
   - 결과는 `ood_cache_all_classes.npz` (3 model 의 logit) + `ood_feats_all_classes.npz` (13D feature) 캐시
   - **결정론적**: `rng = np.random.default_rng(SEED=42)` — 같은 환경 재실행 시 bit-exact

### 3.3 13D Hand-crafted feature (V9 신규)

`compute_feat13(pil)` — AH12 / P_VETO / R4 의 p_pit 추출에 사용:

| 인덱스 | 이름 | 의미 |
|---|---|---|
| 0 | `fft_power_ratio` | FFT 의 고주파 (r_n ∈ [0.4, 1.0]) 에너지 비율 — defect 의 sharpness |
| 1 | `coherence` | structure tensor 의 anisotropy ratio — directional blur 의 방향성 |
| 2 | `cc_count` | Otsu binarize 후 connected component 개수 — defect 의 개체 수 |
| 3-12 | `lbp_hist` (10 bins) | uniform LBP histogram (P=8, R=1) — local texture pattern |

→ inclusion vs pitted_surface 의 **수동 구별** 에 매우 효과적 (특히 motion-blur 환경에서 model 이 헷갈리는 케이스).

---

## 4. Override 방식 차이

### 4.1 iter_001 — override 없음

학습만 하고 `.pth` 저장 후 끝. 별도 추론 노트북 (`jeong-v5-q2-bidir-multi-ood-2026-05-15`) 에서 추론 + ensemble 만 적용 → **score 96.67 (174/180)**.

### 4.2 V9 — 5-layer honest override stack

#### Layer 0: **Champion ensemble** (V9 baseline)
```python
champ_prob   = wA * softmax(logit_BETA) + (1-wA) * softmax(logit_BIDIR)   # wA=0.7
champ_logit  = log(champ_prob) + class_bias                                # bias = [+2,-1.1,+1.9,-0.1,+1.2,-0.7]
champ_final  = softmax(champ_logit)
champ_pred   = argmax(champ_final)
```

#### Layer 1: **R3_pertarget** — 3-way agreement, per-target allowlist
- 조건: `champ_pred ≠ beta_argmax` AND `beta_argmax == cnxt_argmax` AND `cnxt_conf ≥ cnxt_floor_T`
- 각 target class T 마다 train OOD 에서 precision ≥ 0.95 + gain ≥ 3 인 최적 `cnxt_floor` 찾음
- target T 가 그 조건을 만족 못 하면 → **그 class 로의 override 불가** (`BLOCKED`)
- Override: `champ_pred[i] → beta_argmax[i]`

#### Layer 2: **P_VETO** (pit → inc flip 차단)
- R3 가 `champ=pit → beta=inc` 로 flip 하려고 할 때
- 해당 sample 의 `p_pit` (inc-vs-pit feature classifier) 가 `P_VETO` 이상이면 → flip 거절
- `P_VETO` = train OOD 의 R3 inc-target fire 에서 관측된 max p_pit + 0.01
- **honest anchor**: train 데이터에서 "inc 라고 flip 한 게 정말 inc 였다" 케이스의 boundary 활용

#### Layer 3: **R4_pit** — pit 전용 3D grid (target=pit)
- 조건: `champ ≠ beta == cnxt == pit` AND `cnxt_pit ≥ PIT_CNXT_FLOOR` AND `beta_pit ≥ PIT_BETA_FLOOR` AND `p_pit ≥ PIT_PFLOOR`
- 3D grid 에서 prec ≥ 0.95 + gain ≥ 3 만족하는 (cf, bf, pf) 조합 중 gain 최대인 것 선택
- Override: `champ_pred[i] → PIT`

#### Layer 4: **BIDIR_ROL_VETO** (R4 의 pit 차단)
- R4 가 fire 하려고 할 때 BIDIR student 가 `rolled-in_scale` 을 강하게 예측하면 → flip 거절
- `BIDIR_ROL_VETO` = TP (beta_pit ≥ 0.99 AND GT=pit) 에서 관측된 max BIDIR_rol + 0.02

#### Layer 5: **A_AH12** — 13D feature 4-rule intersection (inc → pit)
- 이미 R3/R4 로 flip 된 sample 은 skip
- `final_pred == inc` 인 sample 에 대해 4개 sub-rule 모두 통과하면 → `INC → PIT` flip
  - `A_r1`: FFT power 가 inc 의 하위 5% 면서 coherence 가 상위 5%
  - `A_r2`: pit-vs-inc Gaussian log-likelihood ratio 가 inc 분포의 상위 1%
  - `A_r3`: Mahalanobis distance 가 pit 분포에 더 가깝고, pit 분포의 상위 1% 안
  - `A_r4`: isotonic-calibrated LR 의 p(pit | features) 가 inc 분포의 상위 1%
- Train 1,200 inc sample 에서 FP ≤ 5 면 valid

### 4.3 Override stack flow

```
test image 180장 ──┐
                  ▼
        ensemble champ_pred  (with class_bias)
                  │
                  ▼
        ┌─ R3 fire? ──── YES ── pit→inc? ── YES & p_pit≥P_VETO ── VETO (그대로)
        │                  │                NO                  ── flip champ→beta
        │                  NO ──────────────────────────────────── flip champ→beta
        │                  │
        │  (not flipped)
        │                  ▼
        ├─ R4 fire (target=pit)? ── YES & BIDIR_rol≥VETO ── VETO (그대로)
        │                            YES & BIDIR_rol<VETO ── flip → PIT
        │                            NO ─────────────────── (그대로)
        │
        │  (not flipped, final_pred==inc)
        │                  ▼
        └─ AH12 all 4 sub-rules pass? ── YES ── flip INC→PIT
                                          NO  ── (그대로)
                                                  │
                                                  ▼
                                        final_pred → submission.csv
```

---

## 5. 출력물 (Outputs) 차이

| 항목 | iter_001 | V9 |
|---|---|---|
| `teacher_resnet50.pth` | ✅ | ✅ |
| `student_BETA-LION.pth` | ✅ | ✅ |
| `teacher_resnet50_bidir.pth` | ✅ | ✅ |
| `student_BIDIR.pth` | ✅ | ✅ |
| **`teacher_convnext_tiny.pth`** | ❌ | ⭐ ✅ |
| **`ood_cache_all_classes.npz`** (3-model softmax × 7,200 samples) | ❌ | ⭐ ✅ |
| **`ood_feats_all_classes.npz`** (13D feats × 7,200 samples) | ❌ | ⭐ ✅ |
| **`train_ood_feats.npz`** (inc/pit 만의 13D feats for AH12) | ❌ | ⭐ ✅ |
| **`submission.csv`** (180-row Kaggle 제출 형식) | ❌ | ⭐ ✅ |
| **`status.json`** (모든 rule fit 결과 + md5 + 환경 정보) | ❌ | ⭐ ✅ |
| `verification.md` | ✅ | ❌ (status.json 으로 대체) |

---

## 6. 외부 의존성 (Dependencies) 차이

V9 가 추가로 필요로 하는 라이브러리:

```python
# iter_001 에는 없던 import
import torch.nn.functional as F
import timm                                              # ConvNeXt-T
from sklearn.linear_model import LogisticRegression     # P_VETO / AH12
from sklearn.preprocessing import StandardScaler        # 13D feat 스케일
from sklearn.calibration import CalibratedClassifierCV  # isotonic-calibrated p_pit
from sklearn.metrics import f1_score, classification_report
from skimage.feature import local_binary_pattern        # LBP 10-bin histogram
```

`requirements.txt` 도 그만큼 더 큼 (V9: 추가 lib pin).

---

## 7. 재현성 (Reproducibility) 측면

| 항목 | iter_001 | V9 |
|---|---|---|
| `seed_everything(42)` 호출 | ✅ Phase A, B 시작 시 각 1번 | ✅ Phase A, B, C 시작 시 각 1번 |
| SEED transparency print | ✅ | ✅ (동일 block) |
| OOD augmentation seed | (해당 없음) | ✅ `np.random.default_rng(AUG_SEED=42)` — 별도 RNG instance 로 global state 오염 차단 ([[project_augmentation_seed_isolation_2026_05_13]]) |
| Sklearn `random_state` | (해당 없음) | ✅ `random_state=SEED` on `LogisticRegression` |
| MD5 print | ✅ checkpoint 4개 | ✅ 5개 + `submission.csv` |
| Cross-env 검증 신호 | `.pth` MD5 | `submission.csv` MD5 (최종 deliverable) |

---

## 8. ⭐ V7-AH12-INTERSECTION-ENSEMBLE 대비 차이 (구버전 override → 신버전 override)

> 사용자가 직접 비교 요청한 부분. iter_001 → V9 만 보면 "처음부터 override 가 생겼다" 인데, 실제로는 그 사이에 **V7-AH12 (4-layer override)** 가 있었고 V9 는 그걸 **TA 룰 위반 제거** + **honest 재구성** 한 버전. 두 노트북 모두 같은 13D feature 와 같은 AH12 룰을 쓰지만 **윗 layer 가 완전히 다름**.

### 8.1 한눈에 보는 architecture 차이

| 항목 | V7-AH12-INTERSECTION-ENSEMBLE | V9 HONEST-179-180-from-scratch |
|---|---|---|
| **학습 모델 수** | 3 students + 3 teachers + 1 multi-teacher student | 2 students + 3 teachers |
| **앙상블 멤버** | BETA + BIDIR + **C5 (multi-teacher student)** | BETA + BIDIR + **ConvNeXt-T (teacher 직접 사용)** |
| **Phase 수** | A~L (12 phase) | A~G (7 phase) |
| **Override layer 수** | **4 layer** | **5 layer (3 rule + 2 veto)** |
| **임계값 fit source** | ⚠️ **Val OOD 통계 사용 (Phase I)** | ✅ **Train OOD only** |
| **TA 룰 준수** | ❌ AC10 phase 가 룰 위반 ([[project_v7_ah12_rule_violation]]) | ✅ 전부 honest |
| **Kaggle 예상 점수** | ~100 (룰 위반으로 invalid) | **99.44 (honest)** |
| **Local 점수** | 99.44 (180/180 expected) | **99.44 (179/180)** |

### 8.2 Override layer 1:1 매핑

```
        V7-AH12 (4-layer)                        V9 HONEST (5-layer)
        ────────────────────────                 ─────────────────────────
        ┌──────────────────────┐                 ┌─────────────────────────┐
Layer 1 │ Phase H: CR-C5       │   ───────→      │ R3_pertarget            │ Layer 1
        │  champ=pit & c5=inc  │                 │  champ ≠ beta == cnxt   │
        │  c5_conf≥0.30        │                 │  per-target cnxt_floor  │
        │  champ_conf≤0.75     │                 │  (auto-fit allowlist)   │
        │  → flip pit→inc      │                 │  → flip champ→beta      │
        └──────────────────────┘                 ├─────────────────────────┤
                                                 │ P_VETO (NEW)            │ Layer 2
                                                 │  pit→inc flip 차단      │
                                                 │  p_pit < P_VETO 요구    │
                                                 └─────────────────────────┘
        ┌──────────────────────┐
Layer 2 │ Phase I.AC8          │   ╳ 삭제 ╳     (V9 에서 제거됨 — R3 의
        │  crazing→pitted      │                 per-target allowlist 가
        │  fft<low & coh<low   │                 자동으로 catch 함)
        └──────────────────────┘
        ┌──────────────────────┐                 ┌─────────────────────────┐
Layer 3 │ Phase I.AC10         │   재구성 →      │ R4_pit                  │ Layer 3
        │  rolled-in→pitted    │                 │  champ ≠ beta=cnxt=pit  │
        │  fft<low & coh<low   │                 │  3D grid (cf, bf, pf)   │
        │  & cc<low            │                 │  on TRAIN OOD           │
        │  ⚠️ VAL 통계 fit     │                 │  ✅ TRAIN 통계 fit       │
        └──────────────────────┘                 ├─────────────────────────┤
                                                 │ BIDIR_ROL_VETO (NEW)    │ Layer 4
                                                 │  R4 fire 시             │
                                                 │  BIDIR_rol<veto 요구    │
                                                 └─────────────────────────┘
        ┌──────────────────────┐                 ┌─────────────────────────┐
Layer 4 │ Phase J.AH12         │   ───────→      │ A_AH12 (그대로)         │ Layer 5
        │  inc→pit             │                 │  inc→pit                │
        │  R1∩R2∩R3∩R4         │                 │  R1∩R2∩R3∩R4 동일       │
        │  13D feature         │                 │  (TRAIN inc/pit feats)  │
        └──────────────────────┘                 └─────────────────────────┘
```

### 8.3 각 layer 별 상세 비교

#### Layer 1: V7 CR-C5 → V9 R3_pertarget

| 측면 | V7 (CR-C5) | V9 (R3_pertarget) |
|---|---|---|
| **3rd voter** | C5 (multi-teacher dual-KD student) | ConvNeXt-T teacher 직접 사용 |
| **Trigger 방향** | **고정 1방향**: `champ=pit ∧ c5=inc → flip pit→inc` | **양방향 (6 target 모두 시도)**: `champ ≠ beta == cnxt` 인 모든 target T 에 대해 자동 allowlist |
| **임계값** | `c5_conf ≥ 0.30`, `champ_conf ≤ 0.75` (하드코딩) | `cnxt_floor_T` (per-target, train OOD 에서 prec≥0.95 & gain≥3 만족하는 최소값 자동 선택) |
| **Fit data** | 없음 (상수) | **Train OOD cache** (6 class × 240 × 5 augs) |
| **유연성** | pit→inc 한 케이스만 catch | 6 class 어디로든 자동 검출 |

> V9 가 **flexible** 한 이유: V7 은 "champ=pit & c5=inc 일 때만" 고정인데, V9 는 같은 3-way agreement 패턴을 **모든 (from, to) 쌍** 에 적용. 임계값은 train 데이터로 자동 fit.

#### Layer 2 (V9 NEW): P_VETO — pit→inc 차단

| 동기 | V7 에서는 없음 — R3 의 pit→inc flip 이 **순수 model softmax 기반** 이라 가끔 잘못 flip |
|---|---|
| **로직** | R3 가 `champ=pit → beta=inc` 로 flip 하려고 할 때, feature classifier 의 `p_pit ≥ P_VETO` 면 **flip 거절** |
| **Anchor** | train OOD 에서 R3 가 fire 한 sample 의 max `p_pit` + 0.01 |
| **효과** | pit class 가 feature 상 정말 pit 답게 보이면 model 이 inc 라고 우겨도 flip 안 함 |

#### V7 Layer 2 (AC8) → V9 에서 삭제

| V7 AC8 | "crazing → pitted" rule. `fft<low & coh<low` (val 통계 기반) |
|---|---|
| V9 처리 | **삭제됨**. V9 의 R3_pertarget 이 cnxt_floor 만 충족하면 crazing→pitted 도 자동으로 catch (target 무관). 즉, V7 의 AC8 은 V9 R3 에 **흡수됨** |

#### Layer 3: V7 AC10 → V9 R4_pit (TA 룰 위반 해소)

| 측면 | V7 (AC10) | V9 (R4_pit) |
|---|---|---|
| **목적** | rolled-in→pitted flip | pit-target 으로의 안전한 flip (모든 source 에서) |
| **임계값** | `fft<μ−σ, coh<μ−σ, cc<μ−σ` of rolled-in class | `cnxt_pit ≥ cf` ∧ `beta_pit ≥ bf` ∧ `p_pit ≥ pf` |
| **Fit data** | ⚠️ **val OOD (180 val × 5 augs = 900)** — **TA 룰 위반** | ✅ **train OOD (7,200 samples)** — honest |
| **임계값 형태** | hardcoded statistical (μ±σ) | 3D grid sweep → prec≥0.95 & gain≥3 |

> 이게 사용자가 메모리에서 표시한 [[project_v7_ah12_rule_violation]] 의 핵심: **V7 AC10 phase 가 val OOD 통계로 fit 해서 룰 위반**. V9 는 같은 목적 (pit 으로의 flip) 을 **train OOD 만으로** 달성 — TA 룰 ([[project_ta_ruling_override_rules_20260513]]) 준수.

#### Layer 4 (V9 NEW): BIDIR_ROL_VETO

| 동기 | rolled-in_scale 이 motion blur 환경에서 pitted_surface 와 헷갈리는 케이스 |
|---|---|
| **로직** | R4 가 `→ pit` flip 하려고 할 때, BIDIR student 가 `rolled-in` 을 강하게 예측 (`BIDIR_rol ≥ veto`) 하면 flip 거절 |
| **Anchor** | train OOD 의 TP (beta_pit≥0.99 ∧ GT=pit) 에서 max `BIDIR_rol` + 0.02 |
| **효과** | BIDIR 이 "이건 rolled-in 이다" 라고 강력하게 말하면 R4 의 pit flip 을 차단 — pit ↔ rolled-in 양방향 confusion 방지 |

#### Layer 5: AH12 (V7 = V9 동일!)

| 측면 | V7 Phase J.AH12 | V9 Layer 5 A_AH12 |
|---|---|---|
| 4 sub-rule | R1 (percentile) ∩ R2 (Gaussian LR) ∩ R3 (Maha) ∩ R4 (LogReg) | **완전 동일** |
| 13D feature | FFT power + coherence + CC + LBP(10) | **완전 동일** |
| Fit data | train inc(1,200) + pit(1,200) feats | **완전 동일** |
| 임계값 | 99%ile (R2, R3, R4), 5%ile/95%ile (R1) | **완전 동일** |

> 이 layer 는 V7 → V9 사이에 **건드리지 않은 부분**. 같은 honest 룰이고 같은 train-only fit. ([[project_v7_ah12_breakthrough]] 의 핵심 발견 — V9 가 그대로 계승)

### 8.4 핵심 변경 요약

| 변경 유형 | 항목 |
|---|---|
| **🔴 삭제** | V7 Phase D (C5 multi-teacher student) → V9 는 ConvNeXt-T teacher 직접 사용 |
| **🔴 삭제** | V7 Phase I.AC8 (crazing→pitted) → V9 R3_pertarget 이 자동 catch |
| **🔄 재구성 (룰 위반 해소)** | V7 AC10 (val 통계 fit) → V9 R4_pit (train 통계 fit, 3D grid) |
| **🔄 일반화** | V7 CR-C5 (1방향 hardcoded) → V9 R3_pertarget (6방향 auto-allowlist) |
| **🟢 신규** | V9 P_VETO (pit→inc flip feature 차단) |
| **🟢 신규** | V9 BIDIR_ROL_VETO (pit←rolled flip BIDIR 차단) |
| **✅ 유지** | AH12 4-rule intersection (V7 Phase J = V9 Layer 5) |

### 8.5 왜 layer 가 4 → 5 로 늘었는데 단순해졌나?

V7 은 **하드코딩 임계값** 의 누적 (CR-C5 → AC8 → AC10 → AH12) 으로 4 phase. 각 phase 가 다른 (from, to) 쌍을 따로 처리.
V9 는 **메커니즘** 으로 묶음:
- R3_pertarget → "3-way model agreement" 메커니즘 1개로 (V7 의 CR-C5 + AC8 흡수)
- R4_pit → "feature + softmax 다축 grid" 메커니즘 1개로 (V7 의 AC10 재구성)
- AH12 → 그대로
- 그 위에 **veto 2개** 추가 (P_VETO + BIDIR_ROL_VETO) → false flip 차단

→ **layer 수는 늘었지만 (4 → 5), 각 layer 가 일반화/honest 화 되어 cognitive complexity 는 낮아짐**. 그리고 **TA 룰 준수**.

### 8.6 결과 비교

| 노트북 | Local 점수 | Kaggle 검증 | 룰 상태 |
|---|---|---|---|
| V7-AH12 | 180/180 expected | ~100 (룰 위반으로 invalid) | ⚠️ AC10 phase val 통계 fit |
| V9 HONEST | **179/180** (test_144 local-GT noise 불가) | **99.44** (honest, valid) | ✅ |

> V7 의 "100점" 은 **invalid 점수** (TA 룰 위반). V9 의 99.44 가 **공식 honest baseline** — 1점 손해를 honest 로 받아들이고 가는 게 V9 의 정체성.

### 8.7 V7 → V9 변화의 narrative

```
V7-AH12 (2026-05-12)         V9 HONEST (2026-05-13)
━━━━━━━━━━━━━━━━━━━━━━       ━━━━━━━━━━━━━━━━━━━━━━━━━━
1. 4-layer override         1. TA 룰 발표 (2026-05-13)
   = 학습 데이터              "val 통계 추출 금지"
   + val OOD 통계
   + train OOD 통계         2. V7 의 AC10 layer
                              = val 통계 fit ⇒ 룰 위반 판정
2. Kaggle ~100 expected
                            3. AC10 제거 / 재구성 →
                              R4_pit (train-only)

                            4. CR-C5 → R3_pertarget 일반화
                              + P_VETO 추가
                              + BIDIR_ROL_VETO 추가

                            5. 결과: 179/180 honest, valid
```

---

## 9. 한 줄 요약

> **iter_001** 은 **학습 파이프라인 bit-exact 재현 노트북** (입력: pretrained → 출력: 4 student/teacher .pth).
> **V9** 는 거기에 (1) **ConvNeXt-T 3rd voter 추가**, (2) **train-OOD 기반 honest 임계값 fit** (val 통계 0회 사용), (3) **R3 + R4 + AH12 + 2 veto = 5-layer override stack** 을 single notebook 으로 합쳐 **submission.csv 까지** 한 번에 만드는 self-contained 노트북. 96.67 → **99.44 (179/180)** 의 차이는 거의 전부 **override stack 의 5개 추가 catch + 1개 차단** 에서 나옴 (모델 학습 자체는 거의 동일).

---

## 10. Cross-ref

- [[finding-v9-honest-plateau-2026-05-13]] — V9 honest plateau 179/180 확정
- [[project_ta_ruling_override_rules_20260513]] — val 통계 사용 금지 TA 룰
- [[project_v9_honest_plateau_2026_05_13]] — R4 beta_pit + BIDIR_rol veto stack
- [[project_v7_ah12_breakthrough]] — V7 AH12 4-rule intersection 발견 (V9 가 Layer 5 로 계승)
- [[project_v7_ah12_rule_violation]] — V7 AC10 phase 의 val 통계 fit 룰 위반 사유
- 같은 폴더: [[MODEL_DESCRIPTION.md]] — V9 full methodology (이 문서는 그 중 "iter_001 / V7 대비 변경점" 만 추출)

## 11. Sources

- `output/comparisons/seed-hunt-bit-exact-2026-05-13/iter_001-vanilla-seed42/reproduction.ipynb`
- `output/decisions/V9-HONEST-179-180-FROM-SCRATCH/HONEST-179-180-from-scratch.ipynb`
- `output/decisions/V9-HONEST-179-180-FROM-SCRATCH/MODEL_DESCRIPTION.md`
- `output/decisions/V9-HONEST-179-180-FROM-SCRATCH/status.json`
- `output/decisions/V7-AH12-INTERSECTION-ENSEMBLE/V7-AH12-INTERSECTION-ENSEMBLE.ipynb` (구버전 4-layer override)
