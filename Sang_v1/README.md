# Sang_v1

Smart Factory AI Challenge용 철강 표면 결함 분류 모델 — `Jeong_v1`을 베이스라인으로 한
**블러 증강 전용 ablation 라인**의 v1 entry.

## 결정 사항 (이후 Sang_v* 시리즈 공통 컨벤션)

이 라인은 **데이터 증강(특히 motion blur)만을 변화시켜** 어느 augmentation 정책이
hidden test의 conveyor 모션 블러 분포에 가장 robust한지를 검증하는 것을 목표로 한다.
따라서 다음 규칙을 강제한다.

### Lock (변경 금지)

다음 항목은 baseline (`Jeong_v1`)과 **bit-identical**하게 유지한다.

| 영역 | 값 |
| --- | --- |
| Teacher backbone | `ResNet50` / `IMAGENET1K_V2` |
| Student backbone | `MobileNetV3 Small` / `IMAGENET1K_V1` |
| 출력 head | Teacher `fc`, Student `classifier[-1]` 모두 `Linear(..., 6)` |
| Optimizer | `AdamW`, `lr=0.001`, `weight_decay=1e-4` |
| Loss | Teacher: CE / Student: `alpha*CE + (1-alpha)*KD`, `alpha=0.4`, `temperature=3.0` |
| Image size | `192` |
| Batch size | `32` |
| Seed | `42` |
| Best ckpt 기준 | 1순위 `val_macro_f1` ↑, 2순위 `val_loss` ↓ |
| Test transform | `Resize → ToTensor → ImageNet normalize` (블러 없음) |
| `RandomHorizontalFlip` | 유지 |

### 수정 surface (Sang_v* 간 유일하게 바뀌는 곳)

| 파일 | 위치 |
| --- | --- |
| `Net_v1.ipynb` | **Cell 4** — `RandomConveyorBeltMotionBlur` 클래스 정의 + `train_transform` 내 그 호출 인자 |
| `Net_v1.py`    | **Section "1. 커스텀 데이터 증강 (모션 블러)"** 블록 + `train_transform` 내 호출 인자 |

> 다음 Sang_v2, v3...에서도 위 두 위치(노트북 Cell 4 / 스크립트 동일 블록) **외에는
> 어떤 코드도 수정하지 않는다**. EPOCHS, 모델, 학습 루프, 추론 셀 모두 동결.
> 이 컨벤션이 깨지면 ablation의 인과 해석이 무너지므로, 새 조항이 필요하면
> 이 README의 Lock/수정 surface 표를 먼저 PR하고 그 다음 코드를 바꾼다.

## v1: baseline 대비 변경 내용

### Net_v1.ipynb / Net_v1.py — Cell 4 (블러 증강)

| 항목 | Jeong_v1 (baseline) | Sang_v1 |
| --- | --- | --- |
| `kernel_size` | 고정 `21` | **`[11, 27]` 랜덤** (홀수 보정) |
| 블러 각도 | 항상 수평 (0°) | **`[-15°, +15°]` 랜덤 회전** |
| 적용 확률 `p` | `0.7` | `0.7` (유지) |
| 적용 위치 | `train_transform` 두 번째 step | 동일 |
| 클래스/심볼 이름 | `RandomConveyorBeltMotionBlur` | 동일 (시그니처 인자만 확장) |

### EPOCHS

- baseline 대비 teacher / student 모두 `EPOCHS = 30`으로 통일.
- 이는 v1 ablation의 정착(stabilization) 보장을 위한 것이며, 이후 Sang_v* 도 동일 30 epoch 유지.

### 그 외

- `import Jeong_v1.utils → import Sang_v1.utils` (alias만 변경)
- 그 외 모든 셀/라인 baseline과 동일.

## 변경 의도

NEU-DET 도메인 + hidden test의 conveyor 모션 블러 환경에서:

1. **속도 다양성** — 단일 kernel_size(21)는 한 가지 모션 강도만을 학습시킴.
   실제 conveyor는 line speed가 가변이므로 `[11, 27]` 사이의 다양한 강도 노출.
2. **방향 흔들림** — 카메라/벨트의 미세한 정렬 어긋남으로 인해 모션이 정확히 수평이지
   않은 경우가 존재. `±15°` 회전으로 noise direction에 대한 robustness 확보.
3. **확률 p 유지** — train 분포가 항상 블러는 아니어야 clean-image 신호도 학습됨.
   baseline 검증된 0.7을 그대로 둠.

## 실행 흐름 (baseline과 동일)

1. 데이터 로드
2. train/val/test transform 구성 *(← 본 라인의 수정 지점)*
3. `ResNet50` teacher 준비
4. `MobileNetV3 Small` student 준비
5. teacher 학습 → checkpoint 저장
6. student를 distillation loss로 학습 → best checkpoint 저장
7. (선택) quantization
8. test inference → `submission.csv`

## 평가 지표

- `val_macro_f1` (best 기준 1순위)
- `val_loss` (best 기준 2순위)
- 대회 최종: `Macro F1` × CPU `Inference Time`

## 베이스라인 참고 결과

`Jeong_v1` (kernel_size=21 고정, 수평 only) 30 epoch 시:

```
Epoch 30/30 | train_loss=0.3448 | val_loss=0.0000 | val_acc=1.0000 | val_macro_f1=1.0000
```

→ val 셋은 이미 saturate. 본 라인은 **hidden test (블러 강한 분포)에서의 일반화**
   를 끌어올리는 것이 목표이므로, val 지표보다 hidden test 점수에서의 개선을 확인할 것.

## 저장 산출물

```
saved/checkpoints/teacher_resnet50.pth
saved/checkpoints/student_mobilenetv3_small.pth
saved/submission.csv
```

체크포인트 dict 구조 (baseline 동일):

```python
{
    "state_dict": best_state,
    "class_names": class_names,
    "image_size": IMG_SIZE,
    "loss": best_loss,
    "macro_f1_score": best_score,
}
```

## 기본 하이퍼파라미터

```python
IMG_SIZE = 192
BATCH_SIZE = 32
EPOCHS = 30
lr = 0.001
alpha = 0.4
temperature = 3.0
```
