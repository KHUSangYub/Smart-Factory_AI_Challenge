# Jeong_v1

Smart Factory AI Challenge용 철강 표면 결함 분류 모델 버전.

## 실행 흐름

1. 데이터 로드
2. train/validation/test transform 구성
3. `ResNet50` teacher 준비
4. `MobileNetV3 Small` student 준비
5. teacher checkpoint 로드 또는 학습
6. student를 distillation loss로 학습
7. best checkpoint 저장
8. 선택적으로 quantization 적용
9. test inference 수행
10. `submission.csv` 생성

## 신경망 구성

### Teacher

- 모델: `ResNet50`
- Pretrained weight: `IMAGENET1K_V2`
- 출력층: `fc -> Linear(..., 6)`
- 역할: student 학습용 soft label 생성

### Student

- 모델: `MobileNetV3 Small`
- Pretrained weight: `IMAGENET1K_V1`
- 출력층: `classifier[-1] -> Linear(..., 6)`
- 역할: 최종 제출/추론용 경량 모델

## 사용 방법론

### 1. Transfer Learning

- ImageNet pretrained 모델 사용
- 마지막 classifier layer만 대회 클래스 수에 맞게 교체

### 2. Knowledge Distillation

- Teacher: `ResNet50`
- Student: `MobileNetV3 Small`
- Student loss:

```python
loss = alpha * ce_loss + (1.0 - alpha) * kd_loss
```

- `ce_loss`: student 예측과 실제 label 사이의 cross entropy
- `kd_loss`: teacher logits와 student logits 사이의 KL divergence
- `alpha`: CE loss 비중
- `temperature`: softmax 분포를 부드럽게 만드는 값

### 3. Motion Blur Augmentation

- 함수: `RandomConveyorBeltMotionBlur`
- 목적: hidden test의 강한 모션 블러 환경 대응
- 적용 위치: train transform
- 기본 설정:

```python
kernel_size = 21
p = 0.7
```

## 입력 전처리

### Train Transform

- `Resize((192, 192))`
- `RandomConveyorBeltMotionBlur`
- `RandomHorizontalFlip`
- `ToTensor`
- ImageNet mean/std normalization

### Validation / Test Transform

- `Resize((192, 192))`
- `ToTensor`
- ImageNet mean/std normalization

## 평가 지표

### Validation

- `val_loss`
- `val_acc`
- `val_macro_f1`

### Checkpoint 기준

- 1순위: `val_macro_f1` 최대
- 2순위: F1이 같으면 `val_loss` 최소

### 대회 최종 점수

- 성능 지표: `Macro F1`
- 속도 지표: CPU `Inference Time`

## 주요 파일

| 파일 | 내용 |
| --- | --- |
| `Net_v1.ipynb` | 전체 실험 노트북 |
| `utils.py` | distillation loss, macro F1 |
| `basemodel.py` | 참고 |

## 저장 산출물

### Checkpoint

```text
saved/checkpoints/teacher_resnet50.pth
saved/checkpoints/student_mobilenetv3_small.pth
```

### Checkpoint 내용

```python
{
    "state_dict": best_state,
    "class_names": class_names,
    "image_size": IMG_SIZE,
    "loss": best_loss,
    "macro_f1_score": best_score,
}
```

### Submission

```text
saved/submission.csv
```

컬럼:

- `Id`
- `Expected`
- `inference_time_sec`

## 기본 하이퍼파라미터

```python
IMG_SIZE = 192
BATCH_SIZE = 32
EPOCHS = 8
lr = 0.001
alpha = 0.4
temperature = 3.0
```
