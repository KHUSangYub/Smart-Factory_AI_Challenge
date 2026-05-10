# %%
import copy
import os
import random
import time
from pathlib import Path
try:
    import Jeong_v2.utils as utils
except:
    import utils

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from tqdm import tqdm
import timm
try:
    import cv2
except ImportError:
    cv2 = None

# 시드 고정 (재현성 확보)
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(42)

# ==========================================
# 🌐 데이터셋 경로 설정 (🚨 반드시 본인 환경에 맞게 수정하세요!)
# ==========================================
# [1] Colab 또는 Local에서 실행할 경우: 데이터가 저장된 폴더 경로를 입력하세요.
COLAB_LOCAL_DATA_PATH = '/local_datasets/NEU-DET_open'
BASE_DIR = COLAB_LOCAL_DATA_PATH
SAVE_DIR = './saved'


# 하위 폴더 경로 설정 (압축 푼 폴더 구조에 맞춰 필요시 수정)
TRAIN_DIR = os.path.join(BASE_DIR, 'train', 'images')
VAL_DIR = os.path.join(BASE_DIR, 'validation', 'images')
TEST_DIR = os.path.join(BASE_DIR, 'test', 'images')

print(f"📂 베이스 경로: {BASE_DIR}")
print(f"📂 저장 경로: {SAVE_DIR}")

# %% [markdown]
# # Device

# %%
inference = False
DEVICE = torch.device(
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.mps.is_available()
    else 'cpu'
) if not inference else torch.device('cpu')
print(f"✅ 현재 디바이스: {DEVICE}")

# %% [markdown]
# # Data Loader

# %%
# ==========================================
# 1. 커스텀 데이터 증강 (모션 블러)
# ==========================================
IMG_SIZE = 192
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
TRAIN_BLUR_KERNEL = 15
TRAIN_BLUR_P = 0.7
VAL_BLUR_KERNEL = 15

class RandomConveyorBeltMotionBlur:
    def __init__(self, kernel_size: int = TRAIN_BLUR_KERNEL, p: float = TRAIN_BLUR_P):
        self.kernel_size = int(kernel_size)
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img

        kernel = np.zeros((self.kernel_size, self.kernel_size), dtype=np.float32)
        kernel[self.kernel_size // 2, :] = 1.0 / self.kernel_size

        if cv2 is not None:
            img_np = np.array(img)
            blurred = cv2.filter2D(img_np, -1, kernel)
            return Image.fromarray(blurred)

        return img.filter(ImageFilter.Kernel((self.kernel_size, self.kernel_size), kernel.flatten()))

class FixedConveyorBeltMotionBlur:
    def __init__(self, kernel_size: int = VAL_BLUR_KERNEL):
        self.kernel_size = int(kernel_size)

    def __call__(self, img: Image.Image) -> Image.Image:
        kernel = np.zeros((self.kernel_size, self.kernel_size), dtype=np.float32)
        kernel[self.kernel_size // 2, :] = 1.0 / self.kernel_size

        if cv2 is not None:
            img_np = np.array(img)
            blurred = cv2.filter2D(img_np, -1, kernel)
            return Image.fromarray(blurred)

        return img.filter(ImageFilter.Kernel((self.kernel_size, self.kernel_size), kernel.flatten()))

# ==========================================
# 2. 트랜스폼 및 Dataset 세팅
# ==========================================
base_eval_transform = [
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
]

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    RandomConveyorBeltMotionBlur(kernel_size=TRAIN_BLUR_KERNEL, p=TRAIN_BLUR_P),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Clean validation: 모델 선택/과적합 확인용
eval_transform = transforms.Compose(base_eval_transform)

# Blur validation: hidden/test 분포 proxy 확인용
eval_blur_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    FixedConveyorBeltMotionBlur(kernel_size=VAL_BLUR_KERNEL),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Test는 이미 blur가 들어간 데이터이므로 결정적 전처리만 적용
test_transform = transforms.Compose(base_eval_transform)

# 캐글 제출용 Test Dataset 클래스 (파일명 추출용)
class TestDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_names = sorted(
            f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))
        )

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_name

# Data Loaders
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=eval_transform)
val_blur_dataset = datasets.ImageFolder(VAL_DIR, transform=eval_blur_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
val_blur_loader = DataLoader(val_blur_dataset, batch_size=32, shuffle=False)

NUM_CLASSES = len(train_dataset.classes)
print(f"✅ 클래스 개수: {NUM_CLASSES}")
print(f"✅ Train blur kernel: {TRAIN_BLUR_KERNEL}, p={TRAIN_BLUR_P}")
print(f"✅ Validation loaders: clean={len(val_dataset)}, blur_proxy={len(val_blur_dataset)} (kernel={VAL_BLUR_KERNEL})")

# %% [markdown]
# # Model Definition

# %%
pretrained = True
num_classes = 6

# teacher
weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
teacher = models.convnext_tiny(weights=weights)
teacher.classifier[-1] = nn.Linear(teacher.classifier[-1].in_features, num_classes)

# student
weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
model = models.mobilenet_v3_small(weights=weights)
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)

# %% [markdown]
# # Train Teacher

# %%
teacher_path = os.path.join(SAVE_DIR, 'checkpoints/teacher_convnext_tiny.pth')
Path(teacher_path).parent.mkdir(parents=True, exist_ok=True)
# checkpoint = torch.load(teacher_path)
# best_score = checkpoint['macro_f1_score']
# best_loss = checkpoint['loss']
# best_state = checkpoint['state_dict']

# %%
# hyperparameters
EPOCHS = 30
lr = 0.0002

# preprocess
teacher.to(DEVICE)
class_names = train_dataset.classes
num_classes = len(class_names)
optimizer = optim.AdamW(teacher.parameters(), lr=lr, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

# 최초 1회 학습 후 삭제
best_score = -1.0
best_state = copy.deepcopy(teacher.state_dict())
best_loss = 10

# train and evaluation
for epoch in range(EPOCHS):
    # train
    teacher.train()
    train_loss = 0.0

    for images, labels in tqdm(train_loader, desc="Train", leave=False):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        logits = teacher(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
    
    train_loss /= len(train_loader.dataset)
    # eval
    teacher.eval()
    losses = []
    targets = []
    predictions = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Valid", leave=False):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = teacher(images)
            loss = criterion(logits, labels)

            preds = torch.argmax(logits, dim=1)
            losses.append(loss.item() * images.size(0))
            targets.extend(labels.cpu().tolist())
            predictions.extend(preds.cpu().tolist())
    
    val_loss = sum(losses) / len(val_loader.dataset)
    val_acc = sum(int(t == p) for t, p in zip(targets, predictions)) / len(targets)
    val_f1 = utils.macro_f1_score(targets, predictions, num_classes)


    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"train_loss={train_loss:.4f} | "
        f"val_loss={val_loss:.4f} | "
        f"val_acc={val_acc:.4f} | "
        f"val_macro_f1={val_f1:.4f}"
    )

    if (val_f1 > best_score) or (val_f1 == best_score and val_loss < best_loss):
        print("change best state")
        best_score = val_f1
        best_loss = val_loss
        best_state = copy.deepcopy(teacher.state_dict())


# checkpoint save
checkpoint = {
    "state_dict": best_state,
    "class_names": class_names,
    "image_size": IMG_SIZE,
    "loss": best_loss,
    "macro_f1_score": best_score
}
torch.save(checkpoint, teacher_path)

# %% [markdown]
# # Train Student
# teacher에 대한 학습 완료 후 진행

# %%
teacher.eval()
for parameter in teacher.parameters():
    parameter.requires_grad = False

# %%
model_path = os.path.join(SAVE_DIR, 'checkpoints/student_mobilenetv3_small.pth')
Path(model_path).parent.mkdir(parents=True, exist_ok=True)
#checkpoint = torch.load(model_path)
# best_score = checkpoint['macro_f1_score']
# best_loss = checkpoint['loss']
#best_state = checkpoint['state_dict']

# %%
# hyperparameters
EPOCHS = 30
lr = 0.0002
alpha = 0.4
temperature = 3.0
quantize = False

# preprocess
model.to(DEVICE)
class_names = train_dataset.classes
num_classes = len(class_names)
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

# 최초 1회 학습 후 삭제
best_score = -1.0
best_state = copy.deepcopy(model.state_dict())
best_loss = 10

# train and evaluation
for epoch in range(EPOCHS):
    # train
    model.train()
    train_loss = 0.0

    for images, labels in tqdm(train_loader, desc="Train", leave=False):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        logits = model(images)

        with torch.no_grad():
            teacher_logits = teacher(images)
        
        loss = utils.distillation_loss(logits, labels, teacher_logits, alpha, temperature)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
    
    train_loss /= len(train_loader.dataset)
    
    # eval
    model.eval()
    losses = []
    targets = []
    predictions = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Valid", leave=False):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(images)
            loss = criterion(logits, labels)

            preds = torch.argmax(logits, dim=1)
            losses.append(loss.item() * images.size(0))
            targets.extend(labels.cpu().tolist())
            predictions.extend(preds.cpu().tolist())
    
    val_loss = sum(losses) / len(val_loader.dataset)
    val_acc = sum(int(t == p) for t, p in zip(targets, predictions)) / len(targets)
    val_f1 = utils.macro_f1_score(targets, predictions, num_classes)


    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"train_loss={train_loss:.4f} | "
        f"val_loss={val_loss:.4f} | "
        f"val_acc={val_acc:.4f} | "
        f"val_macro_f1={val_f1:.4f}"
    )

    if (val_f1 > best_score) or (val_f1 == best_score and val_loss < best_loss):
        print("change best state")
        best_score = val_f1
        best_loss = val_loss
        best_state = copy.deepcopy(model.state_dict())


# checkpoint save
checkpoint = {
    "state_dict": best_state,
    "class_names": class_names,
    "image_size": IMG_SIZE,
    "loss": best_loss,
    "macro_f1_score": best_score
}
torch.save(checkpoint, model_path)

if quantize:
    pass

# # %% [markdown]
# # # Model Load

# # %%
# # ==========================================
# # 💡 모델 세팅
# # ==========================================
# # load checkpoint
# LOAD_CHECK_DIR = Path("./artifacts/student_mobilenetv3_small.pth")
# checkpoint = torch.load(LOAD_CHECK_DIR)
# class_names = checkpoint['class_names']
# class_names

# model = models.mobilenet_v3_small(weights=None)
# model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(class_names))
# model.load_state_dict(checkpoint["state_dict"])
# model = model.to(DEVICE).eval()


# # %% [markdown]
# # # Quantization

# # %%
# # Quantization (Linear layer)
# engine = torch.backends.quantized.supported_engines[0]
# print(engine)
# torch.backends.quantized.engine = engine

# model = copy.deepcopy(model).cpu().eval()
# model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

# # %% [markdown]
# # # Inference Code

# # %%
# # ==========================================
# # ⏱️ 추론 시간 측정 및 submission.csv 생성
# # ==========================================

# # Test 데이터 로더 준비
# test_dataset = TestDataset(TEST_DIR, transform=test_transform)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# predictions = []
# image_ids = []

# print("🚀 추론을 시작합니다...")

# # --- ⏱️ 시간 측정 시작 ---
# start_time = time.time()

# with torch.no_grad():
#     for images, img_names in tqdm(test_loader, desc="Inference"):
#         images = images.to(DEVICE)
#         outputs = model(images)
#         _, preds = torch.max(outputs, 1)

#         predictions.extend(preds.cpu().numpy())
#         image_ids.extend(img_names)

# # --- ⏱️ 시간 측정 종료 ---
# end_time = time.time()
# total_inference_time = end_time - start_time

# print(f"✅ 추론 완료! 총 소요 시간: {total_inference_time:.2f}초")

# # ==========================================
# # 📝 제출용 CSV 저장
# # ==========================================
# # inference_col = [""] * len(image_ids)
# # if inference_col:
# #     inference_col[0] = round(total_inference_time, 2)
# submission_df = pd.DataFrame({
#     'Id': image_ids,
#     'Expected': predictions,
#     'inference_time_sec': round(total_inference_time, 2),
# })

# submission_path = os.path.join(SAVE_DIR, 'submission.csv')
# Path(submission_path).parent.mkdir(parents=True, exist_ok=True)
# submission_df.to_csv(submission_path, index=False)

# print(f"🎉 제출 파일 저장 완료: {submission_path}")
# submission_df.head()

# # %%
