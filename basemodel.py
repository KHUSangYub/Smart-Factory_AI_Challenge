import argparse
import copy
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from tqdm import tqdm

try:
    import cv2
except ImportError:
    cv2 = None


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_arg: str, for_inference: bool = False) -> torch.device:
    if device_arg == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("Requested device 'mps', but MPS is not available in this environment.")
        return torch.device("mps")

    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("Requested device 'cuda', but CUDA is not available in this environment.")
        return torch.device("cuda")

    if device_arg == "cpu":
        return torch.device("cpu")

    if not for_inference and torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


class RandomConveyorBeltMotionBlur:
    def __init__(self, kernel_size: int = 21, p: float = 0.7):
        self.kernel_size = kernel_size
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


class TestDataset(Dataset):
    def __init__(self, img_dir: str, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_names = sorted(
            f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )

    def __len__(self) -> int:
        return len(self.img_names)

    def __getitem__(self, idx: int):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_name


def build_transforms(image_size: int):
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            RandomConveyorBeltMotionBlur(kernel_size=21, p=0.7),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, eval_transform


def build_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if model_name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if model_name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model

    raise ValueError(f"Unsupported model: {model_name}")


def macro_f1_score(targets, predictions, num_classes: int) -> float:
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    for target, prediction in zip(targets, predictions):
        confusion[target, prediction] += 1

    f1_scores = []
    for class_idx in range(num_classes):
        tp = confusion[class_idx, class_idx]
        fp = confusion[:, class_idx].sum() - tp
        fn = confusion[class_idx, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0

        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))

    return float(np.mean(f1_scores))


def distillation_loss(
    student_logits: torch.Tensor,
    labels: torch.Tensor,
    teacher_logits: torch.Tensor | None,
    alpha: float,
    temperature: float,
) -> torch.Tensor:
    ce_loss = nn.functional.cross_entropy(student_logits, labels, label_smoothing=0.05)

    if teacher_logits is None:
        return ce_loss

    kd_loss = nn.functional.kl_div(
        nn.functional.log_softmax(student_logits / temperature, dim=1),
        nn.functional.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean",
    ) * (temperature ** 2)

    return alpha * ce_loss + (1.0 - alpha) * kd_loss


def create_loaders(train_dir: str, val_dir: str, image_size: int, batch_size: int, num_workers: int):
    train_transform, eval_transform = build_transforms(image_size)

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=eval_transform)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": False,
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    return train_dataset, train_loader, val_loader, eval_transform


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    teacher_model: nn.Module | None = None,
    alpha: float = 0.4,
    temperature: float = 3.0,
):
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        student_logits = model(images)

        teacher_logits = None
        if teacher_model is not None:
            with torch.no_grad():
                teacher_logits = teacher_model(images)

        loss = distillation_loss(student_logits, labels, teacher_logits, alpha, temperature)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int):
    model.eval()
    losses = []
    targets = []
    predictions = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Valid", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels)

            preds = torch.argmax(logits, dim=1)
            losses.append(loss.item() * images.size(0))
            targets.extend(labels.cpu().tolist())
            predictions.extend(preds.cpu().tolist())

    avg_loss = sum(losses) / len(loader.dataset)
    accuracy = sum(int(t == p) for t, p in zip(targets, predictions)) / len(targets)
    macro_f1 = macro_f1_score(targets, predictions, num_classes)
    return avg_loss, accuracy, macro_f1


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: torch.device,
    num_classes: int,
    teacher_model: nn.Module | None = None,
    alpha: float = 0.4,
    temperature: float = 3.0,
):
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    best_score = -1.0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            teacher_model=teacher_model,
            alpha=alpha,
            temperature=temperature,
        )
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, device, num_classes)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"val_macro_f1={val_f1:.4f}"
        )

        if val_f1 > best_score:
            best_score = val_f1
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model, best_score


def save_checkpoint(model: nn.Module, checkpoint_path: str, class_names: list[str], image_size: int):
    checkpoint = {
        "state_dict": model.state_dict(),
        "class_names": class_names,
        "image_size": image_size,
    }
    torch.save(checkpoint, checkpoint_path)


def load_model_from_checkpoint(checkpoint_path: str | Path, device: torch.device) -> tuple[nn.Module, dict]:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint["class_names"]

    if "mobilenetv3" in checkpoint_path.name:
        model = build_model("mobilenet_v3_small", num_classes=len(class_names), pretrained=False)
    else:
        model = build_model("resnet50", num_classes=len(class_names), pretrained=False)

    if "int8" in checkpoint_path.name:
        model = quantize_for_cpu(model)

    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device).eval()
    return model, checkpoint


def select_quantized_engine() -> str:
    supported_engines = list(torch.backends.quantized.supported_engines)
    preferred_order = ["qnnpack", "fbgemm"]

    for engine in preferred_order:
        if engine in supported_engines:
            torch.backends.quantized.engine = engine
            return engine

    raise RuntimeError(
        "Dynamic quantization is not available in this environment. "
        f"Supported quantized engines: {supported_engines}"
    )


def quantize_for_cpu(model: nn.Module) -> nn.Module:
    model = copy.deepcopy(model).cpu().eval()
    engine = select_quantized_engine()
    print(f"Using quantization engine: {engine}")
    return torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)


def estimated_final_score(macro_f1: float, inference_time_sec: float) -> float:
    if inference_time_sec > 300:
        return 0.0

    penalty = max(0.0, (inference_time_sec - 60.0) * 0.2)
    return max(0.0, (macro_f1 * 100.0) - penalty)


def train_pipeline(args):
    seed_everything(args.seed)
    device = resolve_device(args.device)

    train_dataset, train_loader, val_loader, _ = create_loaders(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    class_names = train_dataset.classes
    num_classes = len(class_names)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Classes: {class_names}")

    teacher = build_model("resnet50", num_classes=num_classes, pretrained=True).to(device)
    teacher, teacher_f1 = fit(
        model=teacher,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.teacher_epochs,
        learning_rate=args.teacher_lr,
        device=device,
        num_classes=num_classes,
    )
    teacher_path = os.path.join(args.output_dir, "teacher_resnet50.pth")
    save_checkpoint(teacher, teacher_path, class_names, args.image_size)
    print(f"Saved teacher: {teacher_path} | best_val_macro_f1={teacher_f1:.4f}")

    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False

    student = build_model("mobilenet_v3_small", num_classes=num_classes, pretrained=True).to(device)
    student, student_f1 = fit(
        model=student,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.student_epochs,
        learning_rate=args.student_lr,
        device=device,
        num_classes=num_classes,
        teacher_model=teacher,
        alpha=args.alpha,
        temperature=args.temperature,
    )
    student_path = os.path.join(args.output_dir, "student_mobilenetv3_small.pth")
    save_checkpoint(student, student_path, class_names, args.image_size)
    print(f"Saved student: {student_path} | best_val_macro_f1={student_f1:.4f}")

    if args.quantize:
        quantized_student = quantize_for_cpu(student)
        quantized_path = os.path.join(args.output_dir, "student_mobilenetv3_small_int8.pth")
        save_checkpoint(quantized_student, quantized_path, class_names, args.image_size)
        print(f"Saved quantized student: {quantized_path}")


def inference_pipeline(args):
    seed_everything(args.seed)
    device = resolve_device(args.device, for_inference=args.device == "auto")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Run training first, for example:\n"
            "  python edge_distill.py train --device mps\n"
            "Then run inference with the saved checkpoint path."
        )

    model, checkpoint = load_model_from_checkpoint(checkpoint_path, device)
    image_size = checkpoint["image_size"]

    if args.quantize and device.type != "cpu":
        raise ValueError("Dynamic quantization is only supported for CPU inference.")

    if args.quantize and "int8" not in checkpoint_path.name:
        model = quantize_for_cpu(model)

    _, eval_transform = build_transforms(image_size)
    test_dataset = TestDataset(args.test_dir, transform=eval_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    predictions = []
    image_ids = []

    start_time = time.time()
    with torch.no_grad():
        for images, img_names in tqdm(test_loader, desc="Inference"):
            logits = model(images.to(device))
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().tolist())
            image_ids.extend(img_names)
    total_time = time.time() - start_time

    inference_column = [""] * len(image_ids)
    if inference_column:
        inference_column[0] = round(total_time, 2)

    submission = pd.DataFrame(
        {
            "Id": image_ids,
            "Expected": predictions,
            "inference_time_sec": inference_column,
        }
    )
    submission.to_csv(args.submission_path, index=False)
    print(f"Saved submission: {args.submission_path}")
    print(f"Inference device: {device}")
    print(f"Total inference time: {total_time:.2f} sec")


def eval_pipeline(args):
    seed_everything(args.seed)
    device = resolve_device(args.device, for_inference=args.device == "auto")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model, checkpoint = load_model_from_checkpoint(checkpoint_path, device)

    if args.quantize and device.type != "cpu":
        raise ValueError("Dynamic quantization is only supported for CPU evaluation.")

    if args.quantize and "int8" not in checkpoint_path.name:
        model = quantize_for_cpu(model)
        device = torch.device("cpu")

    _, eval_transform = build_transforms(checkpoint["image_size"])
    eval_dataset = datasets.ImageFolder(args.data_dir, transform=eval_transform)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    loss, accuracy, macro_f1 = evaluate(model, eval_loader, device, len(eval_dataset.classes))

    cpu_model, cpu_checkpoint = load_model_from_checkpoint(checkpoint_path, torch.device("cpu"))
    if args.quantize and "int8" not in checkpoint_path.name:
        cpu_model = quantize_for_cpu(cpu_model)

    _, cpu_eval_transform = build_transforms(cpu_checkpoint["image_size"])
    cpu_dataset = datasets.ImageFolder(args.data_dir, transform=cpu_eval_transform)
    cpu_loader = DataLoader(
        cpu_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    start_time = time.time()
    with torch.no_grad():
        for images, _ in cpu_loader:
            _ = cpu_model(images)
    cpu_inference_time = time.time() - start_time

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Eval device: {device}")
    print(f"Samples: {len(eval_dataset)}")
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"CPU inference time on this dataset: {cpu_inference_time:.2f} sec")
    print(f"Estimated score on this dataset: {estimated_final_score(macro_f1, cpu_inference_time):.2f}")


def build_parser():
    parser = argparse.ArgumentParser(description="Edge AI challenge distillation baseline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--train-dir", default="competition_dataset/NEU-DET_open/train/images")
    train_parser.add_argument("--val-dir", default="competition_dataset/NEU-DET_open/validation/images")
    train_parser.add_argument("--output-dir", default="artifacts")
    train_parser.add_argument("--image-size", type=int, default=192)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--num-workers", type=int, default=0)
    train_parser.add_argument("--teacher-epochs", type=int, default=8)
    train_parser.add_argument("--student-epochs", type=int, default=18)
    train_parser.add_argument("--teacher-lr", type=float, default=3e-4)
    train_parser.add_argument("--student-lr", type=float, default=7e-4)
    train_parser.add_argument("--alpha", type=float, default=0.4)
    train_parser.add_argument("--temperature", type=float, default=3.0)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--quantize", action="store_true")
    train_parser.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])

    infer_parser = subparsers.add_parser("infer")
    infer_parser.add_argument("--test-dir", required=True)
    infer_parser.add_argument("--checkpoint", required=True)
    infer_parser.add_argument("--submission-path", default="submission.csv")
    infer_parser.add_argument("--batch-size", type=int, default=128)
    infer_parser.add_argument("--num-workers", type=int, default=0)
    infer_parser.add_argument("--seed", type=int, default=42)
    infer_parser.add_argument("--quantize", action="store_true")
    infer_parser.add_argument("--device", default="cpu", choices=["auto", "mps", "cuda", "cpu"])

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--data-dir", default="competition_dataset/NEU-DET_open/validation/images")
    eval_parser.add_argument("--checkpoint", required=True)
    eval_parser.add_argument("--batch-size", type=int, default=128)
    eval_parser.add_argument("--num-workers", type=int, default=0)
    eval_parser.add_argument("--seed", type=int, default=42)
    eval_parser.add_argument("--quantize", action="store_true")
    eval_parser.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        train_pipeline(args)
        return

    if args.command == "infer":
        inference_pipeline(args)
        return

    if args.command == "eval":
        eval_pipeline(args)
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
