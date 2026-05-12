import torch
import torch.nn as nn
import numpy as np

def distillation_loss(
    student_logits: torch.Tensor,
    labels: torch.Tensor,
    teacher_logits: torch.Tensor,
    alpha: float,
    temperature: float,
) -> torch.Tensor:
    ce_loss = nn.functional.cross_entropy(student_logits, labels, label_smoothing=0.05)

    # 두 확률분포의 차이를 계산하는 통계식
    kd_loss = nn.functional.kl_div(
        nn.functional.log_softmax(student_logits / temperature, dim=1),
        nn.functional.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean",
    ) * (temperature ** 2)

    return alpha * ce_loss + (1.0 - alpha) * kd_loss

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