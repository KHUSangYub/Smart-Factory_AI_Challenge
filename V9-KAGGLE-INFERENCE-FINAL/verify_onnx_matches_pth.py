"""
ONNX 가 PyTorch 와 동일 결과 내는지 sanity check.
export_pth_to_onnx.py 실행 후 호출.

실행:
    python verify_onnx_matches_pth.py

기대 결과: max_abs_diff < 1e-4 (FP32 변환의 정상 오차 범위)
"""
from pathlib import Path

import numpy as np
import torch
import timm
import onnxruntime as ort
from torchvision import models

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / 'V9-HONEST-179-180-FROM-SCRATCH' / 'checkpoints'
ART = HERE / 'kaggle_artifacts'

NUM_CLASSES = 6
IMG_SIZE = 192


def build_mbv3():
    s = models.mobilenet_v3_small(weights=None)
    s.classifier[-1] = torch.nn.Linear(s.classifier[-1].in_features, NUM_CLASSES)
    return s


def verify(pth_path, onnx_path, model_factory, name):
    """PyTorch 와 ONNX 결과 비교."""
    pth = torch.load(pth_path, map_location='cpu', weights_only=False)
    model = model_factory()
    model.load_state_dict(pth['state_dict'])
    model.eval()

    # Deterministic test input
    np.random.seed(42)
    x = np.random.randn(4, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)

    # PyTorch forward
    with torch.no_grad():
        pt_out = model(torch.from_numpy(x)).numpy()

    # ONNX forward
    sess = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    onnx_out = sess.run(None, {sess.get_inputs()[0].name: x})[0]

    # Compare
    abs_diff = np.abs(pt_out - onnx_out).max()
    rel_diff = abs_diff / (np.abs(pt_out).max() + 1e-8)

    # Argmax 도 비교 (실제 prediction 일치 확인)
    pt_pred = pt_out.argmax(1)
    onnx_pred = onnx_out.argmax(1)
    argmax_match = bool((pt_pred == onnx_pred).all())

    status = '✅' if (abs_diff < 1e-4 and argmax_match) else '⚠️'
    print(f"  {status} {name:30s} max_abs_diff={abs_diff:.2e}, rel_diff={rel_diff:.2e}, "
          f"argmax_match={argmax_match}")
    return abs_diff < 1e-4 and argmax_match


print("=" * 70)
print("ONNX ↔ PyTorch 결과 일치 검증")
print("=" * 70)

results = []
results.append(verify(SRC / 'student_BETA-LION.pth',
                      ART / 'student_BETA-LION.onnx',
                      build_mbv3, 'BETA-LION student'))

results.append(verify(SRC / 'student_BIDIR.pth',
                      ART / 'student_BIDIR.onnx',
                      build_mbv3, 'BIDIR student'))

results.append(verify(SRC / 'teacher_convnext_tiny.pth',
                      ART / 'teacher_convnext_tiny.onnx',
                      lambda: timm.create_model('convnext_tiny', pretrained=False,
                                                num_classes=NUM_CLASSES),
                      'ConvNeXt-Tiny teacher'))

print("=" * 70)
if all(results):
    print("✅ 모든 ONNX 가 PyTorch 와 일치 — Kaggle 업로드 OK")
else:
    print("⚠️ 일부 ONNX 가 PyTorch 와 불일치 — export 단계 점검 필요")
print("=" * 70)
