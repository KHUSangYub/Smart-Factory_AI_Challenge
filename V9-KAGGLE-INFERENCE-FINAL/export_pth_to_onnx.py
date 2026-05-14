"""
사용자 환경에서 1회 실행 — V9-HONEST 의 .pth + status.json + train_ood_feats.npz
→ kaggle_artifacts/ 에 3 ONNX + thresholds.json + ah12_state.pkl 생성.

실행:
    cd output/decisions/V9-KAGGLE-INFERENCE-FINAL
    python export_pth_to_onnx.py

생성 후:
    kaggle_artifacts/
    ├── student_BETA-LION.onnx
    ├── student_BIDIR.onnx
    ├── teacher_convnext_tiny.onnx
    ├── thresholds.json         (R3/R4/P_VETO/BIDIR_ROL_VETO + wA/class_bias)
    └── ah12_state.pkl          (AH12 의 sklearn scaler + clf + Gaussian fit)

이 폴더를 Kaggle Dataset 으로 업로드한 후 kaggle-inference.ipynb 에서 attach.
"""
import json
import pickle
from pathlib import Path

import numpy as np
import onnx
import torch
import timm
from torchvision import models
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV


def _inline_external_data(onnx_path: Path):
    """torch 2.11+ dynamo exporter 가 만든 .onnx.data 외부 weight 파일을
    .onnx 안에 inline 으로 합침. Kaggle 업로드 시 single file 로 깔끔.
    """
    data_file = onnx_path.with_suffix('.onnx.data')
    if not data_file.exists():
        return  # 이미 single file
    # 외부 데이터까지 같이 load
    m = onnx.load(str(onnx_path), load_external_data=True)
    # inline 으로 다시 저장
    onnx.save(m, str(onnx_path), save_as_external_data=False)
    # 외부 데이터 파일 삭제
    data_file.unlink()

# ============================================================================
# Paths
# ============================================================================
HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'V9-HONEST-179-180-FROM-SCRATCH'
ARTIFACTS = HERE / 'kaggle_artifacts'
ARTIFACTS.mkdir(parents=True, exist_ok=True)

CKPT_DIR = SOURCE / 'checkpoints'
PTH_BETA = CKPT_DIR / 'student_BETA-LION.pth'
PTH_BIDIR = CKPT_DIR / 'student_BIDIR.pth'
PTH_CNXT = CKPT_DIR / 'teacher_convnext_tiny.pth'

STATUS_JSON = SOURCE / 'status.json'
TRAIN_OOD_FEATS = SOURCE / 'train_ood_feats.npz'

# Config
SEED = 42
NUM_CLASSES = 6
IMG_SIZE = 192

# ============================================================================
# Step 1 — PyTorch → ONNX (FP32) 변환
# ============================================================================
print("=" * 60)
print("Step 1: PyTorch → ONNX (FP32) 변환")
print("=" * 60)


def build_mbv3():
    s = models.mobilenet_v3_small(weights=None)
    s.classifier[-1] = torch.nn.Linear(s.classifier[-1].in_features, NUM_CLASSES)
    return s


def export_onnx(model, ckpt_path, out_path):
    """학습된 .pth → ONNX 변환.
    opset 17 사용 (LayerNorm/GELU/GroupNorm 의 native op 변환 → 정확도 향상).
    do_constant_folding=False — constant folding 으로 인한 미세 정밀도 손실 차단.
    """
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    torch.onnx.export(
        model, dummy, str(out_path),
        input_names=['input'], output_names=['logits'],
        dynamic_axes={'input': {0: 'batch'}, 'logits': {0: 'batch'}},
        opset_version=17,                # ⭐ 17 (LayerNorm native), 이전 14 (우회 변환 → 미세 손실)
        do_constant_folding=False,       # ⭐ constant fold 끄기 (정확도 우선)
        dynamo=False,                    # legacy exporter (single file + INT8 호환)
    )
    _inline_external_data(out_path)
    size_mb = out_path.stat().st_size / 1024**2
    print(f"  ✅ {out_path.name:35s} ({size_mb:.1f} MB)")


# BETA student (MobileNetV3-Small)
export_onnx(build_mbv3(), PTH_BETA, ARTIFACTS / 'student_BETA-LION.onnx')

# BIDIR student (MobileNetV3-Small)
export_onnx(build_mbv3(), PTH_BIDIR, ARTIFACTS / 'student_BIDIR.onnx')

# ConvNeXt-Tiny teacher
cnxt = timm.create_model('convnext_tiny', pretrained=False, num_classes=NUM_CLASSES)
ckpt = torch.load(PTH_CNXT, map_location='cpu', weights_only=False)
cnxt.load_state_dict(ckpt['state_dict'])
cnxt.eval()
cnxt_onnx_path = ARTIFACTS / 'teacher_convnext_tiny.onnx'
dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
torch.onnx.export(
    cnxt, dummy, str(cnxt_onnx_path),
    input_names=['input'], output_names=['logits'],
    dynamic_axes={'input': {0: 'batch'}, 'logits': {0: 'batch'}},
    opset_version=17,            # ⭐ LayerNorm native (ConvNeXt-T 핵심)
    do_constant_folding=False,   # ⭐ 정확도 우선
    dynamo=False,
)
_inline_external_data(cnxt_onnx_path)
size_mb = cnxt_onnx_path.stat().st_size / 1024**2
print(f"  ✅ {'teacher_convnext_tiny.onnx':35s} ({size_mb:.1f} MB)")

# ============================================================================
# Step 2 — status.json → thresholds.json
# ============================================================================
print("\n" + "=" * 60)
print("Step 2: status.json → thresholds.json (Phase F fit 결과)")
print("=" * 60)

with open(STATUS_JSON) as f:
    status = json.load(f)

# ALLOWED_TARGETS: [(target_class, cnxt_floor), ...]
allowed = [[at['target_class'], at['cnxt_floor']]
           for at in status.get('rule_3way_allowed_targets', [])]

r4 = status.get('r4_thresholds', {})

thresholds = {
    # Champion ensemble 보정 anchor (train multi-OOD 에서 fit)
    'wA': 0.7,
    'class_bias': [2.0, -1.1, 1.9, -0.1, 1.2, -0.7],

    # R3_pertarget
    'rule_3way_valid': bool(status.get('rule_3way_valid', False)),
    'ALLOWED_TARGETS': allowed,

    # P_VETO (pit → inc flip 차단)
    'P_VETO': float(status.get('p_veto', 1.0)),

    # R4_pit (3D grid + BIDIR_ROL_VETO)
    'r4_valid': bool(status.get('r4_valid', False)),
    'R4_cnxt_floor':  float(r4.get('cnxt_floor', 1.01)),
    'R4_beta_pit_floor': float(r4.get('beta_pit_floor', 1.01)),
    'R4_p_pit_floor':  float(r4.get('p_pit_floor', 1.01)),
    'BIDIR_ROL_VETO':  float(r4.get('bidir_rol_veto', 1.01)),

    # AH12 valid 여부 (실제 임계값은 ah12_state.pkl)
    'ah12_valid': bool(status.get('rule_A_AH12_valid', False)),

    # Class mapping (Python sorted)
    'class_names': ['crazing', 'inclusion', 'patches',
                    'pitted_surface', 'rolled-in_scale', 'scratches'],
    'CRA': 0, 'INC': 1, 'PAT': 2, 'PIT': 3, 'ROL': 4, 'SCR': 5,

    # 입력 spec
    'IMG_SIZE': IMG_SIZE,
    'NUM_CLASSES': NUM_CLASSES,
    'SEED': SEED,
}

with open(ARTIFACTS / 'thresholds.json', 'w') as f:
    json.dump(thresholds, f, indent=2)
print(f"  ✅ thresholds.json")
print(f"     ALLOWED_TARGETS: {allowed}")
print(f"     P_VETO: {thresholds['P_VETO']:.4f}")
print(f"     R4: cf={thresholds['R4_cnxt_floor']}, bf={thresholds['R4_beta_pit_floor']}, "
      f"pf={thresholds['R4_p_pit_floor']}, veto={thresholds['BIDIR_ROL_VETO']}")

# ============================================================================
# Step 3 — AH12 sklearn 객체 재학습 + pickle
# (V9 노트북의 Cell 30 Phase F.2 와 동일한 fit 절차)
# ============================================================================
print("\n" + "=" * 60)
print("Step 3: AH12 sklearn 객체 재학습 + pickle")
print("=" * 60)

feats_npz = np.load(TRAIN_OOD_FEATS)
feats_inc = feats_npz['inc']  # (1200, 13)
feats_pit = feats_npz['pit']  # (1200, 13)
print(f"  feats_inc: {feats_inc.shape}, feats_pit: {feats_pit.shape}")


def fit_gauss(X):
    mu = X.mean(0)
    S = np.cov(X.T) + 1e-6 * np.eye(X.shape[1])
    Sinv = np.linalg.inv(S)
    _, ld = np.linalg.slogdet(S)
    return mu, Sinv, ld


def ll(x, mu, Sinv, ld):
    return -0.5 * ((x - mu) @ Sinv @ (x - mu)) - 0.5 * ld


def d_maha(x, mu, Sinv):
    return float((x - mu) @ Sinv @ (x - mu))


# Gaussian fit on 3D head (FFT, coh, cc)
mu_inc3, Sinv_inc3, ld_inc3 = fit_gauss(feats_inc[:, :3])
mu_pit3, Sinv_pit3, ld_pit3 = fit_gauss(feats_pit[:, :3])

# R1: percentile cut on inclusion
A_r1_fft = float(np.percentile(feats_inc[:, 0], 5))
A_r1_coh = float(np.percentile(feats_inc[:, 1], 95))

# R2: Gaussian log-likelihood ratio threshold
inc_lr = np.array([
    ll(x, mu_pit3, Sinv_pit3, ld_pit3) - ll(x, mu_inc3, Sinv_inc3, ld_inc3)
    for x in feats_inc[:, :3]
])
A_r2_thr = float(np.percentile(inc_lr, 99))

# R3: Mahalanobis bound
pit_d_pit = np.array([d_maha(x, mu_pit3, Sinv_pit3) for x in feats_pit[:, :3]])
A_r3_thr = float(np.percentile(pit_d_pit, 99))

# R4: calibrated LogReg P(pit | 13D)
X_ip = np.concatenate([feats_inc, feats_pit], axis=0)
y_ip = np.concatenate([np.zeros(len(feats_inc)), np.ones(len(feats_pit))])
scaler = StandardScaler().fit(X_ip)
clf = CalibratedClassifierCV(
    LogisticRegression(C=1.0, max_iter=2000,
                       class_weight='balanced', random_state=SEED),
    method='isotonic', cv=5
).fit(scaler.transform(X_ip), y_ip)

p_pit_inc = clf.predict_proba(scaler.transform(feats_inc))[:, 1]
A_r4_thr = float(np.percentile(p_pit_inc, 99))

# Pickle 저장
ah12_state = {
    # sklearn 객체
    'scaler': scaler,
    'clf': clf,

    # Gaussian fit params (3D head)
    'mu_inc3': mu_inc3, 'Sinv_inc3': Sinv_inc3, 'ld_inc3': ld_inc3,
    'mu_pit3': mu_pit3, 'Sinv_pit3': Sinv_pit3, 'ld_pit3': ld_pit3,

    # 4 sub-rule thresholds
    'A_r1_fft': A_r1_fft, 'A_r1_coh': A_r1_coh,
    'A_r2_thr': A_r2_thr,
    'A_r3_thr': A_r3_thr,
    'A_r4_thr': A_r4_thr,
}

with open(ARTIFACTS / 'ah12_state.pkl', 'wb') as f:
    pickle.dump(ah12_state, f)
print(f"  ✅ ah12_state.pkl")
print(f"     A_r1: fft<{A_r1_fft:.4f}, coh>{A_r1_coh:.4f}")
print(f"     A_r2: LR > {A_r2_thr:.4f}")
print(f"     A_r3: d_pit < d_inc AND < {A_r3_thr:.4f}")
print(f"     A_r4: P(pit) > {A_r4_thr:.4f}")

# ============================================================================
# Step 4 — ONNX FP32 → INT8 dynamic quantization (3개 모델 모두)
# 별도 파일명 (_int8 suffix) 으로 저장 → FP32 도 유지, 선택 가능
# ============================================================================
print("\n" + "=" * 60)
print("Step 4: INT8 dynamic quantization (3 ONNX 모델 모두)")
print("=" * 60)

try:
    from onnxruntime.quantization import quantize_dynamic, QuantType
    from onnxruntime.quantization.shape_inference import quant_pre_process
except ImportError:
    print("⚠️ onnxruntime.quantization 없음. `pip install onnxruntime` 필요")
    raise

QUANT_PAIRS = [
    ('student_BETA-LION.onnx',      'student_BETA-LION_int8.onnx'),
    ('student_BIDIR.onnx',          'student_BIDIR_int8.onnx'),
    ('teacher_convnext_tiny.onnx',  'teacher_convnext_tiny_int8.onnx'),
]

for fp32_name, int8_name in QUANT_PAIRS:
    fp32_path = ARTIFACTS / fp32_name
    int8_path = ARTIFACTS / int8_name
    preproc_path = ARTIFACTS / f"_preproc_{fp32_name}"
    if not fp32_path.exists():
        print(f"  ⚠️ FP32 없음: {fp32_name} (skip)")
        continue
    try:
        # Step 1: shape inference + ONNX 그래프 전처리 (dynamo exporter 호환)
        quant_pre_process(
            input_model_path=str(fp32_path),
            output_model_path=str(preproc_path),
            skip_optimization=False,
            skip_onnx_shape=False,
            skip_symbolic_shape=False,
        )
        # Step 2: INT8 dynamic quantization
        quantize_dynamic(
            model_input=str(preproc_path),
            model_output=str(int8_path),
            weight_type=QuantType.QInt8,
        )
        # Step 3: 임시 preproc 파일 삭제
        preproc_path.unlink(missing_ok=True)

        fp32_mb = fp32_path.stat().st_size / 1024**2
        int8_mb = int8_path.stat().st_size / 1024**2
        ratio = fp32_mb / int8_mb if int8_mb > 0 else 0
        print(f"  ✅ {int8_name:40s} {int8_mb:6.1f} MB "
              f"(FP32 {fp32_mb:6.1f} MB → {ratio:.1f}× ↓)")
    except Exception as e:
        # 실패 시 임시 파일도 정리
        preproc_path.unlink(missing_ok=True)
        int8_path.unlink(missing_ok=True)
        print(f"  ❌ {int8_name}: 실패 ({e.__class__.__name__}: {str(e)[:120]})")

# ============================================================================
# 완료
# ============================================================================
total_size = sum(f.stat().st_size for f in ARTIFACTS.iterdir() if f.is_file())
print("\n" + "=" * 60)
print(f"✅ 모든 artifact 생성 완료 → {ARTIFACTS}")
print(f"   총 크기: {total_size / 1024**2:.1f} MB")
print("=" * 60)
print()
print("📝 다음 단계:")
print("   1. python verify_onnx_matches_pth.py  # ONNX 가 PyTorch 와 일치하는지 검증")
print("   2. kaggle_artifacts/ 폴더의 5개 파일을 Kaggle Dataset 으로 업로드")
print("      (Dataset 이름 예: v9-kaggle-inference-final)")
print("   3. kaggle-inference.ipynb 를 Kaggle 에 fork → Dataset attach → run")
