#!/usr/bin/env python3
"""FastSal-FT (.pth) → CoreML .mlpackage — uniforma lo stack saliency a coremltools.

Decisione 10 giu: in produzione UN solo runtime (CoreML/.mlpackage) per entrambi
i backend saliency (DeepGaze già .mlpackage; FastSal-FT era ONNX+CoreML-EP).
Stesso protocollo di deepgaze_coreml_probe_v4: jit.trace → ct.convert(mlprogram,
fp16 default) → fedeltà max|Δ| + CC vs torch fp32 su foto reali → speed per
compute unit. ANE escluso (ANECCompile FAILED su queste macchine, crash duro).

Output: ~/Documents/GitHib/lensiq-saliency-ft/weights/fastsal_ft.mlpackage

    /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
        ~/.claude/scripts/fastsal_ft_coreml.py
"""
from __future__ import annotations
import logging, os, shutil, sys, time
from logging.handlers import RotatingFileHandler
from pathlib import Path

import numpy as np

FASTSAL_REPO = Path.home() / "Library/Mobile Documents/com~apple~CloudDocs/LensIQ_archive/tests/saliency_test/FastSal"
os.environ.setdefault("FASTSAL_DIR", str(FASTSAL_REPO))
FT_REPO = Path.home() / "Documents/GitHib/lensiq-saliency-ft"
CKPT = FT_REPO / "weights/fastsal_ft_best.pth"
MLPKG_OUT = FT_REPO / "weights/fastsal_ft.mlpackage"
PHOTOS = Path.home() / "deepgaze_coreml_bench/photos"

LOG_DIR = Path.home() / ".claude/scripts/logs"; LOG_DIR.mkdir(parents=True, exist_ok=True)
_fmt = logging.Formatter("%(asctime)s.%(msecs)03d %(levelname)s %(message)s", "%Y-%m-%dT%H:%M:%S")
log = logging.getLogger("fastsal_coreml"); log.setLevel(logging.INFO)
_ch = logging.StreamHandler(); _ch.setFormatter(_fmt); log.addHandler(_ch)
_fh = RotatingFileHandler(LOG_DIR / "fastsal_ft_coreml.log", maxBytes=2*1024*1024, backupCount=3)
_fh.setFormatter(_fmt); log.addHandler(_fh)


def main() -> int:
    if not CKPT.exists():
        log.error("manca %s", CKPT); return 2
    if not (FASTSAL_REPO / "model").exists():
        log.error("repo FastSal non accessibile: %s", FASTSAL_REPO); return 2

    sys.path.insert(0, str(FT_REPO))
    import torch
    import coremltools as ct
    from fastsal_loader import load_fastsal, preprocess_np

    # ── 1) carica FT + trace ──
    t = time.time()
    model = load_fastsal(from_checkpoint=str(CKPT), device="cpu")
    model.eval()
    log.info("FastSal-FT caricato in %.1fs", time.time() - t)

    dummy = torch.zeros(1, 3, 192, 256, dtype=torch.float32)
    t = time.time()
    with torch.no_grad():
        traced = torch.jit.trace(model, (dummy,), check_trace=False)
    log.info("trace OK in %.1fs", time.time() - t)

    # ── 2) convert → mlprogram (fp16 default, come DeepGaze v4) ──
    t = time.time()
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="image", shape=(1, 3, 192, 256), dtype=np.float32)],
        outputs=[ct.TensorType(name="saliency")],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS13,
    )
    if MLPKG_OUT.exists():
        shutil.rmtree(MLPKG_OUT)
    mlmodel.save(str(MLPKG_OUT))
    sz = sum(f.stat().st_size for f in MLPKG_OUT.rglob("*") if f.is_file()) / 1e6
    log.info("✅ CONVERT + SAVE OK in %.1fs → %s (%.1f MB)", time.time() - t, MLPKG_OUT.name, sz)

    # ── 3) fedeltà vs torch fp32 su foto reali ──
    photos = sorted(PHOTOS.glob("*.jpg"))[:20]
    cm = ct.models.MLModel(str(MLPKG_OUT), compute_units=ct.ComputeUnit.CPU_AND_GPU)
    out_key = None; max_diff = 0.0; ccs = []
    with torch.no_grad():
        for p in photos:
            x = preprocess_np(str(p))[None]
            o_t = model(torch.from_numpy(x)).numpy().ravel()
            o = cm.predict({"image": x})
            if out_key is None:
                out_key = list(o.keys())[0]
            o_c = np.asarray(o[out_key]).astype(np.float32).ravel()
            max_diff = max(max_diff, float(np.max(np.abs(o_t - o_c))))
            if o_t.std() > 1e-12 and o_c.std() > 1e-12:
                ccs.append(float(np.corrcoef(o_t, o_c)[0, 1]))
    cc_mean = float(np.mean(ccs)) if ccs else float("nan")
    cc_min = float(np.min(ccs)) if ccs else float("nan")
    log.info("fedeltà (n=%d): max|Δ| %.3g · CC mean %.5f / min %.5f", len(photos), max_diff, cc_mean, cc_min)

    # ── 4) speed per compute unit (no ANE: crash noto) ──
    feeds = [{"image": preprocess_np(str(p))[None]} for p in photos]
    results = {}
    for cu_name in ("CPU_ONLY", "CPU_AND_GPU"):
        m = ct.models.MLModel(str(MLPKG_OUT), compute_units=getattr(ct.ComputeUnit, cu_name))
        m.predict(feeds[0])  # warm
        ts = []
        for f in feeds:
            for _ in range(3):
                t = time.time(); m.predict(f); ts.append(time.time() - t)
        results[cu_name] = float(np.median(ts)) * 1000
        log.info("  [%s] median %.2f ms/foto", cu_name, results[cu_name])
        del m

    print("\n" + "=" * 70)
    print("FastSal-FT → CoreML .mlpackage — uniformazione stack")
    print("=" * 70)
    print(f"  .mlpackage          : {MLPKG_OUT}  ({sz:.1f} MB)")
    print(f"  fedeltà vs torch    : max|Δ| {max_diff:.3g} · CC {cc_mean:.5f} (min {cc_min:.5f})")
    for cu, ms in results.items():
        print(f"  speed {cu:<13}: {ms:.2f} ms/foto")
    print(f"  riferimento ONNX+CoreML-EP: 3.9 ms/foto")
    print("=" * 70)
    ok = cc_mean > 0.999 if ccs else False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
