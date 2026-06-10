#!/usr/bin/env python3
"""Export FastSal-FT (.pth) → ONNX + validazione fedeltà + speed CoreML-EP.

Chiude il buco scoperto il 10 giu: i pesi FT (`fastsal_ft_best.pth`) non erano mai
stati esportati — il "5.6ms/foto" del FT era assunto dall'ONNX off-the-shelf, non
misurato. Questo script: carica il FT via fastsal_loader (shim torchvision),
esporta a ONNX (input fisso 1x3x192x256, il formato nativo FastSal), verifica
max|Δ| torch-vs-onnx, misura ms/foto via onnxruntime CoreML-EP.

Output: ~/Documents/GitHib/lensiq-saliency-ft/weights/fastsal_ft.onnx
Il repo FastSal (codice sorgente) sta su iCloud post-cleanup:
  LensIQ_archive/tests/saliency_test/FastSal  → passato via env FASTSAL_DIR.

    /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
        ~/.claude/scripts/fastsal_ft_export.py
"""
from __future__ import annotations
import logging, os, sys, time
from logging.handlers import RotatingFileHandler
from pathlib import Path

import numpy as np

FASTSAL_REPO = Path.home() / "Library/Mobile Documents/com~apple~CloudDocs/LensIQ_archive/tests/saliency_test/FastSal"
os.environ.setdefault("FASTSAL_DIR", str(FASTSAL_REPO))
FT_REPO = Path.home() / "Documents/GitHib/lensiq-saliency-ft"
CKPT = FT_REPO / "weights/fastsal_ft_best.pth"
ONNX_OUT = FT_REPO / "weights/fastsal_ft.onnx"
PHOTOS = Path.home() / "deepgaze_coreml_bench/photos"   # 100 foto test del bundle

LOG_DIR = Path.home() / ".claude/scripts/logs"; LOG_DIR.mkdir(parents=True, exist_ok=True)
_fmt = logging.Formatter("%(asctime)s.%(msecs)03d %(levelname)s %(message)s", "%Y-%m-%dT%H:%M:%S")
log = logging.getLogger("fastsal_export"); log.setLevel(logging.INFO)
_ch = logging.StreamHandler(); _ch.setFormatter(_fmt); log.addHandler(_ch)
_fh = RotatingFileHandler(LOG_DIR / "fastsal_ft_export.log", maxBytes=2*1024*1024, backupCount=3)
_fh.setFormatter(_fmt); log.addHandler(_fh)


def main() -> int:
    if not CKPT.exists():
        log.error("manca il checkpoint FT %s", CKPT); return 2
    if not (FASTSAL_REPO / "model").exists():
        log.error("repo FastSal non accessibile: %s (dataless su iCloud? aprire la dir in Finder)", FASTSAL_REPO)
        return 2

    sys.path.insert(0, str(FT_REPO))
    import torch
    from fastsal_loader import load_fastsal, preprocess_np

    # ── 1) carica FT ──
    t = time.time()
    model = load_fastsal(from_checkpoint=str(CKPT), device="cpu")
    model.eval()
    log.info("FastSal-FT caricato in %.1fs (%s)", time.time() - t, CKPT.name)

    # ── 2) export ONNX (input fisso, formato nativo 192x256) ──
    dummy = torch.zeros(1, 3, 192, 256, dtype=torch.float32)
    t = time.time()
    torch.onnx.export(
        model, (dummy,), str(ONNX_OUT),
        export_params=True, opset_version=17, do_constant_folding=True,
        input_names=["image"], output_names=["saliency"], dynamo=False,
    )
    log.info("✅ export ONNX OK in %.1fs → %s (%.1f MB)",
             time.time() - t, ONNX_OUT.name, ONNX_OUT.stat().st_size / 1e6)

    # ── 3) fedeltà torch vs onnxruntime (CPU exact) su foto reali ──
    photos = sorted(PHOTOS.glob("*.jpg"))[:20]
    if not photos:
        log.warning("nessuna foto in %s — fedeltà su input random", PHOTOS)
    import onnxruntime as ort
    sess_cpu = ort.InferenceSession(str(ONNX_OUT), providers=["CPUExecutionProvider"])
    max_diff = 0.0
    with torch.no_grad():
        for p in (photos or [None] * 3):
            if p is None:
                x = np.random.default_rng(0).standard_normal((1, 3, 192, 256)).astype(np.float32)
            else:
                x = preprocess_np(str(p))[None]
            o_t = model(torch.from_numpy(x)).numpy()
            o_o = sess_cpu.run(None, {"image": x})[0]
            max_diff = max(max_diff, float(np.max(np.abs(o_t - o_o))))
    log.info("fedeltà torch-vs-onnx (CPU, n=%d): max|Δ| = %.3g", len(photos) or 3, max_diff)

    # ── 4) speed via CoreML-EP (lo stack dei 5.6ms off-the-shelf) ──
    prov = [p for p in ("CoreMLExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
    sess_cm = ort.InferenceSession(str(ONNX_OUT), providers=prov or None)
    used = sess_cm.get_providers()[0]
    feeds = [{"image": preprocess_np(str(p))[None]} for p in photos] or \
            [{"image": np.zeros((1, 3, 192, 256), np.float32)}]
    sess_cm.run(None, feeds[0])  # warm
    ts = []
    for f in feeds:
        for _ in range(3):
            t = time.time(); sess_cm.run(None, f); ts.append(time.time() - t)
    med = float(np.median(ts)) * 1000

    print("\n" + "=" * 70)
    print("FastSal-FT → ONNX — export + validazione")
    print("=" * 70)
    print(f"  ONNX                : {ONNX_OUT}  ({ONNX_OUT.stat().st_size/1e6:.1f} MB)")
    print(f"  fedeltà max|Δ|      : {max_diff:.3g}  ({'✅ identico (fp32)' if max_diff < 1e-3 else '⚠️ da capire'})")
    print(f"  speed {used:<22}: {med:.1f} ms/foto (median, n={len(feeds)}x3)")
    print(f"  riferimento off-the-shelf (mag): 5.6 ms/foto")
    print("=" * 70)
    return 0 if max_diff < 1e-3 else 1


if __name__ == "__main__":
    sys.exit(main())
