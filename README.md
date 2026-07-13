# fastsal-coreml

**FastSal-FT → CoreML**: LensIQ's fine-tuned FastSal (distilled from DeepGaze MSDB on a
37,757-photo private corpus, see the `lensiq-saliency-ft` repo) converted to
`.mlpackage`. Ultra-fast on-device saliency: **2.83 ms/photo** on Apple Silicon.

Unlike DeepGaze (see the `deepgaze-coreml` repo), here the model **is in the repo**
(7.4 MB) together with the fine-tuned torch weights.

## Results

| | value | notes |
|---|---|---|
| **Speed** | **2.83 ms/photo** CPU_AND_GPU (M2, 16GB) | CPU_ONLY 3.67 ms |
| **Conversion fidelity** | **CC 0.99999** vs torch-FT fp32 | fp16; max\|Δ\| 0.165 on logit scale = noise |
| **Quality vs DeepGaze** | grid CC **0.83** (3,775-photo val) | from the FT: Δcenter_bias 0.074, see lensiq-saliency-ft |
| Input | **fixed 192×256**, ImageNet norm | `fastsal_loader.preprocess_np` |
| Architecture | MobileNetV2 (~3.5 M params) | distilled on the consumed targets (grid/peak/cb/disp) |

Trade-off vs DeepGaze-CoreML: **~1580× faster** (2.8 ms vs 4.47 s), approximate quality
(grid CC 0.83 vs 0.999). Pick per use case: near-free saliency → FastSal-FT;
maximum fidelity → DeepGaze-CoreML.

## Contents

- `weights/fastsal_ft.mlpackage` — the ready-to-use CoreML model (fp16, 7.4 MB)
- `weights/fastsal_ft_best.pth` — fine-tuned torch weights (epoch 4, grid CC 0.8311)
- `convert_fastsal_coreml.py` — reconversion .pth → .mlpackage (trace + ct.convert, ~2 s)
- `export_fastsal_onnx.py` — alternative export .pth → ONNX (for onnxruntime / CoreML-EP)
- `fastsal_loader.py` — torch loader with torchvision shims + preprocessing (192×256)

## Usage

```python
import coremltools as ct, numpy as np
from fastsal_loader import preprocess_np

m = ct.models.MLModel("weights/fastsal_ft.mlpackage",
                      compute_units=ct.ComputeUnit.CPU_AND_GPU)
x = preprocess_np("photo.jpg")[None]         # [1,3,192,256] fp32
sal = m.predict({"image": x})["saliency"]    # saliency map (logits; min-max downstream)
```

To reconvert from scratch (requires the FastSal source repo for the model code —
path via the `FASTSAL_DIR` env var):

```bash
pip install torch torchvision coremltools pillow numpy
python convert_fastsal_coreml.py
```

NB: `CPU_AND_NE` is excluded — ANE compilation fails on Apple Silicon (same outcome
as DeepGaze: CoreML = GPU only on these machines).

## Provenance & license notes

The fine-tuned weights are distilled from the outputs of
[DeepGaze MSDB](https://github.com/matthias-k/DeepGaze) (research use) on a private
photo corpus; the FastSal architecture and initialization come from
[FastSal](https://github.com/feiyanhu/FastSal) (SALICON weights). Released for
research/portfolio purposes — check the upstream licenses before any commercial use.
