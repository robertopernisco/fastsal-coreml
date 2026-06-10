# fastsal-coreml

**FastSal-FT → CoreML**: il FastSal fine-tuned LensIQ (distillato da DeepGaze MSDB sul
corpus privato 37.757 foto, vedi repo `lensiq-saliency-ft`) convertito in `.mlpackage`.
Saliency on-device ultra-veloce: **2.83 ms/foto** su Apple Silicon.

A differenza di DeepGaze (repo `deepgaze-coreml`), qui il modello È nel repo (7.4 MB)
insieme ai pesi torch FT — questo repo privato fa anche da **backup dei pesi FT**,
che altrove sono gitignored.

## Risultati

| | valore | note |
|---|---|---|
| **Velocità** | **2.83 ms/foto** CPU_AND_GPU (Mac16 M2) | CPU_ONLY 3.67 ms |
| **Fedeltà conversione** | **CC 0.99999** vs torch-FT fp32 | fp16; max\|Δ\| 0.165 su scala logit = rumore |
| **Qualità vs DeepGaze** | gridCC **0.83** (val 3775 foto) | dal FT: Δcenter_bias 0.074, vedi lensiq-saliency-ft |
| Input | **fisso 192×256**, ImageNet-norm | `fastsal_loader.preprocess_np` |
| Architettura | MobileNetV2 (~3.5 M param) | distillato sui target consumati (grid/peak/cb/disp) |

Trade-off vs DeepGaze-CoreML: **~1580× più veloce** (2.8 ms vs 4.47 s), qualità
approssimata (gridCC 0.83 vs 0.999). La scelta è per-caso-d'uso, vedi memo saliency.

## Contenuto

- `weights/fastsal_ft.mlpackage` — il modello CoreML pronto (fp16, 7.4 MB)
- `weights/fastsal_ft_best.pth` — pesi torch FT (epoch 4, gridCC 0.8311) — BACKUP
- `convert_fastsal_coreml.py` — riconversione .pth → .mlpackage (trace + ct.convert, ~2 s)
- `export_fastsal_onnx.py` — export alternativo .pth → ONNX (per onnxruntime/CoreML-EP)
- `fastsal_loader.py` — loader torch con shim torchvision + preprocessing (192×256)

## Uso

```python
import coremltools as ct, numpy as np
from fastsal_loader import preprocess_np

m = ct.models.MLModel("weights/fastsal_ft.mlpackage",
                      compute_units=ct.ComputeUnit.CPU_AND_GPU)
x = preprocess_np("foto.jpg")[None]          # [1,3,192,256] fp32
sal = m.predict({"image": x})["saliency"]    # mappa saliency (logit, min-max a valle)
```

Per riconvertire da zero (richiede il repo sorgente FastSal per il codice modello —
path via env `FASTSAL_DIR`):

```bash
pip install torch torchvision coremltools pillow numpy
python convert_fastsal_coreml.py
```

NB: `CPU_AND_NE` escluso — la compilazione ANE fallisce su Apple Silicon (stesso
esito di DeepGaze: CoreML = GPU only su queste macchine).
