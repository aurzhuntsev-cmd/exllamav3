
## TP cache

The tensor-parallel (TP) cache captures per-device tensors during the initial model load and dumps them to disk. On subsequent loads the dumps are fast-tracked into GPU memory, which significantly reduces model loading time.

Enable with `EXL3_TP_CACHE_ENABLE=1`. Use `EXL3_TP_CACHE_FORCE=1` to skip cache validation (e.g. when doing A/B testing and changing the allocation plan repeatedly).

Measured model-loading times on a Ryzen 7950X with a 6 Gb/s NVMe drive and dual RTX 4080 GPUs:

| Model | No cache avg | Cache avg | Speedup |
| :--- | ---: | ---: | ---: |
| gemma4-26b-a4b | 9.004s | 5.897s | 1.53x |
| mistral-24b | 9.885s | 5.262s | 1.88x |
| qwen3.6-35b | 16.338s | 8.043s | 2.03x |
| qwen3.8-27b | 12.704s | 6.487s | 1.96x |
