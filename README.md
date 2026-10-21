# TP cache

Captures per-device tensors in TP mode during the initial model load and dumps them to disk. On subsequent loads, fast-tracks dumps into GPU memory to improve model loading speed.

Enable with `EXL3_TP_CACHE_ENABLE=1`
Use `EXL3_TP_CACHE_FORCE=1` to skip cache validation (e.g. when doing some A/B testing and allocation plan changes all the time)

Some random models I tested on my hardware (Ryzen 7950X, 6Gb/sec NVMe, dual 4080):
| Model | No cache vvg | Cache avg | Speedup |
| :--- | ---: | ---: | ---: |
| gemma4-26b-a4b | 9.004s | 5.897s | 1.53x |
| mistral-24b | 9.885s | 5.262s | 1.88x |
| qwen3.6-35b | 16.338s | 8.043s | 2.03x |
| qwen3.8-27b | 12.704s | 6.487s | 1.96x |
