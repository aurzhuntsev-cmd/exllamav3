# TP cache

Captures per-device tensors in TP mode during the initial model load and dumps them to disk. On subsequent loads, fast-tracks dumps into GPU memory to improve model loading speed.

Experimental and hacky.

Enable with `EXL3_TP_CACHE_ENABLE=1`