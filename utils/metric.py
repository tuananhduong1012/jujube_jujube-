"""utils/metrics.py – classification metrics + FPS measurement."""

import time
import torch
from sklearn.metrics import classification_report


def compute_metrics(labels, preds, class_names):
    """Return sklearn classification_report as dict."""
    return classification_report(
        labels, preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )


@torch.no_grad()
def measure_fps(model, device, img_size=224, batch_size=1, n_warmup=50, n_runs=200):
    """
    Returns:
        fps        – images per second
        latency_ms – ms per image  (meaningful only for batch_size=1)
    """
    model.eval()
    dummy = torch.randn(batch_size, 3, img_size, img_size).to(device)

    # warm-up
    for _ in range(n_warmup):
        _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(n_runs):
        _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    fps        = (n_runs * batch_size) / elapsed
    latency_ms = elapsed / n_runs * 1000
    return fps, latency_ms
