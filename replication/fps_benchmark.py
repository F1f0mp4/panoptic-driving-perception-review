"""FPS benchmark for A-YOLOM, following the protocol of the official code.

The official repository (ultralytics/yolo/engine/validator.py, ``speed=True``)
times the raw network forward pass, following HybridNets:
  * 5 warm-up passes,
  * 1000 forward passes of one image (batch size 1),
  * 1000 forward passes of the same image repeated 32 times (batch size 32),
using torch.cuda.synchronize(). That code path only works on CUDA, so this
script re-implements the same protocol with the synchronisation primitive of
the selected device (MPS / CUDA / CPU). Input is a rect-letterboxed BDD100K
frame, i.e. 1x3x384x640 (1280x720 -> 640x360 -> padded to a multiple of 32),
which is what batch[0]['img'][0] is in the official validator.

Usage:
    python replication/fps_benchmark.py --weights data/weights/v4.pt --name ayolom_n
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'YOLOv8-multi-task'))


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()


def bench(model, x, iters, device):
    sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        model(x)
    sync(device)
    return (time.perf_counter() - t0) / iters


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
    ap.add_argument('--iters', type=int, default=1000)
    ap.add_argument('--iters32', type=int, default=1000)
    ap.add_argument('--h', type=int, default=384)
    ap.add_argument('--w', type=int, default=640)
    ap.add_argument('--name', required=True)
    ap.add_argument('--out', default=str(ROOT / 'replication' / 'results'))
    a = ap.parse_args()

    torch.manual_seed(0)
    device = torch.device(a.device)
    ckpt = torch.load(a.weights, map_location='cpu')
    model = ckpt['model'].float().fuse().eval().to(device)  # FP32, Conv+BN fused as in AutoBackend
    x = torch.rand(1, 3, a.h, a.w, device=device)

    for _ in range(5):  # warm-up, as in the official code
        model(x)
    t1 = bench(model, x, a.iters, device)
    t32 = None
    if a.iters32 > 0:
        x32 = torch.cat([x] * 32, 0)
        for _ in range(2):
            model(x32)
        t32 = bench(model, x32, a.iters32, device)

    res = {'weights': a.weights, 'device': a.device, 'input': [a.h, a.w], 'precision': 'fp32',
           'bs1': {'iters': a.iters, 'latency_ms': t1 * 1e3, 'fps': 1 / t1},
           'bs32': ({'iters': a.iters32, 'latency_ms_per_batch': t32 * 1e3, 'fps': 32 / t32} if t32 else None),
           'env': {'torch': torch.__version__, 'python': platform.python_version(),
                   'platform': platform.platform()}}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f'fps_{a.name}_{a.device}.json').write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
