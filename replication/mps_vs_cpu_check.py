"""Diagnose the CPU-vs-MPS discrepancy of A-YOLOM inference (PyTorch 2.2.2).

Runs the same pre-processed BDD100K validation image through the released model on CPU and on
Apple MPS (FP32, Conv+BN fused as in AutoBackend), records the output of every top-level layer with
forward hooks and prints the first layers whose outputs differ.

Usage:
    python replication/mps_vs_cpu_check.py --weights data/weights/v4.pt --image data/bdd/images/val2017/<id>.jpg
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'YOLOv8-multi-task'))


def flat(o):
    if isinstance(o, torch.Tensor):
        return [o]
    if isinstance(o, (list, tuple)):
        return [t for x in o for t in flat(x)]
    return []


def run(model, x, device):
    m = model.to(device)
    outs = {}
    hooks = [layer.register_forward_hook(lambda mod, i, o, k=k: outs.__setitem__(k, [t.detach().float().cpu() for t in flat(o)]))
             for k, layer in enumerate(m.model)]
    with torch.no_grad():
        y = m(x.to(device))
    for h in hooks:
        h.remove()
    return outs, [t.detach().float().cpu() for t in flat(y)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--image', required=True)
    ap.add_argument('--out', default=str(ROOT / 'replication' / 'results' / 'mps_vs_cpu.json'))
    a = ap.parse_args()
    im = cv2.imread(a.image)
    # letterbox 1280x720 -> 672x378, padded with grey (114) to 672x384, as in the rect validation loader
    h, w = im.shape[:2]
    r = min(384 / h, 672 / w)
    nh, nw = round(h * r), round(w * r)
    im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (384 - nh) // 2, (672 - nw) // 2
    im = cv2.copyMakeBorder(im, top, 384 - nh - top, left, 672 - nw - left, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    x = torch.from_numpy(im[..., ::-1].transpose(2, 0, 1).copy()).float()[None] / 255
    ck = torch.load(a.weights, map_location='cpu')
    model_c = ck['model'].float().fuse().eval()
    ck = torch.load(a.weights, map_location='cpu')
    model_m = ck['model'].float().fuse().eval()
    oc, yc = run(model_c, x, 'cpu')
    om, ym = run(model_m, x, 'mps')
    report = []
    for k in sorted(oc):
        diffs = [float((p - q).abs().max()) / (float(p.abs().max()) + 1e-6) for p, q in zip(oc[k], om[k])]
        report.append({'layer': k, 'type': type(model_c.model[k]).__name__, 'rel_max_diff': max(diffs) if diffs else None})
    first_bad = next((r for r in report if r['rel_max_diff'] is not None and r['rel_max_diff'] > 1e-2), None)
    final = [float((p - q).abs().max()) for p, q in zip(yc, ym)]
    res = {'image': a.image, 'torch': torch.__version__, 'first_layer_rel_diff_gt_1e-2': first_bad,
           'final_outputs_abs_max_diff': final, 'per_layer': report}
    Path(a.out).write_text(json.dumps(res, indent=2))
    for r in report:
        print(r)
    print('FIRST >1e-2:', first_bad)
    print('final outputs max abs diff:', final)


if __name__ == '__main__':
    main()
