"""Minimal reproduction of the MPS issue found during the replication (PyTorch 2.2.2, macOS, Apple M3 Pro).

The official A-YOLOM evaluation code applies in-place operations to *views* of tensors, e.g.
    gt_mask = mask_list[count].clamp_(max=1)          # DecSeg/val.py, update_metrics_seg
    ops.scale_boxes(..., predn[:, :4], ...)           # in-place -=, /= and clamp_ on a column slice
On the MPS backend these in-place view updates give results that differ from CPU.
"""
import torch

print('torch', torch.__version__)
for dev in ('cpu', 'mps'):
    # (1) in-place clamp_ on a row view with non-zero storage offset (like mask_list[count])
    m = torch.zeros(3, 4, 5, device=dev)
    m[1] = 2.0
    m[2] = 3.0
    s_before = [float(m[i].sum()) for i in range(3)]
    outs = [float(m[i].clamp_(max=1).sum()) for i in range(3)]
    # (2) in-place arithmetic on a column slice (like predn[:, :4] in scale_boxes)
    b = torch.arange(24, dtype=torch.float32, device=dev).reshape(4, 6)
    v = b[:, :4]
    v[..., [0, 2]] -= 1.0
    v[..., :4] /= 2.0
    # (3) in-place clamp_ on a column view (like ops.clip_boxes: boxes[..., 0].clamp_(0, w))
    c = torch.tensor([[-5., 1., 700., 2.], [3., -4., 9., 800.]], device=dev)
    for k, lim in ((0, 672), (1, 384), (2, 672), (3, 384)):
        c[..., k].clamp_(0, lim)
    print(dev, '| row-view clamp_ sums', outs, '(expected [0, 20, 20]) | column-slice arithmetic', b.cpu().sum().item(),
          '| column-view clamp_', c.cpu().tolist(), '(expected [[0,1,672,2],[3,0,9,384]])')
