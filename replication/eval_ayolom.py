"""Replication of the A-YOLOM evaluation on the BDD100K validation split.

Runs the official validator of the YOLOv8-multi-task repository
(https://github.com/JiayuanWang-JW/YOLOv8-multi-task) with exactly the
settings of the repository's ``ultralytics/val.py``:

    iou=0.6, conf=0.001, imgsz=(640, 640), classes=[2, 3, 4, 9, 10, 11],
    combine_class=[2, 3, 4, 9], single_cls=True, batch=16, rect=True

The official code reports segmentation metrics averaged *per image*. In
addition, this script accumulates one confusion matrix over the whole split
per segmentation task, so that dataset-level IoU / accuracy can be reported
next to the official numbers. The official code path is not modified.

Usage:
    python replication/eval_ayolom.py --weights data/weights/v4.pt --name ayolom_n
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / 'YOLOv8-multi-task'
sys.path.insert(0, str(REPO))

from ultralytics import YOLO, __version__ as ul_version  # noqa: E402
from ultralytics.yolo.cfg import get_cfg  # noqa: E402
from ultralytics.yolo.utils import DEFAULT_CFG  # noqa: E402
from ultralytics.yolo.utils.checks import check_imgsz  # noqa: E402
from ultralytics.yolo.v8.DecSeg.val import MultiValidator  # noqa: E402

# ---------------------------------------------------------------------------
# Extra bookkeeping: dataset-level confusion matrices for segmentation tasks.
# ---------------------------------------------------------------------------
GLOBAL_CM = {}
_orig_update_seg = MultiValidator.update_metrics_seg


def _update_seg_with_global_cm(self, preds, batch, task_name=None):
    _orig_update_seg(self, preds, batch, task_name)  # official per-image metrics
    cm = GLOBAL_CM.setdefault(task_name, np.zeros((2, 2), dtype=np.int64))
    gts = batch['masks'].to(self.device).float()
    for i in range(len(batch['im_file'])):
        gt = gts[i].clamp(max=1).long().flatten().cpu().numpy()
        pr = preds[i].squeeze().long().flatten().cpu().numpy()
        valid = (gt >= 0) & (gt < 2)
        cm += np.bincount(2 * gt[valid] + pr[valid], minlength=4).reshape(2, 2)


MultiValidator.update_metrics_seg = _update_seg_with_global_cm


def cm_metrics(cm):
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    iou_fg = tp / (tp + fp + fn)
    iou_bg = tn / (tn + fn + fp)
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    return {
        'IoU_fg': float(iou_fg),
        'IoU_bg': float(iou_bg),
        'mIoU': float((iou_fg + iou_bg) / 2),
        'pixel_acc': float((tp + tn) / cm.sum()),
        'balanced_acc': float((sens + spec) / 2),
        'precision_fg': float(tp / (tp + fp)),
        'recall_fg': float(sens),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--data-root', default=str(ROOT / 'data' / 'bdd'),
                    help='folder containing images/val2017, detection-object, seg-drivable-10, seg-lane-11')
    ap.add_argument('--device', default='cpu',
                    help="'cpu' (default, used for all reported accuracy numbers) or '0' for CUDA. Do NOT use 'mps' "
                         "with PyTorch 2.2.2: in-place clamp_ on tensor views is ignored there and the metrics are wrong "
                         "(see replication/mps_inplace_view_repro.py)")
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--workers', type=int, default=8, help='dataloader workers (official default 8; the validator forces 0 on CPU)')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--name', required=True)
    ap.add_argument('--out', default=str(ROOT / 'replication' / 'results'))
    a = ap.parse_args()

    # The official dataset yaml with only the (absolute) dataset root replaced.
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    yaml_txt = (REPO / 'ultralytics' / 'datasets' / 'bdd-multi.yaml').read_text()
    yaml_txt = '\n'.join(f'path: {Path(a.data_root).resolve()}' if l.startswith('path:') else l
                          for l in yaml_txt.splitlines() if not l.startswith('#path'))
    data_yaml = out / 'bdd-multi-resolved.yaml'
    data_yaml.write_text(yaml_txt + '\n')

    torch.manual_seed(0)
    np.random.seed(0)

    model = YOLO(a.weights)
    # identical to YOLO.val(): rect validation, mode=val, task=multi
    overrides = model.overrides.copy()
    overrides.update(dict(rect=True, data=str(data_yaml), device=a.device, task='multi', name=a.name, iou=0.6,
                          conf=0.001, imgsz=(a.imgsz, a.imgsz), classes=[2, 3, 4, 9, 10, 11],
                          combine_class=[2, 3, 4, 9], single_cls=True, batch=a.batch, workers=a.workers, mode='val',
                          project=str(Path(a.out) / 'runs'), exist_ok=True))
    args = get_cfg(cfg=DEFAULT_CFG, overrides=overrides)
    args.imgsz = check_imgsz(args.imgsz, max_dim=1)

    validator = MultiValidator(args=args, _callbacks=model.callbacks)
    t0 = time.time()
    validator(model=model.model)
    wall = time.time() - t0

    labels = validator.data['labels_list']
    res = {'weights': a.weights, 'device': a.device, 'batch': a.batch, 'workers': a.workers, 'imgsz': a.imgsz,
           'wall_time_s': wall, 'n_params': int(sum(p.numel() for p in model.model.parameters())),
           'speed_ms_per_image_per_task': validator.speed, 'tasks': {}}
    for i, name in enumerate(labels):
        if 'det' in name:
            m = validator.metrics[i]
            p, r, map50, map5095 = m.mean_results()
            res['tasks'][name] = {'images': validator.seen[name],
                                  'instances': int(validator.nt_per_class[name].sum()),
                                  'precision': float(p), 'recall': float(r),
                                  'mAP50': float(map50), 'mAP50-95': float(map5095)}
        else:
            official = {k: float(v.avg) for k, v in validator.seg_result[name].items()}
            official['n_images'] = int(validator.seg_result[name]['IoU'].count)
            res['tasks'][name] = {'official_per_image_mean': official,
                                  'dataset_level': cm_metrics(GLOBAL_CM[name]),
                                  'confusion_matrix[[TN,FP],[FN,TP]]': GLOBAL_CM[name].tolist()}

    res['env'] = {'python': platform.python_version(), 'torch': torch.__version__,
                  'numpy': np.__version__, 'ultralytics_fork': ul_version,
                  'platform': platform.platform(), 'machine': platform.machine()}
    (out / f'{a.name}.json').write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
