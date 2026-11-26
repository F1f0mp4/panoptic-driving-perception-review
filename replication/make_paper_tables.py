"""Builds the replication table of the article directly from the result JSON files."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'replication' / 'results'
OUT = ROOT / 'paper' / 'generated'
OUT.mkdir(parents=True, exist_ok=True)

# Values published in Wang et al., IEEE T-VT 2024 (arXiv v4, Tables I-IV) and in the repository README.
ORIG = {
    'n': {'recall': 85.3, 'map50': 78.0, 'da_miou': 90.5, 'll_acc': 81.3, 'll_iou': 28.2, 'params': 4.43,
          'fps1': 39.9, 'fps32': 172.2},
    's': {'recall': 86.9, 'map50': 81.1, 'da_miou': 91.0, 'll_acc': 84.9, 'll_iou': 28.8, 'params': 13.61,
          'fps1': 39.7, 'fps32': 96.2},
}


WEIGHTS = {'n': ROOT / 'data' / 'weights' / 'v4.pt', 's': ROOT / 'data' / 'weights' / 'v4s.pt'}


def params_unfused(scale):
    """Parameter count of the released checkpoint before Conv+BN fusion (the convention of the paper's Table I)."""
    import sys
    import torch
    sys.path.insert(0, str(ROOT / 'YOLOv8-multi-task'))
    m = torch.load(WEIGHTS[scale], map_location='cpu')['model']
    return sum(p.numel() for p in m.parameters())


def sd(v):
    """Signed difference with two decimals; exact zero after rounding is printed without a sign."""
    return '0.00' if abs(v) < 0.005 else f'{v:+.2f}'


def load(scale, prefix=''):
    ev = json.loads((RES / f'{prefix}ayolom_{scale}.json').read_text())
    fps = json.loads((RES / f'fps_ayolom_{scale}_mps.json').read_text())
    fpsc = json.loads((RES / f'fps_ayolom_{scale}_cpu.json').read_text())
    det = ev['tasks']['detection-object']
    da = ev['tasks']['seg-drivable-10']
    ll = ev['tasks']['seg-lane-11']
    return {
        'recall': 100 * det['recall'], 'map50': 100 * det['mAP50'], 'map5095': 100 * det['mAP50-95'],
        'precision': 100 * det['precision'],
        'da_miou': 100 * da['official_per_image_mean']['mIoU'], 'll_acc': 100 * ll['official_per_image_mean']['subacc'],
        'll_iou': 100 * ll['official_per_image_mean']['IoU'],
        'da_miou_ds': 100 * da['dataset_level']['mIoU'], 'll_acc_ds': 100 * ll['dataset_level']['balanced_acc'],
        'll_iou_ds': 100 * ll['dataset_level']['IoU_fg'],
        'params': params_unfused(scale) / 1e6, 'params_fused': ev['n_params'] / 1e6, 'fps1': fps['bs1']['fps'], 'fps32': fps['bs32']['fps'],
        'fps1_cpu': fpsc['bs1']['fps'],
        'n_img': det['images'], 'n_inst': det['instances'], 'n_seg': da['official_per_image_mean']['n_images'],
        'wall': ev['wall_time_s'], 'env': ev['env'],
    }


def main():
    R = {s: load(s) for s in ('n', 's')}
    M = load('n', prefix='mps_')  # first (defective) run on the MPS backend
    rows = [('Vehicle recall [\\%]', 'recall'), ('Vehicle mAP@0.5 [\\%]', 'map50'),
            ('Drivable-area mIoU [\\%]', 'da_miou'), ('Lane balanced accuracy [\\%]', 'll_acc'),
            ('Lane IoU [\\%]', 'll_iou')]
    lines = []
    for name, k in rows:
        cells = []
        for s in ('n', 's'):
            o, r = ORIG[s][k], R[s][k]
            cells += [f'{o:.1f}', f'{r:.2f}', sd(r - o)]
        lines.append(name + ' & ' + ' & '.join(cells) + ' \\\\')
    lines.append('\\midrule')
    cells = []
    for s in ('n', 's'):
        o, r = ORIG[s]['params'], R[s]['params']
        cells += [f'{o:.2f}', f'{r:.2f}', sd(r - o)]
    lines.append('Parameters [M] & ' + ' & '.join(cells) + ' \\\\')
    for lab, k in (('FPS, batch 1$^\\dagger$', 'fps1'), ('FPS, batch 32$^\\dagger$', 'fps32')):
        cells = []
        for s in ('n', 's'):
            cells += [f'{ORIG[s][k]:.1f}', f'{R[s][k]:.1f}', 'n/a']
        lines.append(lab + ' & ' + ' & '.join(cells) + ' \\\\')
    lines.append('FPS, batch 1, CPU$^\\ddagger$ & -- & ' + f"{R['n']['fps1_cpu']:.1f} & n/a & -- & {R['s']['fps1_cpu']:.1f} & n/a" + ' \\\\')

    t = r"""\begin{table}[t]
\centering
\caption{Original versus replicated results of A-YOLOM on the BDD100K validation set (10{,}000 images). Original values: Tables~I--IV of~\cite{wang2024ayolom}. Replication: released weights, official evaluation code on the Apple M3~Pro CPU. Difference = replication $-$ original (pp for metrics). Parameters counted before Conv+BN fusion. $^\dagger$Original FPS on a GTX~1080~Ti, replication on Apple M3~Pro GPU (MPS, FP32) -- not comparable. $^\ddagger$Apple M3~Pro CPU, FP32, 100 passes.}
\label{tab:repl}
\small
\begin{tabular}{l rrr rrr}
\toprule
 & \multicolumn{3}{c}{A-YOLOM(n)} & \multicolumn{3}{c}{A-YOLOM(s)} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
Metric & Orig. & Repl. & Diff. & Orig. & Repl. & Diff. \\
\midrule
""" + '\n'.join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}

\begin{table}[t]
\centering
\caption{Effect of the aggregation scheme on the segmentation metrics (replication, \%). The official code averages per-image values; ``dataset-level'' uses one confusion matrix accumulated over all 10{,}000 images.}
\label{tab:agg}
\small
\begin{tabular}{l rr rr}
\toprule
 & \multicolumn{2}{c}{A-YOLOM(n)} & \multicolumn{2}{c}{A-YOLOM(s)} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Metric & per-image & dataset & per-image & dataset \\
\midrule
""" + '\n'.join(
        f"{name} & {R['n'][k]:.2f} & {R['n'][k + '_ds']:.2f} & {R['s'][k]:.2f} & {R['s'][k + '_ds']:.2f} \\\\"
        for name, k in (('Drivable-area mIoU', 'da_miou'), ('Lane balanced accuracy', 'll_acc'), ('Lane IoU', 'll_iou'))
    ) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    mrows = [('Vehicle recall', 'recall'), ('Vehicle mAP@0.5', 'map50'), ('Drivable-area mIoU', 'da_miou'),
             ('Lane balanced accuracy', 'll_acc'), ('Lane IoU (per-image)', 'll_iou'), ('Lane IoU (dataset-level)', 'll_iou_ds')]
    t += r"""
\begin{table}[t]
\centering
\caption{A-YOLOM(n): identical code, weights and data evaluated with the PyTorch~2.2.2 MPS (Apple GPU) backend versus the CPU (\%). We attribute the difference to in-place \texttt{clamp\_} operations on tensor views, which have no effect on MPS (see text).}
\label{tab:mps}
\small
\begin{tabular}{l rrr}
\toprule
Metric & CPU & MPS & Diff. \\
\midrule
""" + '\n'.join(f"{n} & {R['n'][k]:.2f} & {M[k]:.2f} & {sd(M[k] - R['n'][k])} \\\\" for n, k in mrows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    (OUT / 'repl_table.tex').write_text(t)
    (OUT / 'repl_numbers.json').write_text(json.dumps({'cpu': R, 'mps_n': M}, indent=2))
    write_macros(R, M)
    print(json.dumps({s: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in R[s].items() if k != 'env'}
                      for s in R}, indent=1))


def write_macros(R, M):
    """Writes the result-dependent sentences of the article (paper/generated/repl_macros.tex)."""
    n, s_ = R['n'], R['s']
    keys = ('recall', 'map50', 'da_miou', 'll_acc', 'll_iou')
    maxdiff = max(abs(R[sc][k] - ORIG[sc][k]) for sc in R for k in keys)
    aggd = [R[sc][k + '_ds'] - R[sc][k] for sc in R for k in ('da_miou', 'll_acc', 'll_iou')]
    env = n['env']
    f = lambda v, d=2: f'{v:.{d}f}'
    m = {}
    m['ReplAbstractSentence'] = (
        f"All ten accuracy values of both released models were reproduced within {f(maxdiff)} percentage points "
        f"(vehicle mAP@0.5 {f(n['map50'])}\\% vs.\\ {ORIG['n']['map50']:.1f}\\%); the reported parameter count of "
        f"A-YOLOM(n) was not ({f(n['params'])} vs.\\ {ORIG['n']['params']:.2f}\\,M), and Apple's MPS backend silently "
        f"produced wrong metrics.")
    m['ReplDataNote'] = (
        "The repository README swaps the download links of the drivable-area and lane annotations (the archive linked as "
        "\\texttt{seg-drivable-10} contains the lane labels, class~11, and vice versa); we identified the archives by "
        "their class identifiers. The validation split contains " + f"{n['n_inst']:,}".replace(',', '{,}') +
        " vehicle instances (car, bus, truck and train merged into one class, as in the original evaluation).")
    m['ReplEnvironment'] = (
        "Apple M3~Pro (11-core CPU with 5 performance and 6 efficiency cores, 14-core GPU), 18\\,GB unified memory, "
        f"macOS~27.2.1; Python~{env['python']}, PyTorch~{env['torch']}, torchvision~0.17.2, NumPy~{env['numpy']}, "
        f"OpenCV~4.9.0.80 and the ultralytics~{env['ultralytics_fork']} fork contained in the repository (full list in "
        "\\path{replication/requirements-lock.txt}). The official validator uses no dataloader worker processes on the "
        f"CPU. One full evaluation took {n['wall'] / 60:.1f}\\,min for A-YOLOM(n) and {s_['wall'] / 60:.1f}\\,min for "
        "A-YOLOM(s).")
    ratio1 = n['fps1'] / s_['fps1']
    ratio32 = n['fps32'] / s_['fps32']
    m['ReplDiscussion'] = (
        f"\\textbf{{Accuracy.}} All ten accuracy values in Table~\\ref{{tab:repl}} agree with the published ones within "
        f"$\\pm${f(maxdiff)}\\,pp. The paper reports one decimal, so every difference is within the rounding interval "
        "($\\pm$0.05\\,pp): the published evaluation of the released checkpoints is reproduced exactly, even with a newer "
        "PyTorch version (2.2.2 instead of 1.13.1) and on a CPU instead of an NVIDIA GPU. For completeness, the "
        f"vehicle mAP@0.5:0.95 is {f(n['map5095'])}\\% for A-YOLOM(n) and {f(s_['map5095'])}\\% for A-YOLOM(s).\n\n"
        "\\textbf{Aggregation of segmentation metrics.} The official code computes IoU and accuracy for every image and "
        "averages them. Accumulating one confusion matrix over the whole validation set gives values that are "
        f"{min(aggd):.1f}--{max(aggd):.1f}\\,pp higher (Table~\\ref{{tab:agg}}), because images with few lane or "
        "drivable-area pixels, where a few wrong pixels cost much, count as much as images with many such pixels. "
        "Neither choice is wrong, but the difference is larger than most improvements claimed in "
        "Table~\\ref{tab:comparison}, and the aggregation is a detail of the evaluation code that readers cannot see "
        "unless the code is released. This gives a measured example of the comparability problem identified in RQ2.\n\n"
        f"\\textbf{{Parameters.}} For A-YOLOM(s), the parameter count of the released checkpoint ({f(s_['params'])}\\,M "
        "before Conv+BN fusion) matches the paper. For A-YOLOM(n), both the checkpoint and the model built from the "
        f"repository's configuration file have {f(n['params'])}\\,M parameters, not {ORIG['n']['params']:.2f}\\,M as "
        "printed in the paper and README. The accuracy of this checkpoint matches the paper, so we consider the "
        "published parameter count a reporting error; its cause cannot be determined from the released material. "
        "After fusion, as used at inference, the models have "
        f"{f(n['params_fused'])}\\,M and {f(s_['params_fused'])}\\,M parameters.\n\n"
        f"\\textbf{{Speed.}} On the M3~Pro GPU, A-YOLOM(n) reaches {n['fps1']:.1f}\\,FPS at batch size~1 and "
        f"{n['fps32']:.1f}\\,FPS at batch size~32; A-YOLOM(s) reaches {s_['fps1']:.1f} and {s_['fps32']:.1f}\\,FPS. "
        f"On the CPU (batch size~1) the values are {n['fps1_cpu']:.1f} and {s_['fps1_cpu']:.1f}\\,FPS. Absolute "
        "values cannot be compared with the GTX~1080~Ti numbers of the paper, but the relative speed of the two models can: "
        f"the paper reports a ratio n/s of {ORIG['n']['fps1'] / ORIG['s']['fps1']:.2f} at batch size~1 and "
        f"{ORIG['n']['fps32'] / ORIG['s']['fps32']:.2f} at batch size~32; we measure {ratio1:.2f} and {ratio32:.2f}. "
        "Both measurements time only the network forward pass, without pre-processing, NMS or mask post-processing, so "
        "they overestimate the end-to-end frame rate.\n\n"
        "\\textbf{Wrong metrics on the MPS backend.} Our first evaluation of A-YOLOM(n), with identical code, weights "
        f"and data on the Apple GPU (MPS backend), gave a vehicle mAP@0.5 of {f(M['map50'])}\\% instead of "
        f"{f(n['map50'])}\\% (Table~\\ref{{tab:mps}}). A layer-wise comparison of the forward pass on one validation "
        "image showed no relevant difference between CPU and MPS (maximum relative difference $<3\\cdot10^{-6}$). A "
        "minimal test (\\path{replication/mps_inplace_view_repro.py}) shows that in PyTorch~2.2.2 an in-place "
        "\\texttt{clamp\\_} applied to a view of a tensor has no effect on MPS. The official post-processing and "
        "metric code use exactly this pattern: \\texttt{clip\\_boxes} clips predicted boxes to the image with "
        "\\texttt{boxes[..., 0].clamp\\_(0, w)}, and \\texttt{update\\_metrics\\_seg} binarises masks with "
        "\\texttt{mask\\_list[i].clamp\\_(max=1)}. We therefore attribute the discrepancy to this backend defect. "
        "No error or warning is raised, so without the published numbers as a reference this run could easily have been "
        "reported as a failed replication. Replication reports should therefore state the compute backend and, where "
        "possible, verify metrics on a second device.")
    m['ReplConclusion'] = (
        f"The replication confirms the published accuracy of A-YOLOM within {f(maxdiff)}\\,pp for all ten metrics of "
        "both released models, but not the published parameter count of A-YOLOM(n), and it shows that the reported "
        "numbers also depend on how segmentation metrics are aggregated and on a correctly working compute backend")
    (OUT / 'repl_macros.tex').write_text(''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in m.items()))


if __name__ == '__main__':
    main()
