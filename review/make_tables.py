"""Generates the LaTeX comparison tables of the article from included_studies.csv."""
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / 'paper' / 'generated'
OUT.mkdir(parents=True, exist_ok=True)
rows = list(csv.DictReader(open(HERE / 'included_studies.csv')))

FAM = {'YOLO': 'A', 'LW': 'B', 'ATT': 'C'}
ORDER = {'YOLO': 0, 'LW': 1, 'ATT': 2}
# Short qualitative assessment per study (from the extracted full texts).
SL = {
    'caldiran2022dynasticnet': ('fastest multi-net of 2022 in its comparison', 'detection only as box-mask IoU; low lane IoU'),
    'kim2022joint': ('multilevel decoder branches, +1.2\\,pp DA vs.\\ YOLOP', 'detection slightly below YOLOP; no code'),
    'wu2022yolop': ('defined the task and protocol; Jetson TX2 23\\,FPS', 'anchor-based; lane metric = recall in code'),
    'han2022yolopv2': ('strong all-round accuracy at 91\\,FPS (V100)', '38.9\\,M params; code not released'),
    'ma2023grouped': ('grouped conv.\\ + attention, fewer params than baseline', 'mAP below YOLOP; no code'),
    'ye2023mobip': ('MobileNetV2, one multi-class seg.\\ head', 'detection below YOLOP'),
    'zeng2023dpyolo': ('110\\,FPS on T4, fewer FLOPs than YOLOP', 'gains $\\le$0.7\\,pp; no code'),
    'li2023multinetgs': ('highest mAP in its table; Jetson TX2 test', 'baseline numbers inconsistent with originals'),
    'guo2023yoloodl': ('high recall; TensorRT 477\\,FPS', 'no code; lane IoU near YOLOP'),
    'liang2023veprompt': ('partial-label MTL, multi-class detection', 'large gap to single-task models'),
    'che2023twinlitenet': ('0.4\\,M params, 415\\,FPS; Jetson deployment', 'no detection; lane IoU only'),
    'tang2024yolopdn': ('dual neck; deployed on Atlas 200I A2', 'slower than YOLOP; no code'),
    'liu2024fusion': ('LiDAR--camera fusion, KITTI test', 'results only as image tables'),
    'lin2024tyolop': ('Taguchi design of head widths', 'detection not evaluated'),
    'liu2024gdmnet': ('GELAN encoder, MPDIoU loss', 'two different detection results reported'),
    'li2024envperception': ('MogaNet + BiFPN, good mAP', 'no code; params not reported'),
    'zhan2024yolopv3': ('best recall/mAP of family A; code', '30.2\\,M params, 37\\,FPS'),
    'wang2024ayolom': ('adaptive concat.; 7.9k-param seg.\\ head; code+weights', 'recall below YOLOP; FPS on GTX 1080 Ti'),
    'che2025twinliteplus': ('Pareto-optimal seg.; INT8; per-condition results', 'no detection'),
    'compress2025': ('$-$32.7\\% params with small loss', 'corrected lane protocol; no code'),
    'vu2025hybridnets': ('BiFPN, strong recall; code', 'slow (37\\,ms on V100)'),
    'mttppnet2025': ('decoupled features; railway transfer', 'no code'),
    'mtfenet2025': ('441\\,FPS (RTX 4090); SODA10M test', 'lane precision instead of accuracy'),
    'trilitenet2025': ('2.35\\,M params incl.\\ detection; Jetson', 'mAP 72.3\\% (below YOLOP)'),
    'ghasemzadeh2026auriganet': ('instance-level drivable area; Orin NX', 'non-standard metrics; low recall'),
    'dbanet2026': ('directional attention for adverse scenes', 'single-run; lane IoU below YOLOP'),
    'kdtwin2026': ('task-aware distillation for tiny models', 'no detection; no speed reported'),
    'wang2026rmtppad': ('RT-DETR + gated adapters; 84.9\\% mAP', 'corrected lane label; 34.3\\,M params'),
    'selvarajan2026safety': ('road anomalies; Indian road dataset', 'BDD100K used only secondarily'),
    'twinmixing2026': ('channel-shuffle mixing; Jetson', 'no detection'),
}


def v(x):
    return x if x not in ('', None) else '--'


def lab(r):
    return f"{r['label']}~\\cite{{{r['key']}}}"


rows.sort(key=lambda r: (ORDER[r['group']], int(r['year']), r['label']))

# ---- Table I: qualitative comparison ----
seen, lines = set(), []
for r in rows:
    if r['key'] in seen:
        continue
    seen.add(r['key'])
    s, l = SL[r['key']]
    lab_ = r['label'].replace('(n)', '') if r['key'] == 'wang2024ayolom' else r['label']
    head = {'NR': 'n.r.'}.get(r['det_head'], r['det_head'])
    bb = 'YOLOv8 (n/s)' if r['key'] == 'wang2024ayolom' else ('n.r.' if r['backbone'] == 'NR' else r['backbone'])
    lines.append(f"{lab_}~\\cite{{{r['key']}}} & {r['year']} & {FAM[r['group']]} & {bb}; {head} & {s} & {l} \\\\")
t1 = r"""\begin{table}[!htbp]
\centering
\caption{Qualitative comparison of the 30 included studies (A: YOLO-derived, B: ultra-lightweight / segmentation-centric, C: transformer / query / prompt-based). All use BDD100K; additional datasets are mentioned in the text.}
\label{tab:qual}
\scriptsize
\begin{tabularx}{\linewidth}{L{2.6cm} c c L{3.6cm} X X}
\toprule
Study & Year & Fam. & Encoder; detection head & Strengths & Limitations \\
\midrule
""" + '\n'.join(lines) + r"""
\bottomrule
\end{tabularx}
\end{table}
"""

# ---- Table II: quantitative BDD100K results ----
lines = []
for r in rows:
    acc = v(r['ll_acc'])
    d = {'balanced': 'b', 'pixel': 'p', 'recall': 'r', '(TP+TN)/all': 'g', 'precision': 'prec.', 'NR': '?',
         'n/a': ''}.get(r['lane_acc_def'], '?')
    if acc != '--' and d:
        acc = f"{acc}\\textsuperscript{{{d}}}"
    fps = v(r['fps'])
    hw = '' if r['fps_hw'] in ('NR', '') else f" ({r['fps_hw']})"
    lines.append(f"{lab(r)} & {FAM[r['group']]} & {v(r['params_M'])} & {v(r['det_recall'])} & {v(r['det_map50'])} & "
                 f"{v(r['da_miou'])} & {acc} & {v(r['ll_iou'])} & {fps}{hw} & {r['code']} \\\\")
t2 = r"""\begin{table}[!htbp]
\centering
\caption{BDD100K validation results as reported by each study (\%). Rec./mAP: vehicle recall and mAP@0.5; DA: drivable-area mIoU; LL: lane-line accuracy and IoU. Superscripts give the stated lane-accuracy definition: p = pixel accuracy, b = balanced accuracy, r = recall, g = global accuracy $(TP+TN)/N$, ? = not defined. Values are not directly comparable across rows (Section~\ref{sec:discussion}); RMT-PPAD and its compressed version use an 8-px lane test label, AurigaNet reports drivable IoU and mAP@0.5:0.95 instead. ``--'': not reported. $^*$Parameter count of YOLOP as reported by~\cite{vu2025hybridnets,wang2024ayolom}.}
\label{tab:comparison}
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabularx}{\linewidth}{L{3.3cm} c r r r r r r X c}
\toprule
Study & Fam. & Params [M] & Rec. & mAP & DA & LL Acc & LL IoU & Speed [FPS] (GPU) & Code \\
\midrule
""" + '\n'.join(lines).replace('7.9*', '7.9$^*$') + r"""
\bottomrule
\end{tabularx}
\end{table}
"""
(OUT / 'comparison_table.tex').write_text(t1 + '\n' + t2)
print('written', OUT / 'comparison_table.tex')
