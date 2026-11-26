"""Generates the literature-review figures of the article from included_studies.csv."""
import csv
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / 'paper' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
rows = list(csv.DictReader(open(HERE / 'included_studies.csv')))

GROUPS = {'YOLO': 'YOLO-derived encoder-decoder', 'LW': 'Ultra-lightweight / segmentation-centric',
          'ATT': 'Transformer / query / prompt-based'}
COLORS = {'YOLO': '#2a6fdb', 'LW': '#e07b00', 'ATT': '#2e9d5b'}
plt.rcParams.update({'font.size': 9, 'font.family': 'serif'})

# ---- Fig. 2: included studies per year and design family (one entry per study key) ----
studies = {}
for r in rows:
    studies.setdefault(r['key'], r)
years = sorted({int(r['year']) for r in studies.values()})
fig, ax = plt.subplots(figsize=(4.6, 2.6))
bottom = [0] * len(years)
for g in GROUPS:
    cnt = Counter(int(r['year']) for r in studies.values() if r['group'] == g)
    vals = [cnt.get(y, 0) for y in years]
    ax.bar(years, vals, bottom=bottom, color=COLORS[g], label=GROUPS[g], width=0.6)
    bottom = [b + v for b, v in zip(bottom, vals)]
for y, b in zip(years, bottom):
    ax.text(y, b + 0.1, str(b), ha='center', va='bottom', fontsize=8)
ax.set_xticks(years)
ax.set_ylabel('Included studies')
ax.set_ylim(0, max(bottom) + 2)
ax.legend(frameon=False, fontsize=7, loc='upper left')
ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / 'fig_years.pdf')

# ---- Fig. 3: accuracy vs model size, and lane IoU by evaluation protocol ----
def num(x):
    try:
        return float(x.replace('*', ''))
    except (ValueError, AttributeError):
        return None

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
ax = axes[0]
for r in rows:
    p, m = num(r['params_M']), num(r['da_miou'])
    if p is None or m is None or r['key'] == 'ghasemzadeh2026auriganet':  # AurigaNet reports IoU, not mIoU
        continue
    ax.scatter(p, m, color=COLORS[r['group']], s=22, zorder=3)
    if r['key'] in {'han2022yolopv2', 'zhan2024yolopv3', 'wang2026rmtppad', 'che2025twinliteplus', 'kdtwin2026',
                    'che2023twinlitenet', 'trilitenet2025', 'vu2025hybridnets', 'ye2023mobip', 'li2023multinetgs',
                    'caldiran2022dynasticnet'} or r['label'].startswith('A-YOLOM'):
        ax.annotate(r['label'], (p, m), fontsize=6, xytext=(3, 2), textcoords='offset points')
ax.set_xscale('log')
ax.set_xlabel('Parameters [M] (log scale)')
ax.set_ylabel('Drivable-area mIoU [%]')
ax.set_title('(a) Drivable-area mIoU vs. model size', fontsize=9)
ax.grid(alpha=0.3)

ax = axes[1]
import random
random.seed(0)
mk = {'std': ('o', '#555555', 'YOLOP protocol (2-px test lines)'),
      'corrected': ('s', '#c0392b', 'Corrected 8-px test label [RMT-PPAD]'),
      'nonstd': ('^', '#8e44ad', 'Other / unspecified protocol')}
LABEL = {'wang2026rmtppad', 'compress2025', 'li2023multinetgs', 'ghasemzadeh2026auriganet', 'caldiran2022dynasticnet'}
for r in rows:
    v = num(r['ll_iou'])
    if v is None:
        continue
    kind = 'corrected' if r['key'] in {'wang2026rmtppad', 'compress2025'} else (
        'nonstd' if r['key'] in {'li2023multinetgs', 'ghasemzadeh2026auriganet'} else 'std')
    y = random.uniform(-0.35, 0.35)
    m, c, _ = mk[kind]
    ax.scatter(v, y, marker=m, color=c, s=22, zorder=3, alpha=0.85)
    if r['key'] in LABEL or r['label'] == 'A-YOLOM(s)':
        ax.annotate(r['label'], (v, y), fontsize=6, xytext=(2, 5), textcoords='offset points')
for kind, (m, c, l) in mk.items():
    ax.scatter([], [], marker=m, color=c, label=l, s=20)
ax.set_yticks([])
ax.set_ylim(-0.6, 1.3)
ax.set_xlim(10, 85)
ax.set_xlabel('Reported lane-line IoU [%]')
ax.set_title('(b) Reported lane IoU by evaluation protocol', fontsize=9)
ax.legend(frameon=False, fontsize=6, loc='upper right')
ax.grid(axis='x', alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / 'fig_tradeoff.pdf')
print('saved to', OUT)
