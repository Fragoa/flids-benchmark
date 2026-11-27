from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
with (root.parent / 'reference' / 'transfer_runs.csv').open() as stream:
    runs = [r for r in csv.DictReader(stream) if r['method'] == 'FedAvg']
domains = sorted({r['target'] for r in runs})
fig, axes = plt.subplots(1, 2, figsize=(7, 2.6), sharey=True)
for ax, metric, label in zip(axes, ['target_fpr', 'target_recall'], ['False-positive rate (%)', 'Attack recall (%)']):
    for y, domain in enumerate(domains):
        rates = [100 * float(r[metric]) for r in runs if r['target'] == domain]
        ax.scatter(rates, y + np.linspace(-.15, .15, len(rates)), s=20, color='#176783')
        ax.plot(np.mean(rates), y, marker='D', markerfacecolor='white', markeredgecolor='black', markersize=6)
    ax.set_xlabel(label)
    ax.set_xlim(left=-1)
    ax.grid(axis='x', alpha=.22)
axes[0].axvline(1, ls='--', lw=1, color='#a14020')
axes[0].set_yticks(range(4), ['BoT-IoT', 'CSE-CIC-IDS2018', 'ToN-IoT', 'UNSW-NB15'])
axes[0].invert_yaxis()
fig.tight_layout()
Path('outputs').mkdir(exist_ok=True)
fig.savefig(Path('outputs') / 'threshold_transfer.pdf', bbox_inches='tight')
