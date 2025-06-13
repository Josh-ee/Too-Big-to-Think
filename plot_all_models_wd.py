#!/usr/bin/env python3
import os
import argparse
import re
import pandas as pd
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import MultipleLocator, FuncFormatter
from model import GPTConfig, GPT

# 1) Bold all text by default
mpl.rcParams.update({
    'font.weight':      'bold',
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
})

# Define model folders per task
task_groups = {
    'math': [
        'math_mini_14_wd_final', 'math_mini_28_wd_final',
        'math_mini_56_wd_final', 'math_mlt_wd_final'
    ],
    'facts': [
        'facts_mini_14_wd_final', 'facts_mini_28_wd_final',
        'facts_mini_56_wd_final', 'facts_mlt_wd_final'
    ],
    'combined': [
        'both_mini_14_wd_final', 'both_mini_28_wd_final',
        'both_mini_56_wd_final', 'both_mlt_wd_final'
    ]
}

# Titles for each group
group_titles = {
    'math':    'Arithmetic Extrapolation: Held-Out (5,7)',
    'facts':   'Factual Memorization: Capital City Recall',
    'combined':'Joint Arithmetic & Factual Recall'
}

# Define legend filters per task
legend_filters = {
    'math': {'Addition', 'Subtraction', 'Held-out (5,7) Correct', 'Train Loss'},
    'facts': {'Facts', 'Train Loss'},
    'combined': None  # None means include all
}

def plot_group(root_dir, group_name, folders, output):
    fig, axes = plt.subplots(2, 2, figsize=(9, 8), sharex=True)
    # make overall title bold
    # fig.suptitle(group_titles.get(group_name, '') + '\nControlled Regularization (dropout = 0.0 and weight decay = 0.1)', fontsize=16, fontweight='bold')
    fig.suptitle(
    group_titles.get(group_name, '') + '\nControlled Regularization (dropout = 0.0 and weight decay = 0.1)',
    fontsize=16,
    fontweight='bold'
)
    axes = axes.flatten()

    for ax, model_folder in zip(axes, folders):
        path = os.path.join(root_dir, model_folder)
        # load checkpoint
        ckpt = torch.load(os.path.join(path, 'ckpt.pt'), map_location='cpu')
        cfg = GPTConfig(**ckpt['model_args'])
        model = GPT(cfg)
        sd = { (k[len('_orig_mod.'):] if k.startswith('_orig_mod.') else k): v
               for k, v in ckpt['model'].items() }
        model.load_state_dict(sd)
        n_params = model.get_num_params()
        # lowercase 'k' for thousands
        size_str = f"{n_params/1e6:.2f}M" if n_params >= 1e6 else f"{n_params/1e3:.1f}k"

        # load results
        df = pd.read_csv(os.path.join(path, 'results.csv'))
        iters = df['iter']
        add_acc = df['add_correct'] / df['add_total']
        sub_acc = df['sub_correct'] / df['sub_total']
        facts_acc = df.get('facts_acc')
        train_loss = df.get('train_loss')

        # 1) Overlay training loss
        ax2 = ax.twinx()
        ax2.plot(iters, train_loss,
                 linestyle='-',
                 label='Train Loss',
                 color='brown',
                 zorder=1)
        ax2.set_ylabel('Train Loss', fontweight='bold')
        ax2.set_ylim(0, 4.5)
        ax2.yaxis.set_major_locator(MultipleLocator(1))
        ax2.set_zorder(1)
        ax.set_zorder(1)

        # 2) Plot facts
        if facts_acc is not None and not facts_acc.eq(0).all():
            ax.plot(iters, facts_acc,
                    label='Facts',
                    color='orange',
                    zorder=2)

        # 3) Addition
        ax.plot(iters, add_acc,
                label='Addition',
                color='green',
                zorder=3)

        # 4) Subtraction
        ax.plot(iters, sub_acc,
                label='Subtraction',
                linestyle=':',
                color='blue',
                zorder=4)

        # 5) Scatter 100% Addition & Subtraction
        perfect = (add_acc == 1.0) & (sub_acc == 1.0)
        if perfect.any():
            ax.scatter(iters[perfect], add_acc[perfect],
                       color='red', marker='o', s=10,
                       label='Held-out (5,7) Correct',
                       zorder=5)

        # y-axis formatting
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.set_ylim(0, 1.05)

        # x-axis formatting: use 5k, 10k, ...
        ax.xaxis.set_major_locator(MultipleLocator(5000))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{int(x/1000)}k'))

        # make tick‐labels bold explicitly
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontweight('bold')

        ax.grid(True)

        # short title
        m = re.search(r'_mini_(\d+)_wd_final$', model_folder)
        if m:
            short = f"n{m.group(1)}"
        elif '_mlt_' in model_folder:
            short = 'MLT'
        else:
            short = model_folder
        ax.set_title(f"{short} ({size_str} Parameters)")

    # only bottom row gets xlabel
    for a in axes[2:]:
        a.set_xlabel('Iteration', fontweight='bold')

    # collect legend entries
    handles, labels = [], []
    for axis in fig.axes:
        h, l = axis.get_legend_handles_labels()
        handles += h
        labels  += l

    seen = {}
    desired = legend_filters[group_name]
    for h, l in zip(handles, labels):
        if desired is None or l in desired:
            if l not in seen:
                seen[l] = h

    # legend in bold
    fig.legend(
        seen.values(), seen.keys(),
        loc='lower center', ncol=4,
        prop={'size': '13', 'weight': 'bold'}
    )

    plt.tight_layout(rect=[0, 0.07, 1, 0.96], pad=0, w_pad=0, h_pad=0)
    out_path = os.path.join(root_dir, output)
    plt.savefig(out_path, dpi=300)
    print(f"Saved {group_name} grid to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='.',
                        help='Project root containing model folders')
    args = parser.parse_args()
    for task, dirs in task_groups.items():
        output = f'grid_{task}_wd.png'
        plot_group(args.root_dir, task, dirs, output)


if __name__ == '__main__':
    main()
