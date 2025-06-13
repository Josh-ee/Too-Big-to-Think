#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import MultipleLocator
import torch
from model import GPTConfig, GPT
import numpy as np

# 1) Bold all text by default
mpl.rcParams.update({
    'font.weight':      'bold',
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
})

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_folder", type=str, required=True,
                        help="Directory containing 'results.csv' and checkpoint")
    parser.add_argument("--checkpoint", type=str, default="ckpt.pt",
                        help="Checkpoint filename to load (e.g. ckpt.pt)")
    parser.add_argument("--operation", type=str, choices=["math","facts","both"],
                        default="both",
                        help="Which curves to plot: math, facts, or both")
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    base = os.path.basename(os.path.normpath(args.model_folder))

    # --- load checkpoint and print model info
    cp_path = os.path.join(args.model_folder, args.checkpoint)
    ckpt = torch.load(cp_path, map_location='cpu')
    iter_num = ckpt.get('iter_num')
    best_val_loss = ckpt.get('best_val_loss')
    if hasattr(best_val_loss, 'item'):
        best_val_loss = best_val_loss.item()
    model_args = ckpt.get('model_args', {})
    print(f"iter_num: {iter_num}")
    print(f"best_val_loss: {best_val_loss}")
    print(f"model_args: {model_args}")

    # --- compute parameter count and format size_str
    cfg = GPTConfig(**model_args)
    model_ins = GPT(cfg)
    state_dict = ckpt['model']
    for k in list(state_dict):
        if k.startswith('_orig_mod.'):
            new_k = k[len('_orig_mod.') :]
            state_dict[new_k] = state_dict.pop(k)
    model_ins.load_state_dict(state_dict)
    param_count = model_ins.get_num_params()

    if param_count >= 1e6:
        size_str = f"{param_count/1e6:.2f}M"
    elif param_count >= 1e3:
        size_str = f"{param_count/1e3:.2f}K"
    else:
        size_str = str(param_count)

    # --- load results.csv
    csv_path = os.path.join(args.model_folder, 'results.csv')
    df = pd.read_csv(csv_path)

    # --- extract metrics
    iters = df['iter']
    add_acc = df['add_correct'] / df['add_total']
    sub_acc = df['sub_correct'] / df['sub_total']
    facts_accs = df.get('facts_acc')
    train_losses = df.get('train_loss')

    # --- determine which series to plot
    plot_math = args.operation in ("math", "both") and (add_acc.sum() > 0 and sub_acc.sum() > 0)
    plot_facts = args.operation in ("facts", "both") and facts_accs is not None and not facts_accs.eq(0).all()

    # --- plotting
    fig, ax1 = plt.subplots(figsize=(8,5))

    # set x-axis to log scale with labels 10^1, 10^2, ...
    ax1.set_xscale('log', base=10)
    ax1.xaxis.set_major_locator(mtick.LogLocator(base=10))
    ax1.xaxis.set_major_formatter(mtick.LogFormatterMathtext(base=10))
    ax1.xaxis.set_minor_formatter(mtick.NullFormatter())

    # plot math curves with explicit colors
    if plot_math:
        ax1.plot(iters, add_acc, '-', label="Addition", color='green', zorder=1)
        ax1.plot(iters, sub_acc, ':', label="Subtraction", color='blue', zorder=2)
        perfect_mask = (add_acc == 1.0) & (sub_acc == 1.0)
        if perfect_mask.any():
            ax1.scatter(iters[perfect_mask], add_acc[perfect_mask],
                        marker='o', s=10, label="100% A&S", color='red', zorder=5)

    # plot facts curve
    if plot_facts:
        ax1.plot(iters, facts_accs, '-', label="Facts", color='orange', zorder=3)

    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax1.set_ylim(0, 1.05)

    ax1.set_xlabel("Iteration", fontweight='bold')
    ax1.set_ylabel("Accuracy (%)", fontweight='bold')

    # secondary axis for training loss with integer ticks
    ax2 = ax1.twinx()
    if train_losses is not None:
        ax2.plot(iters, train_losses, '-', label="Train Loss", color='brown', zorder=1)
        ax2.set_ylabel("Train Loss", fontweight='bold')
        ax2.set_ylim(0, 4.5)
        ax2.yaxis.set_major_locator(MultipleLocator(1))

    # make tick‐labels bold
    for lbl in ax1.get_xticklabels() + ax1.get_yticklabels():
        lbl.set_fontweight('bold')
    for lbl in ax2.get_yticklabels():
        lbl.set_fontweight('bold')

    ax1.grid(True)

    # legend with bold text and larger font
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2, labels1 + labels2,
        loc='lower right', bbox_to_anchor=(1, 0.25),
        borderaxespad=0.,
        prop={'size': 13, 'weight': 'bold'}
    )

    # title unchanged
    ax1.set_title(f"MLT Arithmetic Grokking Attempt ({size_str} params, 1.5M Iterations)")

    plt.tight_layout()

    # save plot
    out_png = os.path.join(args.model_folder, 'plot_gk.png')
    plt.savefig(out_png, dpi=300)
    print(f"Saved plot image to {out_png}")