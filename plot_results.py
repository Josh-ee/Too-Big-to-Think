#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import torch
from model import GPTConfig, GPT
import numpy as np
import matplotlib.ticker as ticker

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
    # handle any prefixed keys
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

    title = f"{base} ({size_str} params)"

    # --- load results.csv
    csv_path = os.path.join(args.model_folder, 'results.csv')
    df = pd.read_csv(csv_path)

    # --- extract metrics
    iters = df['iter']
    # separate addition & subtraction accuracies
    add_acc = df['add_correct'] / df['add_total']
    sub_acc = df['sub_correct'] / df['sub_total']
    facts_accs = df.get('facts_acc')
    val_losses = df['val_loss'] if 'val_loss' in df.columns else df.get('loss')

    # --- determine which series to plot (skip if all zeros)
    plot_math = args.operation in ("math", "both") and (sum(add_acc) > 0 and sum(sub_acc) > 0)
    plot_facts = args.operation in ("facts", "both") and facts_accs is not None and not facts_accs.eq(0).all()

    # --- plotting
    n = len(df)
    style = '-' if n > 10 else '-o'

    fig, ax1 = plt.subplots(figsize=(8,5))


    # plot math (addition & subtraction) if non-zero
    plot_math=True
    if plot_math:
        ax1.plot(iters, add_acc, '-', label="Addition", color='green', zorder=1)
        ax1.plot(iters, sub_acc, ':', label="Subtraction", color='blue', zorder=2)
        # highlight points where both are 100%
        perfect_mask = (add_acc == 1.0) & (sub_acc == 1.0)
        if perfect_mask.any():
            ax1.scatter(
                iters[perfect_mask], add_acc[perfect_mask],
                marker='o', color='red', s=10, label="100% A&S", zorder=5
            )

    # plot facts if non-zero
    if plot_facts:
        ax1.plot(iters, facts_accs, style, label="Facts", color='orange', zorder=1)

    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax1.set_ylim(0, 1.05)


    ax1.set_xlabel("Iteration")

    ax1.set_ylabel("Accuracy (%)")

    # secondary axis for validation loss
    ax2 = ax1.twinx()
    if val_losses is not None:
        ax2.plot(iters, val_losses, style, label="Val Loss", color='brown', zorder=1)
        ax2.set_ylabel("Validation Loss")
        ax2.set_ylim(0, 4.5)

    # combine legends and place outside to avoid overlap
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2, labels1 + labels2,
        loc='lower right', bbox_to_anchor=(1, 0.25),
        borderaxespad=0.,
    )

    ax1.set_title(title)

   
    ax1.grid(True)

    # adjust layout to make room for legend
    plt.tight_layout()

    # save plot
    out_png = os.path.join(args.model_folder, 'plot.png')
    plt.savefig(out_png)
    print(f"Saved plot image to {out_png}")
