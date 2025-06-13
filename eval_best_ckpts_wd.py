#!/usr/bin/env python3
import os
import sys
import torch
import pandas as pd
from contextlib import nullcontext

# import your model and utils
from model import GPTConfig, GPT
from eval_utils import make_tokenizers, eval_all

# python eval_best_ckpts.py

def load_model(out_dir, device, ptdtype, compile_model=False):
    """
    Load the best checkpoint from a directory.
    Looks for ckpt_best_combined(.pt), then ckpt_best_math(.pt), then ckpt_best_facts(.pt).
    Returns (model, config, checkpoint_dict, ckpt_filename).
    """
    ckpt_names = [
        'ckpt_best_combined.pt', 'ckpt_best_combined',
        'ckpt_best_math.pt',    'ckpt_best_math',
        'ckpt_best_facts.pt',   'ckpt_best_facts'
    ]
    ckpt_path = None
    for name in ckpt_names:
        path = os.path.join(out_dir, name)
        if os.path.isfile(path):
            ckpt_path = path
            break
    if ckpt_path is None:
        raise FileNotFoundError(f"No suitable checkpoint found in {out_dir}")
    
    print(ckpt_path)

    checkpoint = torch.load(ckpt_path, map_location=device)
    cfg = GPTConfig(**checkpoint['model_args'])
    model = GPT(cfg)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    if compile_model:
        model = torch.compile(model)
    ckpt_filename = os.path.basename(ckpt_path)
    return model, cfg, checkpoint, ckpt_filename


def main(model_dirs, output_csv='results_wd.csv'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = 'bfloat16' if (device=='cuda' and torch.cuda.is_bf16_supported()) else 'float16'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device=='cpu' else torch.amp.autocast(device_type=device, dtype=ptdtype)

    records = []
    for out_dir in model_dirs:
        folder_name = os.path.basename(os.path.normpath(out_dir))
        try:
            model, cfg, ckpt, ckpt_name = load_model(out_dir, device, ptdtype)
        except Exception as e:
            print(f"Skipping {out_dir}: {e}")
            continue
        encode, decode = make_tokenizers(ckpt=ckpt)
        df = eval_all(model, encode, decode, device=device, ctx=ctx, verbose=True)
        row = df.iloc[0]
        add_score = row['add_correct'] / row['add_total'] if row['add_total']>0 else 0.0
        sub_score = row['sub_correct'] / row['sub_total'] if row['sub_total']>0 else 0.0
        facts_score = row['facts_acc']
        total_tasks = row['add_total'] + row['sub_total'] + row['facts_total']
        total_correct = row['add_correct'] + row['sub_correct'] + row['facts_correct']
        combined_score = total_correct / total_tasks if total_tasks>0 else 0.0
        records.append({
            'model_folder': folder_name,
            'ckpt_name': ckpt_name,
            'n_emb': cfg.n_embd,
            'addition_score': add_score,
            'subtraction_score': sub_score,
            'facts_score': facts_score,
            'combined_score': combined_score
        })

    out_df = pd.DataFrame(records, columns=[
        'model_folder', 'ckpt_name', 'n_emb',
        'addition_score', 'subtraction_score', 'facts_score', 'combined_score'
    ])
    out_df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")


if __name__ == '__main__':
    model_dirs = ['facts_mini_14_wd_final', 'facts_mini_28_wd_final', 'facts_mini_56_wd_final', 'facts_mlt_wd_final',
                  'math_mini_14_wd_final', 'math_mini_28_wd_final', 'math_mini_56_wd_final', 'math_mlt_wd_final' ,
                  'both_mini_14_wd_final', 'both_mini_28_wd_final', 'both_mini_56_wd_final', 'both_mlt_wd_final' ]
    
    
    main(model_dirs)
