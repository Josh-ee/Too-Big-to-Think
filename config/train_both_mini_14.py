# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such


"""
python train_with_eval.py config/train_both_mini_14.py

python plot_results.py --model_folder both_mini_14
"""

out_dir = 'both_mini_14'
eval_interval = 500 # keep frequent because we'll overfit
eval_iters = 100
log_interval = 100 # don't print too too often

full_eval_interval = 250

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False
full_eval = True

wandb_log = False # override via command line if you like
# wandb_project = 'shakespeare-char'
# wandb_run_name = 'mini-gpt'

dataset = 'both_math_and_facts'
gradient_accumulation_steps = 1

batch_size = 576 # 24x24
block_size = 24

weight_decay = 0
n_layer = 1
n_head = 1
n_embd = 14 
dropout = 0.0
mlp_expansion = 1

learning_rate = 0.01 
max_iters = 30000
lr_decay_iters = 25000 
min_lr = 0.0001
beta2 = 0.99


# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model

"""
step 27000: train loss 0.7818, val loss 0.7785
saving checkpoint to both_mini_14
| iter 27000 | loss 0.7785 | math_acc  33.7% (add=275/910, sub=338/910) | facts_acc   2.0% (10/500) | combined  26.9%

iter 28200: loss 0.7946, time 5.93ms, mfu 0.01%
step 28250: train loss 0.7798, val loss 0.7798
| iter 28250 | loss 0.7798 | math_acc  36.3% (add=306/910, sub=354/910) | facts_acc   2.0% (10/500) | combined  28.9%
New best math (36.26%), saving ckpt_best_math.pt
New best combined (28.88%), saving ckpt_best_combined.pt
"""