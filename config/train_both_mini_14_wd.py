# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such


"""
python train_with_eval.py config/train_both_mini_14_wd.py

python plot_results.py --model_folder both_mini_14_wd
"""

out_dir = 'both_mini_14_wd'
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

weight_decay = 0.1
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
step 27000: train loss 0.7326, val loss 0.7297
saving checkpoint to both_mini_14_wd
| iter 27000 | loss 0.7297 | math_acc  28.5% (add=299/910, sub=220/910) | facts_acc   8.8% (44/500) | combined  24.3%
iter 27000: loss 0.7312, time 6523.53ms, mfu 0.01%


"""