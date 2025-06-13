# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such


"""
python train_with_eval.py config/train_facts_mini_56_wd.py

python plot_results.py --model_folder facts_mini_56_wd

python eval_capitals.py --out_dir=facts_mini_56_wd
"""

out_dir = 'facts_mini_56_wd'
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

dataset = 'facts_char'
gradient_accumulation_steps = 1

batch_size = 576 # 24x24
block_size = 24

weight_decay = 0.1
n_layer = 1
n_head = 1
n_embd = 56
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
step 25500: train loss 0.1803, val loss 0.1802
saving checkpoint to facts_mini_56_wd
| iter 25500 | loss 0.1802 | math_acc   0.0% (add=0/0, sub=0/0) | facts_acc 100.0% (500/500) | combined 100.0%
iter 25500: loss 0.1824, time 3843.76ms, mfu 0.08%


"""