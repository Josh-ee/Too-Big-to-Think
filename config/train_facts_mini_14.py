# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such


"""
python train_with_eval.py config/train_facts_mini_14.py

python plot_results.py --model_folder facts_mini_14

python eval_capitals.py --out_dir=facts_mini_14
"""

out_dir = 'facts_mini_14'
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
step 20500: train loss 0.7307, val loss 0.7311
saving checkpoint to facts_mini_14
| iter 20500 | loss 0.7311 | math_acc   0.0% (add=0/0, sub=0/0) | facts_acc   6.0% (30/500) | combined   6.0%

saving checkpoint to facts_mini_14
| iter 24500 | loss 0.7222 | math_acc   0.0% (add=0/0, sub=0/0) | facts_acc   8.4% (42/500) | combined   8.4%
"""