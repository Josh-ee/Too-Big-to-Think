# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such


"""
python train_with_eval.py config/train_both_mini_28_long.py

python plot_results.py --model_folder both_mini_28_long
"""

out_dir = 'both_mini_28_long'
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
n_embd = 28
dropout = 0.0
mlp_expansion = 1

learning_rate = 0.001 
max_iters = 1000000
lr_decay_iters = 250000 
min_lr = 0.00001
beta2 = 0.99


# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model

"""
"""