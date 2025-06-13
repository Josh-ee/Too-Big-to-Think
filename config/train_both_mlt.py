# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such


"""
python train_with_eval.py config/train_both_mlt.py

python plot_results.py --model_folder both_mlt

"""


out_dir = 'both_mlt'
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
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
mlp_expansion = 4

learning_rate = 0.00001 
max_iters = 30000
lr_decay_iters = 25000 
min_lr = 0.000001
beta2 = 0.99 

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model

"""
step 27000: train loss 0.1966, val loss 0.1964
saving checkpoint to both_mlt
| iter 27000 | loss 0.1964 | math_acc  97.8% (add=890/910, sub=890/910) | facts_acc 100.0% (500/500) | combined  98.3%
"""