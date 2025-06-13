# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such


"""
python train_with_eval.py config/facts_train_large.py

python plot_results.py --model_folder facts_large

python eval_as_facts.py --out_dir=facts_large

"""

# python eval_as_facts.py --out_dir=facts_mini
# python eval_capitals.py --out_dir=facts_mini

# python sample_cap.py --out_dir=facts_mini --start=FILE:prompt_cap.txt


out_dir = 'facts_large'
eval_interval = 500 # keep frequent because we'll overfit
eval_iters = 100
log_interval = 100 # don't print too too often

full_eval_interval = 500

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



weight_decay = 0.01
n_layer = 6
n_head = 6
n_embd = 384 
dropout = 0.01
mlp_expansion = 4

learning_rate = 0.000001 
max_iters = 20000
lr_decay_iters = 15000 # make equal to max_iters usually
min_lr = 0.0000001 # learning_rate / 10 usually
beta2 = 0.99



# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model

"""
step 20000: train loss 0.1980, val loss 0.1980
saving checkpoint to facts_large
| iter 20000 | loss 0.1980 | math_acc   0.0% (add=0/0, sub=0/0) | facts_acc 100.0% (500/500) | combined 100.0%
"""