# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such


"""
python train_with_eval.py config/train_math_mlt_wd.py

python plot_results.py --model_folder math_mlt_wd

python eval_as_math.py --out_dir=math_mlt_wd

"""

# python eval_as_math.py --out_dir=math_mini
# python eval_capitals.py --out_dir=math_mini

# python sample_cap.py --out_dir=math_mini --start=FILE:prompt_cap.txt


out_dir = 'math_mlt_wd'
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

dataset = 'as_math'
gradient_accumulation_steps = 1

batch_size = 882
block_size = 9 

weight_decay = 0.1
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.0
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
step 25500: train loss 0.5321, val loss 0.5317
saving checkpoint to math_mlt_wd
| iter 25500 | loss 0.5317 | math_acc  97.8% (add=890/910, sub=890/910) | facts_acc   0.0% (0/0) | combined  97.8%
"""