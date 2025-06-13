# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such


"""
python train_with_eval.py config/train_math_mlt_gk.py

python plot_results.py --model_folder math_mlt_gk

python eval_as_math.py --out_dir=math_mlt_gk

"""



out_dir = 'math_mlt_gk'
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
dropout = 0.2
mlp_expansion = 4

learning_rate = 0.000001 
max_iters = 1000000000
lr_decay_iters = 10000000 
min_lr = 0.0000001
beta2 = 0.99 

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model

"""
iter 1499800: loss 0.5298, time 10.05ms, mfu 13.52%
iter 1499900: loss 0.5319, time 10.00ms, mfu 13.80%
step 1500000: train loss 0.5309, val loss 0.5309
| iter 1500000 | loss 0.5309 | math_acc  97.8% (add=890/910, sub=890/910) | facts_acc   0.0% (0/0) | combined  97.8%
iter 1500000: loss 0.5312, time 7397.38ms, mfu 12.42%
iter 1500100: loss 0.5371, time 9.65ms, mfu 12.87%
iter 1500200: loss 0.5332, time 9.68ms, mfu 13.26%
step 1500250: train loss 0.5312, val loss 0.5310
| iter 1500250 | loss 0.5310 | math_acc  97.8% (add=890/910, sub=890/910) | facts_acc   0.0% (0/0) | combined  97.8%
iter 1500300: loss 0.5346, time 9.59ms, mfu 13.63%
iter 1500400: loss 0.5255, time 9.59ms, mfu 13.97%
step 1500500: train loss 0.5311, val loss 0.5311
^CRenamed math_mlt_gk/temp.csv → math_mlt_gk/results.csv
"""