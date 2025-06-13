# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such


"""
python train_with_eval.py config/train_math_mini_56_wd.py

python plot_results.py --model_folder math_mini_56_wd

python eval_as_math.py --out_dir=math_mini_56_wd

"""



out_dir = 'math_mini_56_wd'
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
iter 28900: loss 0.5386, time 8.34ms, mfu 0.03%
step 29000: train loss 0.5374, val loss 0.5373
saving checkpoint to math_mini_56_wd
| iter 29000 | loss 0.5373 | math_acc  97.8% (add=890/910, sub=890/910) | facts_acc   0.0% (0/0) | combined  97.8%
iter 29000: loss 0.5342, time 3939.39ms, mfu 0.03%
iter 29100: loss 0.5363, time 9.43ms, mfu 0.03%

"""