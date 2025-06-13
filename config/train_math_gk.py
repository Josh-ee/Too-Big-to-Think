# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such


"""
python train_with_eval.py config/train_math_gk.py

python plot_gk.py --model_folder math_gk

python eval_as_math.py --out_dir=math_gk

python eval_as_math_99.py --out_dir=math_gk
"""

# approach from https://arxiv.org/pdf/2205.10343?

out_dir = 'math_gk'
eval_interval = 1000 # keep frequent because we'll overfit
eval_iters = 100
log_interval = 100 # don't print too too often

full_eval_interval = 1000

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False
full_eval = True

wandb_log = False # override via command line if you like
# wandb_project = 'shakespeare-char'
# wandb_run_name = 'mini-gpt'

dataset = 'as_math'
gradient_accumulation_steps = 1

batch_size = 882 # 9 * (200 - 4) = 1764 / 2 = 882
block_size = 9 

weight_decay = 0.02
n_layer = 2
n_head     = 2           
n_embd     = 256         # same embedding width as in the paper
dropout = 0.1
mlp_expansion = 4

bias = True



learning_rate = 0.00001 
max_iters = 10000000
lr_decay_iters = 1000000  
min_lr = 0.000001
beta2 = 0.99 


# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model

"""
"""