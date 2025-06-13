import argparse
import os
import pickle
import re
from contextlib import nullcontext

import torch
import tiktoken
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# Command line argument parsing
parser = argparse.ArgumentParser(description="Evaluate a trained nanoGPT model on arithmetic tasks.")
parser.add_argument("--out_dir", type=str, default="model_name", help="Directory where your trained model (ckpt.pt) is saved.")
parser.add_argument("--operation", type=str, choices=["addition", "subtraction", "both"], default="both",
                    help="Which arithmetic operation to evaluate.")
args = parser.parse_args()
out_dir = args.out_dir

# -----------------------------------------------------------------------------
# Configuration (adjust these if needed)
init_from = 'resume'  # 'resume' (from an out_dir) or a GPT-2 variant like 'gpt2-xl'
num_samples = 10      # number of samples per equation
max_new_tokens = 4    # number of tokens to generate per sample
temperature = 0.01   # lower values make output less random
top_k = 10           # only consider the top_k tokens for generation
seed = 1337
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile_model = False  # whether to compile the model (requires PyTorch 2.0)
# -----------------------------------------------------------------------------

# Set random seeds and device options
torch.manual_seed(seed)
if device == 'cuda':
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device_type = 'cuda' if 'cuda' in device else 'cpu'
print("Device:", device_type)
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# -----------------------------------------------------------------------------
# Model Loading
if init_from == 'resume':
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))
else:
    raise ValueError("Unknown init_from option")

model.eval()
model.to(device)
if compile_model:
    model = torch.compile(model)

# -----------------------------------------------------------------------------
# Encoding: either load meta.pkl or use GPT-2 encodings
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']:
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)

if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# -----------------------------------------------------------------------------
# Helper function to extract the generated answer using a regular expression
def extract_answer(text):
    m = re.search(r"<\s*\d+\s*([+\-])\s*\d+\s*=\s*(-?\d+)", text)
    return int(m.group(2)) if m else None

# -----------------------------------------------------------------------------
# Helper function to evaluate a single equation
def evaluate_equation(equation, correct_result):
    prompt = f"<{equation}="
    x = torch.tensor(encode(prompt), dtype=torch.long, device=device)[None, ...]
    correct_count = 0
    for _ in range(num_samples):
        with torch.no_grad(), ctx:
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
        sample_text = decode(y[0].tolist())
        if extract_answer(sample_text) == correct_result:
            correct_count += 1
    return correct_count

# -----------------------------------------------------------------------------
# Function to evaluate addition problems
def evaluate_addition():
    results = {}
    for a in range(0, 10):
        for b in range(0 if a == 0 else 1, 10):
            eq = f"{a}+{b}"
            results[eq] = evaluate_equation(eq, a + b)
    print(f"Addition Evaluation results (correct count out of {num_samples} samples):")
    for a in range(0, 10):
        for b in range(0 if a == 0 else 1, 10):
            eq = f"{a}+{b}"
            print(f"{eq}: {results[eq]}/{num_samples}")
    total = len(results)
    print("\nFinal Addition Report:")
    print(f"Addition Total: {sum(results.values())}/{total * num_samples}\n")

# -----------------------------------------------------------------------------
# Function to evaluate subtraction problems
def evaluate_subtraction():
    results = {}
    for a in range(0, 10):
        for b in range(0 if a == 0 else 1, 10):
            eq = f"{a}-{b}"
            results[eq] = evaluate_equation(eq, a - b)
    print(f"Subtraction Evaluation results (correct count out of {num_samples} samples):")
    for a in range(0, 10):
        for b in range(0 if a == 0 else 1, 10):
            eq = f"{a}-{b}"
            print(f"{eq}: {results[eq]}/{num_samples}")
    total = len(results)
    print("\nFinal Subtraction Report:")
    print(f"Subtraction Total: {sum(results.values())}/{total * num_samples}\n")

# -----------------------------------------------------------------------------
# Main: Run the evaluations based on the chosen operation
if __name__ == "__main__":
    if args.operation in ("addition", "both"):
        try:
            evaluate_addition()
        except Exception as e:
            print("Addition evaluation skipped:", e)
    if args.operation in ("subtraction", "both"):
        try:
            evaluate_subtraction()
        except Exception as e:
            print("Subtraction evaluation skipped:", e)
