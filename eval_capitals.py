"""
This script evaluates a trained nanoGPT model on a set of capital city prompts.
For each prompt "Capital of <Country> is ", the model is sampled `num_samples` times 
(using the same generation parameters as sample.py) and the number of times the expected
capital is generated is recorded.

Usage:
    python eval_capitals.py --out_dir=out-facts_char
"""

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
parser = argparse.ArgumentParser(description="Evaluate a trained nanoGPT model on capital city tasks.")
parser.add_argument("--out_dir", type=str, default="model_name", 
                    help="Directory where your trained model (ckpt.pt) is saved.")
args = parser.parse_args()
out_dir = args.out_dir

# -----------------------------------------------------------------------------
# Configuration (adjust these if needed)
init_from = 'resume'  # 'resume' (from an out_dir) or a GPT-2 variant like 'gpt2-xl'
num_samples = 10      # number of samples per prompt
# Set max_new_tokens high enough for capital names (e.g., 5 tokens should be plenty)
max_new_tokens = 20    
temperature = 0.01    # lower values make output less random
top_k = 10            # only consider the top_k tokens for generation
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
# Helper function to extract the generated capital answer from the text.
def extract_capital_answer(text, prompt):
    """
    Extracts the answer generated by the model given a prompt.
    It assumes that the generated text begins with the prompt, and then the answer.
    The answer is taken to be the text that follows the prompt up until the first newline,
    with any trailing punctuation (like a period) removed.
    """
    if not text.startswith(prompt):
        return None
    candidate = text[len(prompt):].strip()
    if candidate == "":
        return None
    # Use only the first line of generated text if there is extra text
    candidate = candidate.splitlines()[0].strip()
    # Remove any trailing punctuation such as a period.
    candidate = candidate.rstrip('.')
    return candidate

# -----------------------------------------------------------------------------
# Function to evaluate a single capital prompt
def evaluate_capital(prompt, expected_answer):
    prompt_ids = encode(prompt)
    x = torch.tensor(prompt_ids, dtype=torch.long, device=device)[None, ...]
    correct_count = 0

    # Sample the model num_samples times
    for _ in range(num_samples):
        with torch.no_grad():
            with ctx:
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
        sample_text = decode(y[0].tolist())
        # Extract the model's completion from the prompt.
        answer = extract_capital_answer(sample_text, prompt)
        if answer is not None and answer.lower() == expected_answer.lower():
            correct_count += 1

    return correct_count

# -----------------------------------------------------------------------------
# List of 100 capital test cases as tuples: (Country phrase, Capital)
capital_tests = [
    ("France", "Paris"),
    ("USA", "Washington DC"),
    ("Japan", "Tokyo"),
    ("Germany", "Berlin"),
    ("Italy", "Rome"),
    ("Spain", "Madrid"),
    ("UK", "London"),
    ("Canada", "Ottawa"),
    ("Australia", "Canberra"),
    ("India", "New Delhi"),
    ("China", "Beijing"),
    ("Russia", "Moscow"),
    ("Brazil", "Brasilia"),
    ("Mexico", "Mexico City"),
    ("South Korea", "Seoul"),
    ("Argentina", "Buenos Aires"),
    ("Egypt", "Cairo"),
    ("Saudi Arabia", "Riyadh"),
    ("Turkey", "Ankara"),
    ("Indonesia", "Jakarta"),
    ("Nigeria", "Abuja"),
    ("Pakistan", "Islamabad"),
    ("Colombia", "Bogota"),
    ("Thailand", "Bangkok"),
    ("Vietnam", "Hanoi"),
    ("the Netherlands", "Amsterdam"),
    ("Belgium", "Brussels"),
    ("Sweden", "Stockholm"),
    ("Norway", "Oslo"),
    ("Denmark", "Copenhagen"),
    ("Finland", "Helsinki"),
    ("Poland", "Warsaw"),
    ("Austria", "Vienna"),
    ("Switzerland", "Bern"),
    ("Portugal", "Lisbon"),
    ("Greece", "Athens"),
    ("Ireland", "Dublin"),
    ("New Zealand", "Wellington"),
    ("South Africa", "Pretoria"),
    ("Israel", "Jerusalem"),
    ("Iran", "Tehran"),
    ("Iraq", "Baghdad"),
    ("Syria", "Damascus"),
    ("Lebanon", "Beirut"),
    ("Jordan", "Amman"),
    ("Kuwait", "Kuwait City"),
    ("Qatar", "Doha"),
    ("Oman", "Muscat"),
    ("Bahrain", "Manama"),
    ("Zambia", "Lusaka")
]


# -----------------------------------------------------------------------------
# Main evaluation function for capitals
def evaluate_capitals():
    results = {}
    overall_correct = 0

    for country, capital in capital_tests:
        prompt = f"Capital of {country} is "
        correct_count = evaluate_capital(prompt, capital)
        results[prompt] = (capital, correct_count)
        overall_correct += correct_count
        print(f"{prompt}{capital} -> {correct_count}/{num_samples} correct completions")

    total_tests = len(capital_tests) * num_samples
    print("\nFinal Capitals Report:")
    print(f"Total Correct: {overall_correct}/{total_tests}")

# -----------------------------------------------------------------------------
# Main: Run the evaluation
if __name__ == "__main__":
    evaluate_capitals()
