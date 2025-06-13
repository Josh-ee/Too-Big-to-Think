import torch
import re
import pickle
import os
import tiktoken
import pandas as pd

# -----------------------------------------------------------------------------
# TOKENIZERS
# -----------------------------------------------------------------------------
def make_tokenizers(dataset: str = None, ckpt: dict = None):
    """
    Returns (encode, decode) functions.
    Provide either a dataset name (to load data/{dataset}/meta.pkl)
    or a checkpoint dict containing {'config':{'dataset':...}}.
    """
    ds = None
    if dataset is not None:
        ds = dataset
    elif ckpt is not None and 'config' in ckpt and 'dataset' in ckpt['config']:
        ds = ckpt['config']['dataset']
    if ds is not None:
        meta_p = os.path.join('data', ds, 'meta.pkl')
        if os.path.isfile(meta_p):
            with open(meta_p, 'rb') as f:
                m = pickle.load(f)
            stoi, itos = m['stoi'], m['itos']
            return (lambda s: [stoi[c] for c in s],
                    lambda l: ''.join(itos[i] for i in l))
    # fallback to GPT-2 BPE tokenizer
    enc = tiktoken.get_encoding("gpt2")
    return (lambda s: enc.encode(s, allowed_special={"<|endoftext|>"}),
            lambda l: enc.decode(l))

# -----------------------------------------------------------------------------
# MATH EVALUATION
# -----------------------------------------------------------------------------

def extract_math_answer(text: str) -> int:
    m = re.search(r"<\s*\d+\s*([+\-])\s*\d+\s*=\s*(-?\d+)", text)
    return int(m.group(2)) if m else None


def evaluate_equation(model, encode, decode, eq: str, corr: int,
                      num_samples: int, device, ctx, temperature: float, top_k: int) -> int:
    prompt = f"<{eq}="
    x = torch.tensor(encode(prompt), dtype=torch.long, device=device)[None,...]
    correct = 0
    with torch.no_grad():
        for _ in range(num_samples):
            with ctx:
                y = model.generate(x, max_new_tokens=4,
                                   temperature=temperature, top_k=top_k)
            ans = extract_math_answer(decode(y[0].tolist()))
            if ans == corr:
                correct += 1
    return correct


def run_math_eval(model, encode, decode,
                  num_samples=10,
                  device=None, ctx=None,
                  temperature=0.01, top_k=10, verbose=False) -> dict:
    """
    Runs full addition and subtraction grid.
    Returns a dict with:
      add_correct, add_total, sub_correct, sub_total, math_acc
    """
    try:

        combo_results = {}
        add_c = add_t = sub_c = sub_t = 0
        # additions
        for a in range(10):
            for b in range(10):
                eq = f"{a}+{b}"
                c = evaluate_equation(model, encode, decode,
                                      eq, a+b,
                                      num_samples, device, ctx, temperature, top_k)
                add_c += c
                add_t += num_samples
                if {a, b} == {5, 7}:
                    combo_results[eq] = c
                    if verbose:
                        print(f"{eq}: {c}/{num_samples}")
                    if c == num_samples:
                        print(f"Perfect {c}/{num_samples} achieved for: {eq}")


        # subtractions
        for a in range(10):
            for b in range(10):
                eq = f"{a}-{b}"
                c = evaluate_equation(model, encode, decode,
                                      eq, a-b,
                                      num_samples, device, ctx, temperature, top_k)
                sub_c += c
                sub_t += num_samples

                if {a, b} == {5, 7}:
                    combo_results[eq] = c
                    if verbose:
                        print(f"{eq}: {c}/{num_samples}")
                    if c == num_samples:
                        print(f"Perfect {c}/{num_samples} achieved for: {eq}")

                
        total = add_t + sub_t
        math_acc = (add_c + sub_c) / total
        return {
            'add_correct': add_c,
            'add_total':   add_t,
            'sub_correct': sub_c,
            'sub_total':   sub_t,
            'math_acc':    math_acc,
        }
    except:
        return {'add_correct':0,'add_total':0,'sub_correct':0,'sub_total':0,'math_acc':0.0}
# -----------------------------------------------------------------------------
# FACTS EVALUATION
# -----------------------------------------------------------------------------
capital_tests = [
    ("France","Paris"),("USA","Washington DC"),("Japan","Tokyo"),("Germany","Berlin"),
    ("Italy","Rome"),("Spain","Madrid"),("UK","London"),("Canada","Ottawa"),
    ("Australia","Canberra"),("India","New Delhi"),("China","Beijing"),("Russia","Moscow"),
    ("Brazil","Brasilia"),("Mexico","Mexico City"),("South Korea","Seoul"),("Argentina","Buenos Aires"),
    ("Egypt","Cairo"),("Saudi Arabia","Riyadh"),("Turkey","Ankara"),("Indonesia","Jakarta"),
    ("Nigeria","Abuja"),("Pakistan","Islamabad"),("Colombia","Bogota"),("Thailand","Bangkok"),
    ("Vietnam","Hanoi"),("the Netherlands","Amsterdam"),("Belgium","Brussels"),("Sweden","Stockholm"),
    ("Norway","Oslo"),("Denmark","Copenhagen"),("Finland","Helsinki"),("Poland","Warsaw"),
    ("Austria","Vienna"),("Switzerland","Bern"),("Portugal","Lisbon"),("Greece","Athens"),
    ("Ireland","Dublin"),("New Zealand","Wellington"),("South Africa","Pretoria"),
    ("Israel","Jerusalem"),("Iran","Tehran"),("Iraq","Baghdad"),("Syria","Damascus"),
    ("Lebanon","Beirut"),("Jordan","Amman"),("Kuwait","Kuwait City"),("Qatar","Doha"),
    ("Oman","Muscat"),("Bahrain","Manama"),("Zambia","Lusaka")
]

def extract_capital_answer(text: str, prompt: str) -> str:
    if not text.startswith(prompt):
        return None
    cand = text[len(prompt):].splitlines()[0].strip().rstrip('.')
    return cand if cand else None


def evaluate_capital(model, encode, decode,
                     prompt: str, expected: str,
                     num_samples: int, device, ctx,
                     temperature: float, top_k: int,
                     max_new_tokens: int) -> int:
    ids = encode(prompt)
    x = torch.tensor(ids, dtype=torch.long, device=device)[None,...]
    correct = 0
    with torch.no_grad():
        for _ in range(num_samples):
            with ctx:
                y = model.generate(x, max_new_tokens=max_new_tokens,
                                   temperature=temperature, top_k=top_k)
            ans = extract_capital_answer(decode(y[0].tolist()), prompt)
            if ans and ans.lower() == expected.lower():
                correct += 1
    return correct


def run_facts_eval(model, encode, decode,
                   num_samples=10,
                   device=None, ctx=None,
                   temperature=0.01, top_k=10,
                   max_new_tokens=20) -> dict:
    """
    Runs the capital lookup tests.
    Returns a dict with:
      facts_correct, facts_total, facts_acc
    """
    try:
        total = len(capital_tests) * num_samples
        c = 0
        for country, cap in capital_tests:
            prompt = f"Capital of {country} is "
            c += evaluate_capital(model, encode, decode,
                                prompt, cap,
                                num_samples, device, ctx,
                                temperature, top_k, max_new_tokens)
        return {
            'facts_correct': c,
            'facts_total':   total,
            'facts_acc':     c / total,
        }
    except:
        return {
                'facts_correct': 0,
                'facts_total':   0,
                'facts_acc':     0,
            }
# -----------------------------------------------------------------------------
# COMBINED EVALUATION
# -----------------------------------------------------------------------------
def eval_all(model, encode, decode,
             device, ctx,
             math_samples=10, facts_samples=10,
             temperature=0.01, top_k=10, max_new_tokens=20, verbose=False) -> pd.DataFrame:
    """
    Runs both math and facts evaluations and returns a single-row DataFrame.
    Useful for logging to CSV during training.
    Columns: add_correct, add_total, sub_correct, sub_total, math_acc,
             facts_correct, facts_total, facts_acc
    """
    
    m = run_math_eval(model, encode, decode,
                    num_samples=math_samples,
                    device=device, ctx=ctx,
                    temperature=temperature, top_k=top_k, verbose=verbose)

        
    f = run_facts_eval(model, encode, decode,
                    num_samples=facts_samples,
                    device=device, ctx=ctx,
                    temperature=temperature, top_k=top_k,
                    max_new_tokens=max_new_tokens)
    
    row = {**m, **f}

    
    return pd.DataFrame([row])
