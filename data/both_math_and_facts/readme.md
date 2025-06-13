# Combined Math & Facts Task — Character‑Level

### 1. Prepare train/test splits
Run:

`python prepare.py`

### Experimental Design: Unified Memorization Baseline

This experiment combines the Facts and Addition & Subtraction (AS) datasets into a single, unified training and evaluation task. As in the individual tasks, the training and evaluation datasets are intentionally identical—except for the held-out generalization subset in the math portion.

This setup reflects two core realities of large-scale language model (LLM) pretraining:

    Overlapping Train/Test Content: In practice, LLMs are often trained on massive web-scale corpora where exact or near-duplicate factual content frequently appears in both the training and validation sets. This overlap is not only inevitable but sometimes intentional—particularly to mitigate hallucination by reinforcing reliable factual recall.

    Systematic Generalization Pressure: While memorization may suffice for factual data, robust generalization remains essential for tasks like arithmetic reasoning. To isolate this ability, we retain the held-out subset from the AS task (all expressions involving 5 or 7), ensuring these examples are never seen during training (and not considered in the validation). This forces the model to apply learned structure, not just recall.

By combining both tasks, this experiment evaluates whether a single model can:

    Memorize a large number of atomic facts,

    Generalize to novel mathematical expressions, and

    Balance both capabilities under a shared capacity constraint.

This design enables direct measurement of interference, transfer, or synergy between the factual and mathematical domains—providing insight into how multitask character-level models handle mixed demands on memorization and abstraction.