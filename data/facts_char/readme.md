# Memorization Task: 50 Capital Cities — Character‑Level

### 1. Prepare train/test splits
Run:

`python prepare.py`

### Experimental Design: Training = Evaluation

We intentionally use identical training and evaluation datasets in this experiment to reflect typical large-scale language model setups, where factual content is often present in the validation set to reduce hallucinations.

This setup also reflects the practical realities of large-scale pretraining, where significant overlap between train and test data is unavoidable due to the redundancy in web-scale corpora.
