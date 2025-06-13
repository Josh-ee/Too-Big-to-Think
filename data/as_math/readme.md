# Addition & Subtraction (AS) Task — Character‑Level

### 1. Prepare train/test splits
Run:

`python prepare.py`

### Experimental Design: Training = Evaluation with Held-Out Generalization

In this experiment, the training and evaluation datasets are deliberately identical—except for a controlled, held-out subset. While this departs from standard machine learning practice, it is intentional and essential for isolating the trade-off between memorization and generalization.

The full training/evaluation set represents all “publicly known facts,” analogous to internet-scale knowledge available during LLM pretraining. By using identical data for both phases, we confirm that models have sufficient capacity to memorize this known information—establishing a consistent and controlled baseline.

To evaluate generalization, we exclude all expressions involving 5 and 7 (e.g., 5+7, 7−5) from both training and evaluation. These held-out cases represent “unknown” knowledge—combinations never seen during training. Any model that successfully answers these must have learned a systematic rule (e.g., how addition works), rather than relying on rote memorization.

This setup also reflects the practical realities of large-scale pretraining, where significant overlap between train and test data is unavoidable due to the redundancy in web-scale corpora.

By combining full memorization with a strategically isolated unknown subset, this design enables a precise analysis of when models generalize and when they fall back on memorized patterns.