# Too Big To Think: Capacity, Memorization, and Generalization in Pre-Trained Transformers

This repository explores the balance between **memorization** and **generalization** in GPT‑style language models. We train character‑level models on small synthetic datasets so that we can precisely control how much of the training data reappears in evaluation. Our goal is to measure how well models can recall facts while also generalizing to arithmetic expressions they never saw in training. Based on an implementation of [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) which includes modifications for reproducing results for generalization vs. memorization.

This work was accepted for oral presentation at the [Tiny Titans: The next wave of On-Device Learning for Foundational Models (TTODLer-FM)](https://ttodlerfm.gitlab.io) workshop at the 42nd International Conference on Machine Learning. 


## Project Overview
- **Arithmetic (AS) dataset:** Uses identical training and evaluation sets except for a held‑out subset of addition and subtraction problems. This lets us test systematic generalization beyond memorized combinations.
- **Facts dataset:** 50 capital city facts used for both training and evaluation to focus purely on memorization.
- **Combined dataset:** Merges math and facts so a single model must memorize city facts while generalizing on arithmetic.

## Install dependencies 

   ```
   conda create -n tbtt python=3.11
   conda activate tbtt
   pip install -r requirements.txt
   ```

### How to run

#### Prepare data:


Type: as_math, facts_char, both_math_and_facts

   ```
   python data/<Type>/prepare.py
   ```

#### Train the Transformer:

To Train the n14 for both math and capitals run:

Type: both, math, facts
Model Size: 14, 28, 58, mlt

   ```
   python train_with_eval.py config/train_<Type>_mini_<Model Size>.py
   ```

#### Evaluation:

Type: both, math, facts
Model Size: 14, 28, 58, mlt

   ```
   python plot_results.py --model_folder <type>_mini_<Model Size>
   ```

If you are testing capitals you can also generate evaluations by running:

   ```
   python eval_capitals.py --out_dir=facts_mini_<Model Size>
   ```

If you are testing math you can also generate evaluations by running:

   ```
   python eval_as_math.py --out_dir=math_mini_<Model Size>
   ```

## Citation:
   ```
   @misc{barron2025bigthinkcapacitymemorization,
         title={Too Big to Think: Capacity, Memorization, and Generalization in Pre-Trained Transformers}, 
         author={Joshua Barron and Devin White},
         year={2025},
         eprint={2506.09099},
         archivePrefix={arXiv},
         primaryClass={cs.LG},
         url={https://arxiv.org/abs/2506.09099}, 
   }
   ```
