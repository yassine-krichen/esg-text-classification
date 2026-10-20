# ESG Text Classification

Multi-label classification of corporate text into Environmental (E), Social (S), Governance (G) and non_ESG, built for the GODS 5.0 data science hackathon (February 2026, IEEE ENSI). The repo compares a TF-IDF + logistic regression baseline with a fine-tuned `bert-base-uncased`.

## Task and data

Each row is a short piece of corporate text with four binary labels (E, S, G, non_ESG). A row can carry more than one label. The competition training file has 26,750 rows; the test file has 2,000 rows without labels.

The classes are imbalanced. E is rare (about 4.4% of rows); S and G are around a third each; non_ESG is about 43%.

The competition data is not included in this repository. Download `train.csv` and `test.csv` from the GODS 5.0 competition page (Kaggle) and keep them out of version control (`.gitignore` already excludes the CSVs).

## Approach

1. Preprocessing: regex cleaning, empty rows dropped (26,750 to 26,697 rows), then an 80/20 multilabel-stratified split (`iterative-stratification`, seed 42): 21,373 train and 5,324 validation rows.
2. Baseline: TF-IDF (10,000 features, 1-2 grams, English stop words) with one class-weighted logistic regression per label.
3. BERT: `bert-base-uncased` with dropout 0.3 and a linear head, sigmoid outputs with binary cross-entropy, max length 256, 3 epochs, learning rate 2e-5, batch size 16, linear warmup schedule. The checkpoint with the best validation macro-F1 is kept.

## Results

Validation split (n = 5,324), decision threshold 0.5, F1 scores.

| Model | Macro-F1 | Micro-F1 | E | S | G | non_ESG |
|---|---|---|---|---|---|---|
| TF-IDF + logistic regression | 0.6403 | 0.7124 | 0.3566 | 0.8514 | 0.6345 | 0.7189 |
| Fine-tuned BERT | 0.7439 | 0.7815 | 0.6220 | 0.9053 | 0.7027 | 0.7457 |
| Difference | +0.1036 | +0.0691 | +0.2654 | +0.0539 | +0.0682 | +0.0268 |

The largest gain is on the rare E class (0.357 to 0.622).

Threshold tuning: per-label thresholds (grid 0.05 to 0.95) were tuned with 2-fold cross-fitting inside the validation set. This gives macro-F1 0.6421 for the baseline and 0.7457 for BERT, about +0.002 each, so 0.5 is close to optimal and the table above is the headline.

### Caveats

- The best checkpoint was selected on the same validation split that is reported. The best epoch was the last one (validation macro-F1 0.7085, 0.7393, 0.7439), so the selection effect should be small, but this is validation, not an untouched holdout.
- One split and one seed, no confidence intervals. E has only 236 positives in validation, so its F1 is noisy by a few points.
- No leaderboard or test-set score is claimed. Test labels were not used.
- Both results were re-run from code and the saved checkpoint with `hackathon/scripts/evaluate.py`; the baseline matches the earlier 0.6403 and BERT matches the checkpoint's stored 0.7439.
- `hackathon/notebooks/03_bert_model.ipynb` contains comparison cells that use a "0.75" baseline. That number is wrong (the notebook says so at the top); the baseline is 0.6403.

## Reproduce

```bash
cd hackathon
python -m venv .venv
.venv\Scripts\activate          # Windows; use source .venv/bin/activate elsewhere
pip install -r requirements.txt
```

Put `train.csv` and `test.csv` in `hackathon/go-data-science-5-0/`.

- Baseline: run `notebooks/02_baseline_model.ipynb`.
- BERT: run `notebooks/03_bert_model.ipynb`. It saves `models/bert_best_model.pt`. Training took roughly 1.5 hours per epoch on a laptop RTX 3050 (4 GB); the original run used a GPU, the evaluation re-run used CPU.
- Evaluate both models and reproduce the table:

```bash
python scripts/evaluate.py --data-dir go-data-science-5-0 --checkpoint models/bert_best_model.pt
```

BERT inference on CPU for the 5,324 validation rows takes about 37 minutes. Results go to `eval_out/results.json`.

`test_bert_modules.py` is a smoke test for the dataset, model and training utilities.

## Repository layout

```
hackathon/
  src/                  config, preprocessing, data loader, BERT classifier, training utils, metrics
  notebooks/            02_baseline_model, 03_bert_model
  scripts/evaluate.py   re-run both models on the validation split
  analyze_data.py       quick label statistics
  fix_submission.py     submission file formatting
  test_bert_modules.py  smoke test
  go-data-science-5-0/  sample_submission.csv only (competition data not included)
  requirements.txt
```
