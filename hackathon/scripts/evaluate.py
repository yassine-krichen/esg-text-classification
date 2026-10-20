"""Evaluate a saved BERT checkpoint against the TF-IDF + logistic regression
baseline on the project's validation split.

Usage (from the hackathon/ directory):
    python scripts/evaluate.py --data-dir <dir with train.csv>         --checkpoint <path to bert_best_model.pt> [--out-dir eval_out]

The competition data (train.csv) is NOT included in this repository.
Download it from the GODS 5.0 competition page and pass its folder as
--data-dir. Model weights are not included either; train them with
notebooks/03_bert_model.ipynb.

Writes validation predictions and results.json to --out-dir. BERT inference
is cached in bert_proba.npy; delete it to force a re-run.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multioutput import MultiOutputClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import src.preprocessing as prep  # noqa: E402

parser = argparse.ArgumentParser(description='Evaluate BERT vs TF-IDF baseline.')
parser.add_argument('--data-dir', required=True, type=Path,
                    help='folder containing train.csv')
parser.add_argument('--checkpoint', required=True, type=Path,
                    help='BERT checkpoint saved by the training notebook')
parser.add_argument('--out-dir', type=Path, default=Path('eval_out'))
args = parser.parse_args()

LABELS = ['E', 'S', 'G', 'non_ESG']
OUT = args.out_dir
OUT.mkdir(exist_ok=True)

train = pd.read_csv(args.data_dir / 'train.csv')
train_clean = prep.preprocess_dataframe(train.copy())
train_df, val_df = prep.stratified_split_multilabel(
    train_clean, LABELS, test_size=0.2, random_state=42)
print('rows raw/clean', len(train), len(train_clean), 'train/val', len(train_df), len(val_df))
val_df.to_csv(OUT / 'val_split.csv', index=False)
y_val = val_df[LABELS].values
y_tr = train_df[LABELS].values


def report(name, y, pred):
    r = {
        'macro_f1': f1_score(y, pred, average='macro', zero_division=0),
        'micro_f1': f1_score(y, pred, average='micro', zero_division=0),
    }
    per = f1_score(y, pred, average=None, zero_division=0)
    for l, v in zip(LABELS, per):
        r[f'{l}_f1'] = v
    print(name, json.dumps({k: round(v, 4) for k, v in r.items()}))
    return r


results = {}

# Baseline: exact config from 02_baseline_model.ipynb
vec = TfidfVectorizer(max_features=10000, min_df=2, max_df=0.9,
                      ngram_range=(1, 2), stop_words='english')
Xtr = vec.fit_transform(train_df['text'])
Xva = vec.transform(val_df['text'])
base = MultiOutputClassifier(LogisticRegression(
    max_iter=1000, random_state=42, class_weight='balanced', C=1.0))
base.fit(Xtr, y_tr)
base_proba = np.array([c.predict_proba(Xva)[:, 1] for c in base.estimators_]).T
results['baseline'] = report('baseline', y_val, base.predict(Xva))
np.save(OUT / 'base_proba.npy', base_proba)

# BERT
proba_path = OUT / 'bert_proba.npy'
if not proba_path.exists():
    from transformers import BertTokenizer
    from torch.utils.data import DataLoader
    from src.data_loader import ESGDataset
    from src.models import get_model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device', device, 'threads', torch.get_num_threads())
    tok = BertTokenizer.from_pretrained('bert-base-uncased')
    model = get_model('bert', 4, 'bert-base-uncased', 0.3, False, device)
    ck = torch.load(args.checkpoint, map_location=device,
                    weights_only=False)
    print('ckpt keys', list(ck.keys()), 'epoch', ck.get('epoch'), 'val_f1', ck.get('val_f1'))
    res = model.load_state_dict(ck['model_state_dict'])
    print('load_state_dict', res)
    model.eval()
    ds = ESGDataset(val_df['text'].tolist(), y_val, tok, 256)
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    outs = []
    t0 = time.time()
    with torch.no_grad():
        for i, b in enumerate(dl):
            lo = model(b['input_ids'].to(device), b['attention_mask'].to(device))
            outs.append(torch.sigmoid(lo).cpu().numpy())
            if i % 10 == 0:
                print(i, len(dl), round(time.time() - t0), 's', flush=True)
    np.save(proba_path, np.vstack(outs))
bp = np.load(proba_path)
results['bert@0.5'] = report('bert@0.5', y_val, (bp > 0.5).astype(int))

# Threshold tuning with 2-fold cross-fitting on the validation set (tune on one half, score on other)
rng = np.random.RandomState(0)
idx = rng.permutation(len(y_val))
halves = [idx[: len(idx) // 2], idx[len(idx) // 2:]]
grid = np.arange(0.05, 0.96, 0.05)


def tune(p, y):
    return np.array([grid[np.argmax([f1_score(y[:, j], p[:, j] > t, zero_division=0)
                                     for t in grid])] for j in range(4)])


def crossfit(p):
    pred = np.zeros_like(y_val)
    ths = []
    for a, b in [(0, 1), (1, 0)]:
        th = tune(p[halves[a]], y_val[halves[a]])
        ths.append(th.round(2).tolist())
        pred[halves[b]] = (p[halves[b]] > th).astype(int)
    return pred, ths


for name, p in [('bert', bp), ('baseline', base_proba)]:
    pred, ths = crossfit(p)
    results[f'{name}@tuned_crossfit'] = report(f'{name}@tuned_crossfit', y_val, pred)
    results[f'{name}@tuned_crossfit']['thresholds'] = ths

json.dump(results, open(OUT / 'results.json', 'w'), indent=1, default=float)
