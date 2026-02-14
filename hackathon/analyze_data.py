import pandas as pd
import numpy as np

# Load data
train = pd.read_csv('go-data-science-5-0/train.csv')
test = pd.read_csv('go-data-science-5-0/test.csv')
sample_sub = pd.read_csv('go-data-science-5-0/sample_submission.csv')

print('=' * 60)
print('TRAIN DATA OVERVIEW')
print('=' * 60)
print(f'Shape: {train.shape}')
print(f'Columns: {train.columns.tolist()}')

print('\n' + '=' * 60)
print('LABEL DISTRIBUTION')
print('=' * 60)
label_cols = ['E', 'S', 'G', 'non_ESG']
for col in label_cols:
    count = train[col].sum()
    pct = (count / len(train)) * 100
    print(f'{col:10s}: {count:6d} ({pct:5.2f}%)')

print(f'\nTotal labels: {train[label_cols].sum().sum()}')

print('\n' + '=' * 60)
print('MULTI-LABEL STATISTICS')
print('=' * 60)
labels_per_sample = train[label_cols].sum(axis=1)
print(f'Samples with 0 labels: {(labels_per_sample == 0).sum()}')
print(f'Samples with 1 label:  {(labels_per_sample == 1).sum()}')
print(f'Samples with 2 labels: {(labels_per_sample == 2).sum()}')
print(f'Samples with 3 labels: {(labels_per_sample == 3).sum()}')
print(f'Samples with 4 labels: {(labels_per_sample == 4).sum()}')

print('\n' + '=' * 60)
print('LABEL COMBINATIONS (Top 10)')
print('=' * 60)
combinations = train[label_cols].value_counts().head(10)
for idx, (combo, count) in enumerate(combinations.items(), 1):
    combo_str = ' '.join([label_cols[i] for i, val in enumerate(combo) if val == 1])
    if not combo_str:
        combo_str = 'NONE'
    print(f'{idx:2d}. {combo_str:20s}: {count:5d} samples')

print('\n' + '=' * 60)
print('TEXT LENGTH STATISTICS')
print('=' * 60)
train['text_length'] = train['text'].str.len()
print(train['text_length'].describe())

print('\n' + '=' * 60)
print('SAMPLE TEXTS')
print('=' * 60)
for i in [0, 5, 10, 15, 20]:
    row = train.iloc[i]
    labels = [col for col in label_cols if row[col] == 1]
    print(f"\nID {row['id']}: Labels = {labels}")
    print(f"Text: {row['text'][:200]}...")

print('\n' + '=' * 60)
print('TEST DATA')
print('=' * 60)
print(f'Test shape: {test.shape}')
print(f'Test columns: {test.columns.tolist()}')
print(f'\nFirst test sample:')
print(test.iloc[0]['text'][:300])
