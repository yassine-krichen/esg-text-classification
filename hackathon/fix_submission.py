import pandas as pd
import numpy as np

# Read current submission
df = pd.read_csv('submissions/baseline_tfidf_lr.csv')

print('Current submission (probabilities):')
print(df.head(10))
print('\nValue ranges:')
for col in ['E', 'S', 'G', 'non_ESG']:
    print(f'{col}: [{df[col].min():.4f}, {df[col].max():.4f}]')

# Convert to binary predictions (threshold = 0.5)
print('\n' + '='*60)
print('Converting to BINARY predictions (0 or 1)...')
print('='*60)

df_binary = df.copy()
for col in ['E', 'S', 'G', 'non_ESG']:
    df_binary[col] = (df[col] >= 0.5).astype(int)

print('\nNew submission (binary):')
print(df_binary.head(10))
print('\nValue ranges (should be 0 or 1):')
for col in ['E', 'S', 'G', 'non_ESG']:
    print(f'{col}: [{df_binary[col].min()}, {df_binary[col].max()}] - Unique: {df_binary[col].unique()}')

# Check distribution
print('\nLabel distribution in predictions:')
for col in ['E', 'S', 'G', 'non_ESG']:
    count = df_binary[col].sum()
    pct = (count / len(df_binary)) * 100
    print(f'{col:10s}: {count:4d} / {len(df_binary)} ({pct:5.2f}%)')

# Save corrected submission
df_binary.to_csv('submissions/baseline_tfidf_lr_binary.csv', index=False)
print('\n✅ Saved corrected submission: submissions/baseline_tfidf_lr_binary.csv')
print('🚀 Upload this file to Kaggle!')
