"""
Quick test script for BERT modules

Run this before training to verify everything works.
"""
import sys
sys.path.append('..')

import torch
from transformers import BertTokenizer
import pandas as pd
import numpy as np

print("="*70)
print("🧪 TESTING BERT MODULES")
print("="*70)

# Test 1: Check imports
print("\n1️⃣  Testing imports...")
try:
    from src.data_loader import ESGDataset, create_data_loaders
    from src.models.bert_classifier import BERTClassifier, get_model
    from src.train_utils import train_epoch, validate_epoch, predict
    print("   ✅ All imports successful!")
except ImportError as e:
    print(f"   ❌ Import error: {e}")
    sys.exit(1)

# Test 2: Check device
print("\n2️⃣  Testing device...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"   Device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
else:
    print("   ⚠️  No GPU - training will be slow")

# Test 3: Test tokenizer
print("\n3️⃣  Testing tokenizer...")
try:
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    test_text = "The company reduced carbon emissions by 20%."
    encoding = tokenizer(
        test_text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    print(f"   ✅ Tokenizer works!")
    print(f"   Input IDs shape: {encoding['input_ids'].shape}")
except Exception as e:
    print(f"   ❌ Tokenizer error: {e}")
    sys.exit(1)

# Test 4: Test dataset
print("\n4️⃣  Testing ESGDataset...")
try:
    texts = ["Sample text 1", "Sample text 2", "Sample text 3"]
    labels = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    
    dataset = ESGDataset(texts, labels, tokenizer, max_length=128)
    sample = dataset[0]
    
    print(f"   ✅ Dataset works!")
    print(f"   Sample keys: {sample.keys()}")
    print(f"   Shapes: input_ids={sample['input_ids'].shape}, "
          f"labels={sample['labels'].shape}")
except Exception as e:
    print(f"   ❌ Dataset error: {e}")
    sys.exit(1)

# Test 5: Test model
print("\n5️⃣  Testing BERT model...")
try:
    model = get_model(
        model_type='bert',
        n_classes=4,
        freeze_bert=False,
        device=device
    )
    
    # Test forward pass
    batch_size = 2
    seq_length = 128
    input_ids = torch.randint(0, 30522, (batch_size, seq_length)).to(device)
    attention_mask = torch.ones(batch_size, seq_length).to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
    
    print(f"   ✅ Model works!")
    print(f"   Output shape: {outputs.shape} (expected: ({batch_size}, 4))")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Parameters: {total_params:,} total, {trainable_params:,} trainable")
    
except Exception as e:
    print(f"   ❌ Model error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Test data loader
print("\n6️⃣  Testing data loader...")
try:
    # Create sample dataframe
    df = pd.DataFrame({
        'text': ["Sample " + str(i) for i in range(100)],
        'E': np.random.randint(0, 2, 100),
        'S': np.random.randint(0, 2, 100),
        'G': np.random.randint(0, 2, 100),
        'non_ESG': np.random.randint(0, 2, 100)
    })
    
    from torch.utils.data import DataLoader
    dataset = ESGDataset(
        df['text'].tolist(),
        df[['E', 'S', 'G', 'non_ESG']].values,
        tokenizer,
        max_length=128
    )
    
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch = next(iter(loader))
    
    print(f"   ✅ DataLoader works!")
    print(f"   Batch keys: {batch.keys()}")
    print(f"   Batch shapes: input_ids={batch['input_ids'].shape}, "
          f"labels={batch['labels'].shape}")
    
except Exception as e:
    print(f"   ❌ DataLoader error: {e}")
    sys.exit(1)

# All tests passed!
print("\n" + "="*70)
print("🎉 ALL TESTS PASSED!")
print("="*70)
print("\n✅ Your BERT modules are ready to use!")
print("\nNext steps:")
print("   1. Open: notebooks/03_bert_model.ipynb")
print("   2. Run all cells")
print("   3. Wait ~10-15 minutes for training")
print("   4. Upload submission to Kaggle")
print("\nHappy training! 🚀")
