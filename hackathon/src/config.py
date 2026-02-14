"""
Configuration file for ESG Classification project
"""
import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "go-data-science-5-0"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
MODELS_DIR = PROJECT_ROOT / "models"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"

# Create directories if they don't exist
MODELS_DIR.mkdir(exist_ok=True)
SUBMISSIONS_DIR.mkdir(exist_ok=True)
NOTEBOOKS_DIR.mkdir(exist_ok=True)

# Data paths
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"

# Label columns
LABEL_COLUMNS = ['E', 'S', 'G', 'non_ESG']
TEXT_COLUMN = 'text'
ID_COLUMN = 'id'

# Model hyperparameters (will be tuned)
RANDOM_SEED = 42
VALIDATION_SPLIT = 0.2
N_FOLDS = 5

# BERT configurations
BERT_MODEL_NAME = 'bert-base-uncased'
MAX_LENGTH = 512  # BERT max sequence length
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
WARMUP_STEPS = 100

# Device configuration
import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Configuration loaded. Using device: {DEVICE}")
