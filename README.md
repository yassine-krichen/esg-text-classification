# 🏆 ESG Text Classification Hackathon

> **Competition**: Multi-label classification of corporate ESG (Environmental, Social, Governance) text  
> **Goal**: Build a winning deep learning model and master NLP fundamentals  
> **Status**: 🔥 Phase 1 Complete - Baseline Model Submitted

---

## 📊 Problem Overview

Classify corporate text into 4 ESG categories (multi-label):

- **E** (Environmental) - Climate action, carbon emissions, resource efficiency
- **S** (Social) - Employee welfare, diversity, community impact
- **G** (Governance) - Board structure, compliance, ethics, transparency
- **non_ESG** - General corporate information

**Dataset**: 26,750 training samples, 2,000 test samples  
**Challenge**: Severe class imbalance (E: 4.41% vs non_ESG: 42.58%)

---

## 🚀 Quick Start

```bash
# Clone repository
git clone <your-repo-url>
cd my_project/hackathon

# Setup environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run baseline model
jupyter notebook notebooks/02_baseline_model.ipynb

# Generate submission
python fix_submission.py
# Upload: submissions/baseline_tfidf_lr_binary.csv
```

---

## 📁 Repository Structure

```
my_project/
├── hackathon/                    # Main competition project
│   ├── notebooks/                # Jupyter notebooks
│   │   └── 02_baseline_model.ipynb
│   ├── src/                      # Reusable Python modules
│   │   ├── config.py            # Configuration
│   │   ├── preprocessing.py     # Text preprocessing
│   │   └── evaluation.py        # Metrics & evaluation
│   ├── go-data-science-5-0/     # Competition data (not in git)
│   ├── submissions/             # Kaggle submission files
│   ├── models/                  # Model checkpoints
│   ├── ROADMAP.md              # Complete strategy (5 phases)
│   ├── README.md               # Competition details
│   └── requirements.txt        # Dependencies
│
├── deep_learning_fundementals/  # Course materials
│   └── (NVIDIA DLI notebooks)
│
└── .gitignore                   # Git ignore rules
```

---

## 📈 Development Progress

### ✅ Phase 1: Foundations & Baseline (COMPLETE)

- [x] Data exploration & analysis
- [x] Text preprocessing pipeline
- [x] TF-IDF + Logistic Regression baseline
- [x] Evaluation framework
- [x] First Kaggle submission

**Results**:

- Model: TF-IDF + Logistic Regression
- Training time: <1 minute
- Validation Macro F1: ~0.75
- Kaggle Score: TBD

### 🔄 Phase 2: BERT Model (NEXT)

- [ ] BERT tokenization & data loading
- [ ] Fine-tuning architecture
- [ ] Multi-label classification head
- [ ] Training & validation loops
- [ ] Hyperparameter tuning

**Target**: F1 > 0.80 (+10-15% improvement)

### 📅 Phase 3: Advanced Optimizations

- [ ] Class imbalance handling (weighted loss, focal loss)
- [ ] Long text strategies (chunking, hierarchical models)
- [ ] Data augmentation (back-translation, synonym replacement)
- [ ] Model variants (RoBERTa, DeBERTa)

**Target**: F1 > 0.85

### 📅 Phase 4: Ensemble & Winning Model

- [ ] Multi-model ensemble
- [ ] Pseudo-labeling (semi-supervised)
- [ ] Test-time augmentation
- [ ] Threshold optimization

**Target**: F1 > 0.90, Top 3 position

### 📅 Phase 5: Real-World Demo

- [ ] Streamlit web application
- [ ] Real-time ESG classification
- [ ] Business use cases

---

## 🛠️ Tech Stack

| Component           | Technology                           |
| ------------------- | ------------------------------------ |
| **Deep Learning**   | PyTorch, Transformers (Hugging Face) |
| **NLP**             | BERT, RoBERTa, Tokenizers            |
| **ML Baseline**     | scikit-learn, TF-IDF                 |
| **Data Processing** | pandas, numpy                        |
| **Visualization**   | matplotlib, seaborn                  |
| **Deployment**      | Streamlit, Gradio                    |

---

## 🎓 Key Learnings

### Data Insights

- Multi-label complexity: 15.6% of samples have multiple labels
- 200 samples have ALL 4 labels simultaneously
- Text length varies: 33-5,630 characters (median: 176)
- Label distribution: S (37%), G (33%), non_ESG (43%), E (4.41%)

### Technical Challenges

1. **Rare 'E' class** - Only 1,180 positive samples (4.41%)
2. **Multi-label classification** - Not mutually exclusive categories
3. **Long sequences** - BERT's 512 token limit requires chunking
4. **Imbalanced evaluation** - Need per-class metrics, not just accuracy

### Solutions Applied

- Stratified multi-label train/validation split
- Class-balanced Logistic Regression
- Binary Cross Entropy loss for multi-label
- Per-class F1, precision, recall metrics

---

## 📊 Model Performance

| Model              | Macro F1 (Val) | E F1 | S F1 | G F1 | non_ESG F1 | Training Time |
| ------------------ | -------------- | ---- | ---- | ---- | ---------- | ------------- |
| TF-IDF + LR        | ~0.75          | Low  | High | High | High       | <1 min        |
| BERT (planned)     | ~0.80          | TBD  | TBD  | TBD  | TBD        | ~15 min       |
| Ensemble (planned) | ~0.90+         | TBD  | TBD  | TBD  | TBD        | ~1 hour       |

---

## 📖 Course Materials Applied

This project applies concepts from the NVIDIA Deep Learning Fundamentals course:

| Notebook                              | Concept Applied               | Our Implementation   |
| ------------------------------------- | ----------------------------- | -------------------- |
| **06_nlp.ipynb**                      | BERT tokenization, embeddings | Text classification  |
| **07_assessment.ipynb**               | Transfer learning, VGG16      | BERT fine-tuning     |
| **05b_presidential_doggy_door.ipynb** | Data augmentation             | Text augmentation    |
| **utils.py**                          | Training loops                | Multi-label training |

---

## 🎯 Competition Rules Compliance

✅ **Allowed**:

- Pretrained models (BERT, RoBERTa, etc.)
- Public libraries (PyTorch, Transformers, sklearn)
- Data augmentation techniques

❌ **Prohibited**:

- External ESG datasets
- LLMs (GPT-4, Claude, Gemini)
- AutoML tools
- External forums/help

---

## 🤝 Contributing

This is a hackathon competition project. Team collaboration guidelines:

1. **Code Quality**: Follow PEP 8, add docstrings
2. **Notebooks**: Clear markdown explanations, reproducible
3. **Commits**: Descriptive messages (see below)
4. **Experiments**: Log all hyperparameters and results

### Commit Message Format

```
[Phase X] Brief description

- Detailed change 1
- Detailed change 2

Results: Metric improvements
```

Example:

```
[Phase 1] Add TF-IDF baseline model

- Implement text preprocessing pipeline
- Create stratified multi-label split
- Train Logistic Regression baseline
- Add evaluation metrics

Results: Val Macro F1 = 0.75
```

---

## 📝 Submission Checklist

Before Kaggle submission:

- [ ] Binary predictions (0 or 1), not probabilities
- [ ] Correct format: id, E, S, G, non_ESG
- [ ] 2,000 rows (one per test sample)
- [ ] No NaN values
- [ ] File saved in `submissions/` directory

---

## 🏆 Goals

- **Primary**: Win 1st place in the hackathon
- **Learning**: Master deep learning for NLP
- **Impact**: Build production-ready ESG classifier
- **Community**: Share knowledge with team

---

## 📧 Contact

**Team**: [Your Team Name]  
**Competition**: go-data-science-5-0  
**Timeline**: Start: 8:00 PM, End: 10:00 AM

---

## 🙏 Acknowledgments

- NVIDIA Deep Learning Institute for foundational course
- Hugging Face for Transformers library
- scikit-learn for baseline utilities
- Kaggle for hosting the competition

---

**Last Updated**: February 14, 2026  
**Current Phase**: 1 → 2 (Transitioning to BERT)  
**Next Milestone**: BERT model submission, F1 > 0.80

🚀 Let's win this! 🏆
