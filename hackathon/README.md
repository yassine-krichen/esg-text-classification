# 🏆 ESG Text Classification - Hackathon Project

**Competition**: go-data-science-5-0  
**Goal**: Multi-label classification of corporate ESG text  
**Target**: 🥇 First Place + Deep Learning Mastery

---

## 📋 Project Overview

This project classifies corporate text into ESG categories:

- **E** (Environmental) - Climate, carbon emissions, resource efficiency
- **S** (Social) - Employee welfare, diversity, community impact
- **G** (Governance) - Board structure, compliance, ethics
- **non_ESG** - General corporate information

**Challenge**: Multi-label classification with severe class imbalance (E: 4.41%, S: 37%, G: 33%, non_ESG: 43%)

---

## 🗂️ Project Structure

```
hackathon/
├── go-data-science-5-0/       # Competition data
│   ├── train.csv              # Training data (26,750 samples)
│   ├── test.csv               # Test data (2,000 samples)
│   └── sample_submission.csv  # Submission format
│
├── notebooks/                 # Jupyter notebooks
│   ├── 01_eda.ipynb          # (To create) Exploratory analysis
│   └── 02_baseline_model.ipynb # ✅ TF-IDF baseline
│
├── src/                       # Source code
│   ├── config.py             # ✅ Configuration settings
│   ├── preprocessing.py      # ✅ Text preprocessing utils
│   ├── evaluation.py         # ✅ Metrics & evaluation
│   └── models/               # Model implementations (to create)
│
├── models/                    # Saved model checkpoints
├── submissions/              # Kaggle submission files
├── app/                      # Streamlit demo app (Phase 5)
│
├── ROADMAP.md               # ✅ Complete strategy document
├── requirements.txt         # ✅ Python dependencies
├── README.md               # ✅ This file
└── analyze_data.py         # ✅ Quick data analysis script
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd hackathon
pip install -r requirements.txt
```

### 2. Run Baseline Model

```bash
# Option 1: Open Jupyter notebook
jupyter notebook notebooks/02_baseline_model.ipynb

# Option 2: Convert to Python and run
# jupyter nbconvert --to script notebooks/02_baseline_model.ipynb
# python notebooks/02_baseline_model.py
```

### 3. Submit to Kaggle

1. Find file: `submissions/baseline_tfidf_lr.csv`
2. Upload to Kaggle competition page
3. See your score on public leaderboard!

---

## 📈 Development Roadmap

### ✅ Phase 1: Foundations (CURRENT)

- [x] Data exploration & analysis
- [x] Project structure setup
- [x] Preprocessing pipeline
- [x] Evaluation framework
- [x] TF-IDF baseline model
- [ ] **TODO**: Run baseline notebook & submit

**Expected Score**: 0.70-0.75

### 🔄 Phase 2: BERT Basics (NEXT)

- [ ] BERT tokenization & data loading
- [ ] Simple BERT fine-tuning
- [ ] Multi-label classification head
- [ ] Training & validation loops
- [ ] Hyperparameter tuning

**Expected Score**: 0.80-0.85

### 📅 Phase 3: Advanced Optimizations

- [ ] Class imbalance handling (weighted loss, focal loss)
- [ ] Long text strategies (chunking, hierarchical)
- [ ] Data augmentation (back-translation, synonyms)
- [ ] Model variants (RoBERTa, DeBERTa)
- [ ] Advanced training (mixed precision, gradient accumulation)

**Expected Score**: 0.85-0.90

### 📅 Phase 4: Ensemble & Winning Model

- [ ] Multi-model ensemble
- [ ] Pseudo-labeling (semi-supervised)
- [ ] Test-time augmentation
- [ ] Threshold optimization
- [ ] Final model selection

**Expected Score**: 0.90+ (Top 3)

### 📅 Phase 5: Real-World Demo

- [ ] Streamlit web application
- [ ] User-friendly interface
- [ ] Real-world use cases
- [ ] Presentation materials

---

## 🛠️ Key Technologies

| Component       | Technology                           |
| --------------- | ------------------------------------ |
| Deep Learning   | PyTorch, Transformers (Hugging Face) |
| NLP             | BERT, RoBERTa, Tokenizers            |
| ML Baseline     | scikit-learn, TF-IDF                 |
| Data Processing | pandas, numpy                        |
| Evaluation      | sklearn.metrics                      |
| Visualization   | matplotlib, seaborn                  |
| Deployment      | Streamlit                            |

---

## 📊 Current Status

| Metric                   | Value               |
| ------------------------ | ------------------- |
| **Phase**                | 1 - Foundations     |
| **Models Built**         | 1 (TF-IDF Baseline) |
| **Best Val F1**          | TBD (run notebook)  |
| **Kaggle Score**         | Not submitted yet   |
| **Leaderboard Position** | N/A                 |

---

## 🎓 Key Learnings

### Data Insights

- **26,750 training samples**, 2,000 test samples
- **Severe class imbalance**: E (4.41%), S (37%), G (33%), non_ESG (43%)
- **15.6% multi-label samples** (multiple categories per text)
- **Average text length**: 207 characters (range: 33-5,630)

### Challenge Areas

1. **Rare 'E' class** - Only 1,180 positive samples
2. **Multi-label complexity** - 200 samples have all 4 labels
3. **Long texts** - 512 token limit for BERT requires chunking
4. **Overfitting risk** - Small test set (2,000 samples)

---

## 📖 Learning Resources

### From Course Materials

- `06_nlp.ipynb` - BERT tokenization & question answering
- `07_assessment.ipynb` - Transfer learning & fine-tuning
- `05b_presidential_doggy_door.ipynb` - Data augmentation
- `utils.py` - Training & validation loops

### External Resources

- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [Multi-label Classification Guide](https://scikit-learn.org/stable/modules/multiclass.html)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)

---

## 🎯 Next Steps

### Immediate Actions:

1. **Run baseline notebook**: `notebooks/02_baseline_model.ipynb`
2. **Make first submission**: Upload `submissions/baseline_tfidf_lr.csv`
3. **Check leaderboard**: See where we stand
4. **Start Phase 2**: Begin BERT implementation

### This Week:

- Complete TF-IDF baseline (Today)
- Implement BERT model (Day 2-3)
- Optimize & experiment (Day 4-5)
- Build ensemble (Day 6-7)

---

## 🐛 Troubleshooting

### Common Issues

**ImportError for iterative-stratification:**

```bash
pip install iterative-stratification
```

**Memory issues with large models:**

- Reduce batch size in config
- Use gradient accumulation
- Enable mixed precision training (FP16)

**Slow training:**

- Ensure using GPU (`torch.cuda.is_available()`)
- Reduce max_length for shorter sequences
- Use DistilBERT instead of BERT

---

## 🤝 Competition Rules Compliance

✅ **Allowed:**

- Pretrained classical models (BERT, RoBERTa, etc.)
- Public datasets for pretraining (ImageNet, Wikipedia)
- sklearn, PyTorch, Transformers libraries

❌ **Prohibited:**

- External ESG datasets
- LLMs (GPT-4, Claude, Gemini)
- AutoML tools
- External help/forums

---

## 📝 Submission Guidelines

### File Format

```csv
id,E,S,G,non_ESG
0,0.1234,0.8765,0.2345,0.0987
1,0.0012,0.0034,0.9876,0.0123
...
```

### Requirements

- **2,000 rows** (one per test sample)
- **5 columns**: id, E, S, G, non_ESG
- **Values**: Probabilities [0, 1]
- **Total submissions**: Max 40 (2 final for scoring)

---

## 📧 Contact & Notes

**Competition**: go-data-science-5-0  
**Start**: 8:00 PM  
**End**: 10:00 AM

**Verification Requirements**:

- Code reproducibility
- Technical presentation (Top 10)
- Live demo (Top 5)

---

## 🏆 Success Metrics

| Goal                | Target               |
| ------------------- | -------------------- |
| Phase 1 Complete    | F1 > 0.70            |
| Phase 2 Complete    | F1 > 0.80            |
| Phase 3 Complete    | F1 > 0.85            |
| **Win Competition** | **F1 > 0.90, Top 3** |
| Learn Deep Learning | ✅ Comprehensive     |

---

**Last Updated**: February 14, 2026  
**Current Phase**: 1 - Foundations  
**Next Milestone**: First Kaggle submission

🚀 Let's win this! 🏆
