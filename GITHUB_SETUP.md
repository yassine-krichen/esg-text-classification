# 🚀 GitHub Setup Guide

## Quick Commands

### First Time Setup

```bash
cd C:\Users\user\Downloads\dp\my_project

# Configure git (use your actual details)
git config user.name "Your Name"
git config user.email "your.email@example.com"

# Check status
git status

# Add files
git add .

# First commit
git commit -m "[Phase 1] Initial commit - Baseline model complete

- Project structure with modular code
- TF-IDF + Logistic Regression baseline
- Text preprocessing pipeline
- Evaluation metrics framework
- ROADMAP for 5-phase strategy
- Fixed submission format (binary predictions)

Results: Ready for first Kaggle submission"
```

### Create GitHub Repository

1. Go to https://github.com/new
2. **Repository name suggestions**:
    - `esg-text-classification`
    - `esg-nlp-hackathon`
    - `deep-learning-esg-classifier`
    - `hackathon-esg-multiLabel`
    - `corporate-esg-classification`

3. **Description**:

    ```
    🏆 Multi-label ESG text classification using BERT & deep learning.
    From TF-IDF baseline to ensemble models. Course project + hackathon entry.
    ```

4. **Settings**:
    - ✅ Public (to share with friends)
    - ❌ Don't initialize with README (we have one)
    - ❌ Don't add .gitignore (we have one)

5. **Connect and push**:
    ```bash
    git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git
    git branch -M main
    git push -u origin main
    ```

### Daily Workflow

```bash
# Check what changed
git status

# Stage changes
git add <specific-files>
# OR add all
git add .

# Commit with message
git commit -m "[Phase X] Description"

# Push to GitHub
git push
```

---

## 📝 Recommended Commit Messages

### Format

```
[Phase X] Brief title (50 chars max)

- Detailed bullet point 1
- Detailed bullet point 2
- What changed and why

Results: Metrics/outcomes if applicable
```

### Examples

**Phase 1** (Current):

```
[Phase 1] Fix submission format to binary predictions

- Convert probabilities to 0/1 using threshold 0.5
- Add fix_submission.py script
- Update notebook with binary conversion

Results: Ready for Kaggle submission
```

**Phase 2** (Next):

```
[Phase 2] Implement BERT baseline model

- Add BERT tokenization with 512 max length
- Create custom PyTorch dataset class
- Implement binary cross-entropy loss
- Add training and validation loops

Results: Val F1 = 0.82 (+0.07 from baseline)
```

**Phase 3**:

```
[Phase 3] Add focal loss for class imbalance

- Implement focal loss for rare 'E' class
- Add class weights based on distribution
- Tune gamma parameter (γ=2.0)

Results: E F1 improved from 0.45 to 0.68
```

---

## 🌟 Repository Best Practices

### What to Commit ✅

- Source code (`.py` files)
- Notebooks (`.ipynb`)
- Documentation (`.md` files)
- Configuration (`requirements.txt`, `config.py`)
- Small submission files (< 1MB)
- Scripts and utilities

### What NOT to Commit ❌

- Large datasets (`.csv` files > 10MB) → Already in `.gitignore`
- Model checkpoints (`.pt`, `.pth`, `.h5`) → Use Git LFS or cloud storage
- Virtual environments (`.venv/`, `venv/`)
- Kaggle credentials (`.kaggle/`)
- IDE settings (`.vscode/`, `.idea/`)
- Temporary files (`__pycache__/`, `.ipynb_checkpoints/`)

---

## 🔒 Security Tips

### Never commit:

1. **API Keys**: Kaggle credentials, Hugging Face tokens
2. **Personal Info**: Email, phone, addresses in code
3. **Large Files**: >100MB (GitHub will block)

### If you accidentally commit secrets:

```bash
# Remove from history (careful!)
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch path/to/secret-file" \
  --prune-empty --tag-name-filter cat -- --all

# Force push (this rewrites history!)
git push origin --force --all
```

Better: Use `.env` files (already in `.gitignore`)

---

## 📊 GitHub Features to Use

### README Badges

Add to your README.md:

```markdown
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-in%20progress-yellow.svg)
```

### Project Board

Track progress: `Your Repo > Projects > New Project`

- Todo, In Progress, Done columns
- Link issues to commits

### Issues

Document bugs or ideas: `Your Repo > Issues > New Issue`

- Label: `bug`, `enhancement`, `documentation`

---

## 💡 Collaboration Tips

### Sharing with Friends

1. **Add as collaborators**:
    - `Settings > Collaborators > Add people`

2. **Or share read-only**:
    - Just share the URL (public repo)

3. **Fork workflow**:
    - Friends fork your repo
    - Make changes in their fork
    - Submit pull requests
    - You review and merge

### Code Review Checklist

- [ ] Code runs without errors
- [ ] Follows PEP 8 style
- [ ] Has docstrings
- [ ] No hardcoded paths
- [ ] No sensitive data
- [ ] README updated if needed

---

## 🎯 Repository Naming Suggestions

Ranked by professionalism and clarity:

1. **esg-text-classification** ⭐ (Clear, professional)
2. **deep-learning-esg-nlp** (Technical, descriptive)
3. **corporate-esg-classifier** (Business-focused)
4. **hackathon-esg-bert** (Project context)
5. **multilabel-esg-classification** (Technique-focused)
6. **bert-esg-prediction** (Model-focused)
7. **esg-nlp-competition** (Competition-focused)

Choose based on:

- **Portfolio**: Use #1-3 (professional)
- **Learning**: Use #4-5 (shows project context)
- **Technical**: Use #6-7 (highlights techniques)

---

## 📚 Useful Git Commands

```bash
# View commit history
git log --oneline --graph

# Undo last commit (keep changes)
git reset --soft HEAD~1

# Discard all local changes
git reset --hard HEAD

# Create branch for experiments
git checkout -b experiment-roberta

# Switch back to main
git checkout main

# Merge experiment to main
git merge experiment-roberta

# Delete branch
git branch -d experiment-roberta

# See what changed in a file
git diff path/to/file.py

# Stash changes temporarily
git stash
git stash pop
```

---

Ready to push to GitHub! 🚀

Next steps:

1. Run the commands above to make first commit
2. Create GitHub repo
3. Connect and push
4. Share with friends!
