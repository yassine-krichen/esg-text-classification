# Quick Git Setup Script
# Run from: C:\Users\user\Downloads\dp\my_project

Write-Host "🚀 ESG Hackathon - Git Setup" -ForegroundColor Cyan
Write-Host "=" * 50

# Configure git (update with your details!)
Write-Host "`n📝 Step 1: Configure Git Identity" -ForegroundColor Yellow
Write-Host "Current config:"
git config user.name
git config user.email

$name = Read-Host "`nEnter your name (or press Enter to skip)"
if ($name) { git config user.name $name }

$email = Read-Host "Enter your email (or press Enter to skip)"
if ($email) { git config user.email $email }

# Show status
Write-Host "`n📊 Step 2: Current Status" -ForegroundColor Yellow
git status

# Add all files
Write-Host "`n➕ Step 3: Adding files to git..." -ForegroundColor Yellow
git add .

# Show what will be committed
Write-Host "`n📋 Files to be committed:" -ForegroundColor Yellow
git status --short

# Commit
Write-Host "`n💾 Step 4: Creating first commit..." -ForegroundColor Yellow
git commit -m "[Phase 1] Initial commit - ESG classification baseline

- Complete project structure with modular code
- TF-IDF + Logistic Regression baseline model  
- Text preprocessing pipeline (cleaning, stratified split)
- Comprehensive evaluation metrics framework
- 5-phase ROADMAP from baseline to winning model
- Fixed submission format (probabilities → binary)
- Documentation: README, ROADMAP, GITHUB_SETUP

Files included:
- Source code: src/config.py, preprocessing.py, evaluation.py
- Notebooks: 02_baseline_model.ipynb
- Scripts: analyze_data.py, fix_submission.py
- Documentation: README.md, ROADMAP.md, GITHUB_SETUP.md
- Config: requirements.txt, .gitignore

Next: Create GitHub repo and push
Results: Baseline ready for Kaggle submission (Val F1 ~0.75)"

Write-Host "`n✅ Commit created!" -ForegroundColor Green

# Show log
Write-Host "`n📜 Commit History:" -ForegroundColor Yellow
git log --oneline

Write-Host "`n🎯 Next Steps:" -ForegroundColor Cyan
Write-Host "1. Create GitHub repository at: https://github.com/new"
Write-Host "   Suggested names:"
Write-Host "   - esg-text-classification"
Write-Host "   - deep-learning-esg-nlp"
Write-Host "   - corporate-esg-classifier"
Write-Host ""
Write-Host "2. Connect and push:"
Write-Host "   git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git"
Write-Host "   git branch -M main"
Write-Host "   git push -u origin main"
Write-Host ""
Write-Host "3. Share with friends! 🎉"
Write-Host ""
Write-Host "See GITHUB_SETUP.md for detailed instructions"
