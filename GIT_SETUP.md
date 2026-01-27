# Git Setup Instructions

## Step 1: Commit Your Files

Run these commands in PowerShell from the DiffusionRWR_model_repo directory:

```powershell
cd C:\Users\inigo\OneDrive\Documents\Fourth_Year\Computations\Dissertation\DiffusionRWR_model_repo

# Stage all files
git add .

# Create initial commit
git commit -m "Initial commit: DiffusionRWR multi-layer graph model"
```

## Step 2: Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `DiffusionRWR_model`
3. Description: "Multi-layer graph random walk model for biological data"
4. Choose Public or Private
5. **DO NOT** initialize with README (we already have one)
6. Click "Create repository"

## Step 3: Push to GitHub

GitHub will show you commands. Use these:

```powershell
# Add remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/DiffusionRWR_model.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## Step 4: Verify

Visit your repository URL to confirm all files are uploaded!

## Future Updates

After making changes:

```powershell
git add .
git commit -m "Description of changes"
git push
```

## Files Created

✓ .gitignore - Excludes Python cache files, virtual environments, and data files
✓ README.md - Project documentation with usage instructions
✓ This file - Setup instructions
