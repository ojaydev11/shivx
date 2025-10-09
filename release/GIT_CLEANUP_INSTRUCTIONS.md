# Git Repository Cleanup Instructions

**Problem:** Push to GitHub rejected due to large files (backups/, dist/, build/) exceeding 100 MB limit.

**Root Cause:** Repository history contains large binary files:
- `backups/shivx-20250828-*/` (909 MB zip files, 944 MB DLLs)
- `dist/shivx/*.exe` (92 MB executables)
- `dash/frontend/node_modules/` (123 MB node files)

---

## Option 1: Clean Repository (RECOMMENDED)

**Creates a new clean branch without large files**

### Step 1: Create .gitignore (DONE)
Already created comprehensive `.gitignore` excluding all large files.

### Step 2: Remove Large Files from History

```powershell
# WARNING: This rewrites git history!
# Backup your work first!

# Option A: Use git-filter-repo (recommended)
# Install: pip install git-filter-repo

git filter-repo --path backups/ --invert-paths
git filter-repo --path dist/ --invert-paths  
git filter-repo --path build/ --invert-paths
git filter-repo --path node_modules/ --invert-paths
git filter-repo --path memory/ --invert-paths
git filter-repo --path data/ --invert-paths
git filter-repo --path ShivX-Portable.zip --invert-paths

# Option B: Use BFG Repo-Cleaner
# Download from: https://rtyley.github.io/bfg-repo-cleaner/
java -jar bfg.jar --strip-blobs-bigger-than 50M
git reflog expire --expire=now --all
git gc --prune=now --aggressive
```

### Step 3: Force Push Clean History

```powershell
# WARNING: Force push! Coordinate with team
git push origin release/shivx-hardening-001 --force
```

---

## Option 2: Fresh Hardening Branch (EASIEST)

**Start from a clean commit with only hardening files**

### Step 1: Create New Branch from Clean Commit

```powershell
# Find a clean commit before large files were added
git log --oneline --all

# Create new hardening branch from clean commit
git checkout -b release/shivx-hardening-001-clean <clean-commit-sha>

# Or start fresh from current main
git checkout main
git checkout -b release/shivx-hardening-001-clean
```

### Step 2: Copy Only Hardening Files

```powershell
# Copy hardening files from old branch
git checkout release/shivx-hardening-001 -- release/
git checkout release/shivx-hardening-001 -- scripts/dev_bootstrap.ps1
git checkout release/shivx-hardening-001 -- scripts/dev_bootstrap.sh
git checkout release/shivx-hardening-001 -- scripts/run_all_tests.ps1
git checkout release/shivx-hardening-001 -- scripts/load_tests.ps1
git checkout release/shivx-hardening-001 -- scripts/chaos_suite.ps1
git checkout release/shivx-hardening-001 -- scripts/security_scan.ps1
git checkout release/shivx-hardening-001 -- pytest.ini
git checkout release/shivx-hardening-001 -- .pre-commit-config.yaml
git checkout release/shivx-hardening-001 -- .github/workflows/shivx_hardening.yml
git checkout release/shivx-hardening-001 -- utils/
git checkout release/shivx-hardening-001 -- .gitignore
```

### Step 3: Commit and Push Clean Branch

```powershell
git add .
git commit -m "[HARDENING-001] Production Hardening - Clean Branch

All hardening infrastructure without large binary files:
- 17 core hardening files
- GitHub Actions CI workflow
- GOLD execution roadmap
- Comprehensive .gitignore"

git push -u origin release/shivx-hardening-001-clean
```

---

## Option 3: Git LFS (For Future)

**Use Git Large File Storage for legitimate large files**

```powershell
# Install Git LFS
git lfs install

# Track large file patterns
git lfs track "*.exe"
git lfs track "*.dll"
git lfs track "*.zip"
git lfs track "*.pth"
git lfs track "*.bin"

git add .gitattributes
git commit -m "Configure Git LFS"
```

**Note:** Still won't help with files already in history - need Option 1 or 2 first.

---

## Option 4: Exclude From Current Push (TEMPORARY)

**Push only hardening branch without large files**

```powershell
# Use sparse checkout or push specific files
# This is complex - Option 2 is easier
```

---

## RECOMMENDED APPROACH

**Use Option 2: Fresh Hardening Branch**

This is the cleanest and safest approach:
1. ✅ No history rewriting required
2. ✅ Only hardening files in the new branch
3. ✅ Easy to verify what's being pushed
4. ✅ No risk of breaking other branches

**Execution:**

```powershell
# 1. Create clean branch from main
git checkout main
git pull origin main  # Ensure main is up-to-date
git checkout -b release/shivx-hardening-clean

# 2. Apply .gitignore (already created)
git add .gitignore
git commit -m "Add comprehensive .gitignore"

# 3. Cherry-pick hardening commits OR copy files manually
git checkout release/shivx-hardening-001 -- release/
git checkout release/shivx-hardening-001 -- scripts/dev_bootstrap.ps1 scripts/dev_bootstrap.sh scripts/run_all_tests.ps1 scripts/load_tests.ps1 scripts/chaos_suite.ps1 scripts/security_scan.ps1
git checkout release/shivx-hardening-001 -- pytest.ini .pre-commit-config.yaml
git checkout release/shivx-hardening-001 -- .github/
git checkout release/shivx-hardening-001 -- utils/

# 4. Commit hardening files
git add .
git commit -m "[HARDENING-001] Production Hardening - SILVER Certification

Complete hardening infrastructure:
✅ 8-phase certification framework
✅ Test infrastructure (570 tests)  
✅ Load/stress/soak test harness (5 profiles)
✅ Chaos engineering suite
✅ Security scans (SAST/secrets/SBOM)
✅ GitHub Actions CI workflow
✅ Documentation (quickstart, runbooks, roadmap)

Certification: SILVER → GOLD (execution ready)
Branch: release/shivx-hardening-clean (no large files)
Files: 19 (release/, scripts/, utils/, .github/, configs)"

# 5. Push clean branch
git push -u origin release/shivx-hardening-clean

# 6. Create PR from GitHub UI:
# https://github.com/ojaydev11/shivx/compare/main...release/shivx-hardening-clean
```

---

## After Cleanup

Once the clean branch is pushed successfully:

1. **Create PR** on GitHub
2. **GitHub Actions** will automatically run the hardening CI
3. **Review** CI results
4. **Merge** to main when all gates are GREEN
5. **Delete** old `release/shivx-hardening-001` branch (optional)

---

## Prevention for Future

1. ✅ Comprehensive `.gitignore` now in place
2. ✅ Add pre-commit hook to check file sizes
3. ✅ Document build artifact exclusion in CONTRIBUTING.md
4. ✅ Use Git LFS for legitimate large files (models, datasets)
5. ✅ Regular `git gc` to clean up unreachable objects

---

**Current Status:**
- ❌ `release/shivx-hardening-001` - blocked by large files
- ✅ `.gitignore` - created and comprehensive
- ⏳ `release/shivx-hardening-clean` - **NEXT: Create this branch**

**Recommended Next Command:**
```powershell
git checkout main
git pull origin main
git checkout -b release/shivx-hardening-clean
# Then follow steps above to copy hardening files
```

