# CI Pipeline Bug Report — `ml_pipeline.yml`

This report documents all bugs found in the original GitHub Actions workflow and Python source files, their root causes, and the fixes applied to achieve a passing (green) CI run.

---

## Part 1 — YAML Workflow Bugs

### Bug 1 — Missing Root-Level `requirements.txt`

**Location:** `.github/workflows/ml_pipeline.yml` — `Install Dependencies` step

**Original code:**
```yaml
- name: Install Dependencies
  run: pip install -r requirements.txt
```

**Error:**
```
ERROR: Could not open requirements file: [Errno 2] No such file or directory: 'requirements.txt'
```

**Root Cause:** The only `requirements.txt` in the repo was at `student_A/requirements.txt`. The CI runner executes from the repo root, so it cannot find a nested requirements file without an explicit path.

**Fix:** Created a root-level `requirements.txt` with all project dependencies:
```
pandas==2.1.1
numpy==1.26.0
torch==2.1.0
torchvision==0.16.0
mlflow
flake8
```

---

### Bug 2 — Missing `README.md` Causes Artifact Upload to Fail

**Location:** `.github/workflows/ml_pipeline.yml` — `README Artifact` step

**Original code:**
```yaml
- name: README Artifact
  uses: actions/upload-artifact@v4
  with:
    name: project-doc
    path: README.md
```

**Error:**
```
Error: No files were found with the provided path: README.md
```

**Root Cause:** No `README.md` existed at the repository root. `actions/upload-artifact@v4` treats a missing file as a hard error.

**Fix:** Created a `README.md` at the repository root documenting the project structure, dependencies, and how to run locally.

---

### Bug 3 — `README Artifact` Step Out of Order

**Location:** `.github/workflows/ml_pipeline.yml` — step ordering

**Original order:**
```
1. Checkout Code
2. Set up Python
3. README Artifact      ← ❌ artifact uploaded before any validation
4. Install Dependencies
5. Linter Check
```

**Root Cause:** The artifact upload was placed before `Install Dependencies` and `Linter Check`. This means the artifact was published even if the build subsequently failed — a misleading result.

**Fix:** Moved `README Artifact` to the final step so it only runs after all checks pass:
```
1. Checkout Code
2. Set up Python
3. Install Dependencies
4. Linter Check
5. README Artifact      ← ✅ only uploaded on full success
```

---

## Part 2 — Python Linting Bugs (Flake8)

The `Linter Check` step exposed flake8 violations across three Python files. All were fixed to achieve a clean run.

### `student_A/student_a_gan.py` & `Student_B/student_a_gan.py`

| Line | Code | Description | Fix |
|------|------|-------------|-----|
| 2 | F401 | `import numpy as np` imported but unused | Removed the import |
| 14 | E302 | Expected 2 blank lines before class, found 1 | Added a blank line before `class Generator` |
| 18 | E301 | Expected 1 blank line before method, found 0 | Added blank line before `def forward` |
| 21 | E302 | Expected 2 blank lines before class, found 1 | Added a blank line before `class Discriminator` |
| 25 | E301 | Expected 1 blank line before method, found 0 | Added blank line before `def forward` |
| 28 | E305 | Expected 2 blank lines after class definition | Added blank lines after `Discriminator` |
| 55 | W292 | No newline at end of file | Ensured file ends with `\n` |

### `student_a_gan_mlflow.py`

| Line | Code | Description | Fix |
|------|------|-------------|-----|
| 16–17 | E501 | Lines too long (122–124 chars, max 120) | Split `large_batch` and `bigger_model` dict entries across two lines |
| 23 | E302 | Expected 2 blank lines before class, found 1 | Added blank line before `class Generator` |
| 77–82 | E221 | Multiple spaces before operator (alignment spacing) | Removed extra alignment spaces from variable assignments |
| 186 | E303 | Too many blank lines (3) before `if __name__` | Reduced to 2 blank lines |

---

## Final Fixed Workflow

```yaml
# .github/workflows/ml_pipeline.yml
name: INIT CI

on:
  push:
    branches-ignore:
      - main
  pull_request:

jobs:
  validate-and-test:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout Code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install Dependencies
        run: pip install -r requirements.txt

      - name: Linter Check
        run: |
          flake8 . --max-line-length=120 --exclude=.git,__pycache__,mlruns,data

      - name: README Artifact
        uses: actions/upload-artifact@v4
        with:
          name: project-doc
          path: README.md
```

---

## Summary

| # | File | Bug | Fix |
|---|------|-----|-----|
| 1 | `ml_pipeline.yml` | Missing root-level `requirements.txt` | Created `requirements.txt` at repo root |
| 2 | `ml_pipeline.yml` | Missing `README.md` → artifact upload fails | Created `README.md` at repo root |
| 3 | `ml_pipeline.yml` | Artifact step before validation steps | Moved artifact upload to last step |
| 4 | `student_A/student_a_gan.py` | F401, E302, E301, E305, W292 | Removed unused import, fixed blank lines, added EOF newline |
| 5 | `Student_B/student_a_gan.py` | Same as above | Same fixes |
| 6 | `student_a_gan_mlflow.py` | E501, E302, E221, E303 | Split long lines, fixed spacing and blank lines |
