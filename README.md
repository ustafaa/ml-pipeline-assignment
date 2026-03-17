# ML Pipeline Assignment

This repository contains a GAN (Generative Adversarial Network) implementation trained on the MNIST dataset, with MLflow experiment tracking and a GitHub Actions CI pipeline.

## Project Structure

```
assignment/
├── student_A/
│   ├── student_a_gan.py        # Original GAN script (Student A)
│   ├── requirements.txt        # Python dependencies for Student A
│   ├── Dockerfile              # Docker image for Student A's script
│   └── env.yaml                # Conda environment definition
├── Student_B/
│   ├── student_a_gan.py        # Reproducibility-improved GAN script
│   └── report.md               # Student B's MLOps/SRE analysis report
├── student_a_gan_mlflow.py     # Enhanced GAN with MLflow experiment tracking (5 runs)
├── docker-compose.yml          # Docker Compose configuration
├── requirements.txt            # Root-level Python dependencies for CI
└── .github/
    └── workflows/
        └── ml_pipeline.yml     # GitHub Actions CI pipeline
```

## CI / CD Pipeline

The GitHub Actions workflow (`ml_pipeline.yml`) runs on every push to **any branch except `main`**. It:

1. Checks out the code
2. Sets up Python 3.10
3. Installs dependencies from `requirements.txt`
4. Runs `flake8` linting (max line length: 120)
5. Uploads `README.md` as a GitHub Actions artifact named `project-doc`

## Running Locally

### Student A — Original Script
```bash
cd student_A
pip install -r requirements.txt
python student_a_gan.py
```

### MLflow Experiment Tracking (5 runs)
```bash
pip install -r requirements.txt
python student_a_gan_mlflow.py
```

### View MLflow UI
```bash
mlflow ui
# Open http://localhost:5000
```

### Docker
```bash
docker-compose up --build
```

## Dependencies

| Package    | Version  |
|------------|----------|
| pandas     | 2.1.1    |
| numpy      | 1.26.0   |
| torch      | 2.1.0    |
| mlflow     | latest   |
| torchvision| latest   |
| flake8     | latest   |
