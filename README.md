# Marvel Characters MLOps Platform

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![MLflow](https://img.shields.io/badge/MLflow-3.1.1-blue.svg)](https://mlflow.org/)
[![Databricks](https://img.shields.io/badge/Databricks-Asset_Bundle-orange.svg)](https://docs.databricks.com/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.6.0-green.svg)](https://lightgbm.readthedocs.io/)

> An end-to-end MLOps platform for predicting Marvel character survival using production-grade machine learning pipelines deployed on Databricks.

## 🎯 Project Overview

This project demonstrates a complete MLOps lifecycle implementation using the Marvel Characters dataset. It showcases industry best practices for building, deploying, and monitoring machine learning models in production environments. The platform predicts character survival status (alive/dead) based on various features including physical attributes, universe affiliation, identity, and origin.

**Key Highlights:**
- Production-ready ML pipeline with automated data preprocessing
- Custom model wrappers with human-readable prediction outputs
- Automated model deployment with A/B testing capabilities
- Real-time model serving with Databricks endpoints
- Lakehouse monitoring with drift detection
- CI/CD workflows using Databricks Asset Bundles
- Comprehensive test coverage with pytest

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Data Ingestion Layer                     │
│  Marvel Characters Dataset → Data Preprocessing → Delta     │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────────┐
│                   Feature Engineering                        │
│  • Categorical Encoding  • Missing Value Imputation          │
│  • Feature Derivation    • Universe Normalization            │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────────┐
│                    Model Training Layer                      │
│  LightGBM Classifier → MLflow Tracking → Model Registry     │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────────┐
│                   Model Deployment Layer                     │
│  Custom PyFunc Wrapper → Model Serving Endpoint             │
│  • Versioning  • A/B Testing  • Auto-scaling                 │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────────┐
│                   Monitoring & Observability                 │
│  Lakehouse Monitor → Inference Logs → Drift Detection       │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Key Features

### 1. **Advanced Data Processing**
- Automated preprocessing pipeline with configurable feature transformations
- Smart categorical feature handling (Universe grouping, Origin normalization)
- Feature engineering (Magic detection, Mutant classification)
- Synthetic data generation with optional drift injection for testing

### 2. **Dual Model Architecture**
- **Basic Model**: LightGBM classifier with scikit-learn pipeline integration
- **Custom Model**: PyFunc wrapper with human-readable outputs ("alive"/"dead")
- Model comparison and performance-based automatic promotion

### 3. **Production Deployment**
- Databricks Model Serving with auto-scaling and scale-to-zero
- Version management with model aliases ("latest-model")
- Blue-green deployment support through A/B testing capabilities
- Automated deployment workflows via Databricks Jobs

### 4. **Monitoring & Observability**
- Real-time inference logging to Delta tables
- Lakehouse monitoring with 30-minute granularity
- Automated drift detection and alerts
- Change Data Feed for audit trails

### 5. **MLOps Best Practices**
- Environment-specific configurations (dev/acc/prd)
- Git-based version tracking with SHA tagging
- Automated testing with pytest and coverage reporting
- Pre-commit hooks with ruff for code quality
- Databricks Asset Bundles for infrastructure as code

## 📊 Dataset

**Source**: [Marvel Characters Dataset](https://www.kaggle.com/datasets/mohitbansal31s/marvel-characters) from Kaggle

**Features**:
- **Numerical**: Height (m), Weight (kg)
- **Categorical**: Universe, Identity, Gender, Marital Status, Teams, Origin
- **Derived**: Magic (boolean), Mutant (boolean)
- **Target**: Alive (binary classification)

**Dataset Statistics**:
- 4,000+ Marvel characters
- 13 original features + 2 engineered features
- Multi-universe coverage (Earth-616, Earth-1610, etc.)

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **ML Framework** | scikit-learn, LightGBM, MLflow |
| **Data Processing** | pandas, NumPy, PySpark |
| **Cloud Platform** | Databricks (Unity Catalog, Model Serving, Jobs) |
| **Development** | Python 3.12, UV package manager |
| **Testing** | pytest, pytest-cov |
| **Code Quality** | ruff, pre-commit |
| **Orchestration** | Databricks Workflows, Asset Bundles |
| **Monitoring** | Databricks Lakehouse Monitoring |

## 📁 Project Structure

```
marvel-characters/
├── src/marvel_characters/       # Core application code
│   ├── config.py                # Configuration management with Pydantic
│   ├── data_processor.py        # Data preprocessing pipeline
│   ├── monitoring.py            # Lakehouse monitoring setup
│   ├── utils.py                 # Utility functions
│   ├── models/
│   │   ├── basic_model.py       # LightGBM classification model
│   │   └── custom_model.py      # PyFunc wrapper with custom logic
│   └── serving/
│       └── model_serving.py     # Databricks serving endpoint manager
├── scripts/                     # Deployment and workflow scripts
│   ├── process_data.py          # Data preprocessing entry point
│   ├── train_register_custom_model.py  # Model training workflow
│   ├── deploy_model.py          # Model deployment automation
│   └── refresh_monitor.py       # Monitoring table refresh
├── resources/                   # Databricks Asset Bundle configs
│   ├── model_deployment.yml     # Deployment workflow definition
│   └── bundle_monitoring.yml    # Monitoring workflow definition
├── tests/                       # Test suite
│   └── marvel_characters/
│       └── test_data_processor.py  # Unit tests for data processing
├── data/                        # Dataset storage
│   └── marvel_characters_dataset.csv
├── databricks.yml               # Databricks bundle configuration
├── project_config_marvel.yml    # Project-specific configuration
├── pyproject.toml               # Python project metadata and dependencies
└── README.md                    # This file
```

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.12
- Databricks workspace with Unity Catalog enabled
- [UV package manager](https://docs.astral.sh/uv/getting-started/installation/) (recommended)

### Local Development Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd marvel-characters
   ```

2. **Create virtual environment and install dependencies**
   ```bash
   # Using UV (recommended)
   uv sync --extra dev

   # Or using pip
   pip install -e ".[dev]"
   ```

3. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

4. **Configure Databricks connection**
   - Set up your Databricks CLI profile (named "marvelous" by default)
   - Update `databricks.yml` with your workspace URL
   ```bash
   databricks configure --profile marvelous
   ```

5. **Update configuration**
   - Edit `project_config_marvel.yml` with your catalog and schema names
   - Ensure your Unity Catalog has the necessary permissions

### Running Tests

```bash
# Run all tests with coverage
pytest

# Run specific test file
pytest tests/marvel_characters/test_data_processor.py

# Run with verbose output
pytest -v --cov=src --cov-report=html
```

## 🚀 Usage

### 1. Data Preprocessing

```bash
# Local execution
python scripts/process_data.py \
    --root_path /path/to/workspace \
    --env dev
```

This script:
- Loads the Marvel dataset
- Applies feature engineering transformations
- Splits data into train/test sets (80/20)
- Saves to Unity Catalog as Delta tables
- Enables Change Data Feed for monitoring

### 2. Model Training & Registration

```bash
python scripts/train_register_custom_model.py \
    --root_path /path/to/workspace \
    --env dev \
    --git_sha $(git rev-parse HEAD) \
    --branch $(git branch --show-current) \
    --job_run_id local-run
```

This script:
- Trains LightGBM classifier on preprocessed data
- Logs experiments to MLflow
- Compares performance with existing models
- Registers improved models to Unity Catalog
- Creates custom PyFunc wrapper for serving

### 3. Model Deployment

```bash
python scripts/deploy_model.py \
    --root_path /path/to/workspace \
    --env dev
```

This script:
- Retrieves the latest model version
- Creates or updates Databricks serving endpoint
- Configures workload size and auto-scaling
- Enables real-time inference API

### 4. Monitoring Setup

```bash
python scripts/refresh_monitor.py \
    --root_path /path/to/workspace \
    --env dev
```

This script:
- Processes inference logs from serving endpoint
- Creates monitoring table with parsed features
- Sets up Lakehouse Monitor for drift detection
- Configures 30-minute granularity metrics

## 🔄 Databricks Workflows

### Deploy via Databricks Asset Bundles

```bash
# Validate bundle configuration
databricks bundle validate -t dev

# Deploy to development environment
databricks bundle deploy -t dev

# Run the deployment workflow
databricks bundle run deployment -t dev

# Deploy to production
databricks bundle deploy -t prd
databricks bundle run deployment -t prd
```

### Workflow Tasks

**Model Deployment Workflow** (`model_deployment.yml`):
1. **preprocessing**: Loads and preprocesses Marvel data
2. **train_model**: Trains model and evaluates performance
3. **model_updated**: Conditional check for model improvement
4. **deploy_model**: Deploys to serving endpoint (if improved)

**Monitoring Workflow** (`bundle_monitoring.yml`):
- **refresh_monitor_table**: Refreshes monitoring dashboards and metrics
- Runs weekly (every Monday at 6 AM Amsterdam time)

## 📈 Model Performance

The model uses LightGBM with the following hyperparameters:
- Learning rate: 0.01
- Number of estimators: 1000
- Max depth: 6

**Evaluation Metrics**:
- F1 Score (primary metric for model comparison)
- Accuracy
- Precision & Recall
- Confusion Matrix

Models are automatically promoted only if F1 score improves over the current "latest-model".

## 🧪 Testing & Quality Assurance

### Test Coverage
- Unit tests for data preprocessing pipeline
- Mock-based tests for Spark and Databricks SDK interactions
- Synthetic data generation testing
- CI/CD exclusion markers for environment-specific tests

### Code Quality Tools
- **ruff**: Linting and formatting (120 char line length)
- **pre-commit**: Automated checks before commits
- **pytest-cov**: Coverage reporting (target: src/)

### Quality Standards
```bash
# Run linter
ruff check src/ scripts/

# Format code
ruff format src/ scripts/

# Type checking with annotations
# Enforced via ruff ANN rules
```

## 🔐 Configuration Management

### Environment-Specific Settings

The project supports three environments with isolated resources:

```yaml
dev:
  catalog_name: mlops_dev
  schema_name: marvel_characters
  
acc:
  catalog_name: mlops_acc
  schema_name: marvel_characters
  
prd:
  catalog_name: mlops_prd
  schema_name: marvel_characters
```

### Model Configurations

**Features**:
- Numerical: `Height`, `Weight`
- Categorical: `Universe`, `Identity`, `Gender`, `Marital_Status`, `Teams`, `Origin`, `Magic`, `Mutant`
- Target: `Alive` (binary: 0=dead, 1=alive)

**MLflow Experiments**:
- Basic Model: `/Shared/marvel-characters-basic`
- Custom Model: `/Shared/marvel-characters-custom`

## 📝 API Usage

### Model Serving Endpoint

Once deployed, you can query the model via REST API:

```python
import requests
import os

# Databricks workspace URL and token
DATABRICKS_URL = "https://your-workspace.cloud.databricks.com"
DATABRICKS_TOKEN = os.environ["DATABRICKS_TOKEN"]

endpoint_url = f"{DATABRICKS_URL}/serving-endpoints/marvel-character-model-serving/invocations"

headers = {
    "Authorization": f"Bearer {DATABRICKS_TOKEN}",
    "Content-Type": "application/json"
}

# Example prediction request
payload = {
    "dataframe_records": [
        {
            "Height": 1.88,
            "Weight": 90.0,
            "Universe": "Marvel",
            "Identity": "Secret",
            "Gender": "Male",
            "Marital_Status": "Single",
            "Teams": "1",
            "Origin": "Human",
            "Magic": "0",
            "Mutant": "0"
        }
    ]
}

response = requests.post(endpoint_url, headers=headers, json=payload)
print(response.json())
# Output: {"Survival prediction": ["alive"]}
```

## 🎓 Key Learnings & MLOps Practices

This project demonstrates:

1. **Model Lifecycle Management**: Version control, experiment tracking, and automated model promotion
2. **Production Deployment**: Scalable serving infrastructure with monitoring
3. **Data Governance**: Unity Catalog integration with environment isolation
4. **Pipeline Automation**: Orchestrated workflows with conditional logic
5. **Monitoring & Observability**: Drift detection and inference logging
6. **Testing Strategy**: Unit tests, integration tests, and CI/CD patterns
7. **Code Quality**: Linting, formatting, and type annotations
8. **Infrastructure as Code**: Declarative deployment via Asset Bundles

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is available for portfolio and educational purposes.

## 🙏 Acknowledgments

- Dataset: [Marvel Characters Dataset](https://www.kaggle.com/datasets/mohitbansal31s/marvel-characters) by Mohit Bansal
- Platform: Databricks for MLOps infrastructure
- Tools: MLflow, LightGBM, scikit-learn communities

## 📧 Contact

For questions or collaboration opportunities, please reach out via:
- GitHub: [Your GitHub Profile]
- LinkedIn: [Your LinkedIn Profile]
- Email: [Your Email]

---

**Built with ❤️ for demonstrating end-to-end MLOps capabilities**
