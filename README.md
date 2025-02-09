# Fraud Detection System

This project implements a comprehensive fraud detection system using machine learning algorithms to identify fraudulent transactions in credit card data.

## Project Overview

The system uses multiple machine learning models to detect fraudulent transactions, featuring:
- Data preprocessing and scaling
- SMOTE for handling imbalanced data
- Multiple model training and comparison
- Comprehensive model evaluation and visualization
- Feature importance analysis
- Discord integration for logging and monitoring
- Model persistence for future use

## Project Structure

```
fraud-detection/
├── data/               # Data directory for storing datasets
├── models/            # Directory for model implementations and saved models
│   ├── base_model.py
│   ├── logistic_regression.py
│   └── mlp.py
├── output/            # Directory for generated visualizations
│   └── images/
├── src/              # Source code
│   ├── config.py
│   ├── data_preprocessor.py
│   ├── discord_logger.py
│   ├── model_evaluator.py
│   ├── model_operations_handler.py
│   └── model_tester.py
├── .env.fraud        # Environment variables configuration
├── .env.fraud.example # Example environment variables template
├── main.py          # Main execution script
└── requirements.txt  # Project dependencies
```

## Features

- Data preprocessing and feature scaling
- Handling of imbalanced datasets using SMOTE
- Multiple model implementations:
  - Logistic Regression
  - Multi-layer Perceptron (MLP)
  - XGBoost
- Visualization outputs:
  - Class distribution
  - Transaction amount distribution
  - Feature correlation matrices
  - Confusion matrices
  - ROC curves
  - PR curves
  - Feature importance plots
- Model performance metrics and evaluation
- Discord integration for training monitoring and logging
- Flexible model training/testing configuration
- Environment-based configuration management

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd fraud-detection
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv .env
source .env/bin/activate  # On Windows: .env\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.fraud.example .env.fraud
# Edit .env.fraud with your configuration
```

## Configuration

The project uses environment variables for configuration. Copy `.env.fraud.example` to `.env.fraud` and configure:
- Discord webhook URL for logging (optional)
- Model selection preferences
- Training/testing mode options

## Usage

Run the main script to execute the complete fraud detection pipeline:

```bash
python main.py
```

The script will:
1. Load and preprocess the credit card transaction data
2. Generate initial data visualizations
3. Apply SMOTE for class balancing
4. Train selected models
5. Evaluate model performance
6. Generate comprehensive performance visualizations
7. Save trained models and metrics
8. Log progress and results to Discord (if configured)

## Data

The project expects a credit card transaction dataset (`creditcard.csv`) in the `data/` directory. The dataset should contain transaction features and a binary 'Class' column indicating fraudulent transactions (1 for fraud, 0 for normal).

## Models

The system includes implementations for:
- Logistic Regression
- Multi-layer Perceptron (MLP)
- XGBoost

Models are automatically saved in the `models/` directory after training. The system compares model performance and identifies the best performing model based on various metrics.

## Output

The system generates various visualizations in the `output/images/` directory:
- Distribution plots
- Correlation matrices
- ROC and PR curves
- Confusion matrices
- Feature importance plots
- Model comparison metrics

## Requirements

Key dependencies include:
- Python 3.x
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- xgboost
- imbalanced-learn (SMOTE)
- discord-webhook (optional, for logging)

See `requirements.txt` for a complete list of dependencies and versions.

## License

[Your chosen license] 