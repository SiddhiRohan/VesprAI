"""
Configuration file for sentiment analysis project
"""
import os
from pathlib import Path

# Project structure
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks" 
SRC_DIR = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Model configuration
MODEL_CONFIG = {
    "model_name": "distilbert-base-uncased",
    "num_labels": 3,
    "max_length": 128,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "num_epochs": 5,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "save_steps": 200,
    "eval_steps": 200,
    "logging_steps": 50
}

# Summarization configuration  
SUMMARIZATION_CONFIG = {
    "model_name": "t5-small",
    "max_input_length": 512,
    "max_output_length": 128,
    "num_epochs": 3,
    "batch_size": 4,
    "learning_rate": 1e-4
}

# Data configuration
DATA_CONFIG = {
    "test_size": 0.2,
    "validation_size": 0.1,
    "random_seed": 42,
    "dataset_name": "financial_phrasebank"
}

# Labels mapping
LABEL_MAP = {
    0: "Negative",
    1: "Neutral", 
    2: "Positive"
}

# Reverse mapping
LABEL_TO_ID = {v: k for k, v in LABEL_MAP.items()}

# Paths
PATHS = {
    "train_dataset": DATA_DIR / "train_dataset",
    "test_dataset": DATA_DIR / "test_dataset", 
    "val_dataset": DATA_DIR / "val_dataset",
    "raw_data": DATA_DIR / "raw_data.csv",
    "final_model": MODELS_DIR / "final_model",
    "training_logs": RESULTS_DIR / "training_logs",
    "evaluation_results": RESULTS_DIR / "evaluation_results.json",
    "summarizer_model": MODELS_DIR / "summarizer",
    "summarization_data": DATA_DIR / "summarization_data.json", 
    "summarization_results": RESULTS_DIR / "summarization_results.json"
}
