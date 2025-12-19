# VesprAI - Integrated Financial Intelligence Platform

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**A comprehensive NLP-powered financial analysis system combining transformer-based sentiment analysis, document summarization, hybrid fraud detection, and investment intelligence.**

[Features](#-key-features) • [Quick Start](#-quick-start) • [Performance](#-performance-metrics) • [Architecture](#-system-architecture) • [Documentation](#-documentation)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Performance Metrics](#-performance-metrics)
- [System Architecture](#-system-architecture)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Module Details](#-module-details)
- [Dashboard](#-streamlit-dashboard)
- [Project Structure](#-project-structure)
- [Usage Examples](#-usage-examples)
- [Configuration](#-configuration)
- [Contributing](#-contributing)
- [Acknowledgments](#-acknowledgments)

---

## 🎯 Overview

VesprAI is an integrated financial intelligence platform developed as a capstone project for **MSML641/DATA641 - Advanced NLP Applications** at the University of Maryland. The system unifies four specialized NLP modules into a cohesive analytical framework, accessible through an interactive Streamlit dashboard.

### Why VesprAI?

Financial institutions process enormous volumes of unstructured text daily—earnings reports, SEC filings, news articles, and transaction records. VesprAI addresses the challenge of extracting actionable insights from this data by combining:

- **Transformer-based sentiment analysis** for market perception
- **Abstractive summarization** for document comprehension  
- **Hybrid NLP + numerical fraud detection** for risk assessment
- **Weighted integration** for investment recommendations

---

## ✨ Key Features

### 📊 Module 1: Financial Sentiment Analyzer
- **Model**: DistilBERT-base-uncased fine-tuned with Focal Loss + Label Smoothing
- **Performance**: **93.8% accuracy**, **0.942 F1-macro**
- **Enhancement**: Layered learning rates, gradient clipping, cosine LR scheduling
- **Output**: Three-class sentiment (Positive/Neutral/Negative) with confidence scores

### 📄 Module 2: Intelligent Document Summarizer
- **Model**: T5-small fine-tuned on 53 real financial documents
- **Performance**: **ROUGE-L 0.350**, 60-80% compression ratio
- **Training**: 0.55 minutes on CPU (highly efficient)
- **Output**: Abstractive summaries preserving key financial metrics

### 🛡️ Module 3: Unified Fraud Risk Scorer
- **Approach**: Hybrid supervised (Sentence-BERT embeddings + numerical features)
- **Performance**: **0.9563 ROC-AUC** (+38.5% over NLP-only baseline)
- **Features**: 384 text dimensions + 16 engineered numerical features
- **Output**: Risk scores, fraud probability, and interpretable risk factors

### 💡 Module 4: Investment Insight Generator
- **Integration**: Weighted combination of all modules
- **Weights**: Sentiment (40%) + Document Quality (35%) + Risk Assessment (25%)
- **Output**: BUY/HOLD/SELL recommendations with detailed rationale

### 💬 Module 5: Integrated Financial Chatbot
- **Interface**: Conversational AI with intent-based module routing
- **Deployment**: Streamlit dashboard with 5 interactive tabs
- **Capability**: Natural language queries with structured, quantified responses

---

## 📈 Performance Metrics

<div align="center">

| Module | Metric | Target | Achieved | Status |
|:-------|:-------|:------:|:--------:|:------:|
| **Sentiment Analysis** | Accuracy | ≥75% | **93.8%** | ✅ +25% |
| **Sentiment Analysis** | F1-Macro | ≥75% | **94.2%** | ✅ +26% |
| **Document Summarization** | ROUGE-L | ≥0.30 | **0.350** | ✅ +17% |
| **Fraud Detection** | ROC-AUC | ≥0.80 | **0.9563** | ✅ +20% |
| **System Integration** | Modules | 4/4 | **4/4** | ✅ Complete |

</div>

### Key Findings

| Insight | Value | Implication |
|---------|-------|-------------|
| Hybrid vs NLP-only fraud detection | +38.5% AUC improvement | Numerical features capture fraud signals text cannot |
| Unsupervised fraud recall | 0% | Pure anomaly detection fails on semantically similar fraud |
| T5-small vs extractive baseline | +40% ROUGE-L | Abstractive summarization essential for financial docs |
| Training efficiency | <10 min total (CPU) | Production-ready without GPU infrastructure |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           VesprAI System Architecture                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌─────────────────────────────────────────────────┐    │
│  │  📰 News     │───▶│  Module 1: Sentiment Analysis                   │    │
│  │  📊 Headlines│    │  DistilBERT + Focal Loss + Label Smoothing      │────┐│
│  └──────────────┘    │  Output: Sentiment Score (93.8% accuracy)       │    ││
│                      └─────────────────────────────────────────────────┘    ││
│                                                                              ││
│  ┌──────────────┐    ┌─────────────────────────────────────────────────┐    ││
│  │  📄 SEC      │───▶│  Module 2: Document Summarization               │    ││
│  │  📋 Reports  │    │  T5-small fine-tuned on financial docs          │────┤│
│  └──────────────┘    │  Output: Summary + Quality Score (ROUGE-L 0.35) │    ││
│                      └─────────────────────────────────────────────────┘    ││
│                                                                              ││
│  ┌──────────────┐    ┌─────────────────────────────────────────────────┐    ││
│  │  💳 Txns     │───▶│  Module 3: Fraud Risk Scorer                    │    ││
│  │  ⚖️ Balances │    │  Sentence-BERT (384d) + Numeric (16d) = 400d   │────┤│
│  └──────────────┘    │  Output: Risk Score (0.9563 AUC)                │    ││
│                      └─────────────────────────────────────────────────┘    ││
│                                                                              ││
│                      ┌─────────────────────────────────────────────────┐    ││
│                      │  Module 4: Investment Insight Generator         │◀───┘│
│                      │  Weighted Integration: 40% / 35% / 25%          │     │
│                      │  Output: BUY / HOLD / SELL Recommendation       │     │
│                      └──────────────────────┬──────────────────────────┘     │
│                                             │                                │
│                                             ▼                                │
│                      ┌─────────────────────────────────────────────────┐     │
│                      │  Module 5: Integrated Chatbot + Dashboard       │     │
│                      │  Streamlit UI with 5 Interactive Tabs           │     │
│                      │  Intent Detection → Module Routing → Response   │     │
│                      └─────────────────────────────────────────────────┘     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Installation

### Prerequisites

- **Python**: 3.8 or higher
- **RAM**: 8GB+ recommended
- **Storage**: 5GB+ for models and data
- **GPU**: Optional (CPU-optimized, GPU accelerates training)

### Step 1: Clone Repository

```bash
git clone https://github.com/your-username/VesprAI.git
cd VesprAI
```

### Step 2: Create Virtual Environment

```bash
# Create environment
python -m venv venv

# Activate (Linux/macOS)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download Data

```bash
# PaySim dataset for fraud detection
# Option 1: Kaggle CLI
kaggle datasets download -d ealaxi/paysim1 -p data/raw/ --unzip

# Option 2: Manual download from Kaggle
# https://www.kaggle.com/datasets/ealaxi/paysim1
# Place CSV in: data/raw/paysim.csv
```

---

## 🚀 Quick Start

### Option 1: Launch Dashboard (Recommended)

```bash
streamlit run app.py
# Opens browser at http://localhost:8501
```

### Option 2: Run Training Notebooks

```bash
jupyter notebook

# Execute in order:
# 1. 01_data_exploration_sentiment.ipynb
# 2. 02_data_exploration_fraud.ipynb
# 3. 03_sentiment_analysis.ipynb
# 4. 04_document_summarization.ipynb
# 5. 05_fraud_detection.ipynb
# 6. 06_investment_insights.ipynb
# 7. 07_integrated_chatbot.ipynb
```

### Option 3: Python API

```python
from src.investment_insight_generator import InvestmentInsightGenerator

# Initialize and load modules
generator = InvestmentInsightGenerator()
generator.load_modules()

# Generate investment insight
insight = generator.generate_insight(
    company_name="Apple Inc.",
    news_text="Apple reports record quarterly revenue, beating analyst expectations.",
    document_text="Revenue of $394B, net income $99.8B, strong cash position."
)

print(f"Recommendation: {insight['recommendation']}")
print(f"Confidence: {insight['overall_score']:.1%}")
```

---

## 📦 Module Details

### Module 1: Sentiment Analysis

<details>
<summary><b>Technical Specifications</b></summary>

| Component | Specification |
|-----------|---------------|
| Base Model | DistilBERT-base-uncased (~67M parameters) |
| Dataset | Financial PhraseBank (~4,800 sentences) |
| Classes | Negative, Neutral, Positive |
| Loss Function | Focal Loss (γ=2) + Label Smoothing (ε=0.1) |
| Optimizer | AdamW with Layered Learning Rates |
| LR Schedule | Cosine Annealing with Warmup |
| Training Time | 6.0 minutes (CPU) |

**Per-Class Performance:**
- Negative: F1 = 0.992
- Neutral: F1 = 0.917  
- Positive: F1 = 0.918

</details>

### Module 2: Document Summarization

<details>
<summary><b>Technical Specifications</b></summary>

| Component | Specification |
|-----------|---------------|
| Base Model | T5-small (~60M parameters) |
| Dataset | 53 real financial documents (42 train / 11 test) |
| Document Types | SEC filings, earnings reports, market analyses |
| Max Input | 512 tokens |
| Max Output | 128 tokens |
| Training Time | 0.55 minutes (CPU) |

**Performance by Document Type:**
- Market Analysis: 0.89 quality, 42% compression
- Financial News: 0.88 quality, 25% compression
- Investment Reports: 0.87 quality, 35% compression
- Earnings Reports: 0.85 quality, 32% compression
- SEC Filings: 0.82 quality, 28% compression

</details>

### Module 3: Fraud Detection

<details>
<summary><b>Technical Specifications</b></summary>

| Component | Unsupervised | Hybrid Supervised |
|-----------|--------------|-------------------|
| Text Encoder | all-MiniLM-L6-v2 | all-MiniLM-L6-v2 |
| Text Dims | 384 → 80 (PCA) | 384 |
| Numeric Features | 0 | 16 |
| Total Features | 80 | 400 |
| Algorithm | Isolation Forest | Logistic Regression |
| ROC-AUC | 0.6903 | **0.9563** |
| Recall | 0% | 71.4% |
| Training Time | 3.78 min | 4.06 min |

**Key Insight**: Adding 16 numerical features (4% of total) drives 38.5% performance improvement, demonstrating that fraud detection requires quantitative signals that text embeddings alone cannot capture.

</details>

---

## 🖥️ Streamlit Dashboard

The interactive dashboard provides 5 specialized tabs:

| Tab | Icon | Function | Key Features |
|-----|------|----------|--------------|
| **Chatbot** | 💬 | Conversational AI | Natural language queries, module routing |
| **Sentiment** | 📰 | News Analysis | Real-time sentiment, confidence gauges |
| **Summarizer** | 📝 | Document Condensation | Compression metrics, quality scores |
| **Fraud** | 🔒 | Risk Detection | Risk meters, red flag identification |
| **Insights** | 📈 | Investment Analysis | Radar charts, weighted recommendations |

### Dashboard Preview

```
┌─────────────────────────────────────────────────────────────┐
│  🔧 VesprAI System Status                                   │
├─────────────────────────────────────────────────────────────┤
│  ✅ Modules Loaded: 4/4                                     │
│  ✅ Sentiment Analyzer: Online (93.8% accuracy)             │
│  ✅ Document Summarizer: Online (ROUGE-L 0.35)              │
│  ✅ Fraud Detector: Online (AUC 0.9563)                     │
│  ✅ Investment Insights: Online                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
VesprAI/
├── 📂 data/                    # Data directory
│   └── raw/                    # Raw datasets (paysim.csv)
├── 📂 models/                  # Trained model weights
│   ├── final_model/            # Sentiment (DistilBERT)
│   ├── best_fraud_scorer/      # Fraud detection
│   └── summarizer/             # T5 summarization
├── 📂 notebooks/               # Jupyter notebooks
│   ├── 01_data_exploration_sentiment.ipynb
│   ├── 02_data_exploration_fraud.ipynb
│   ├── 03_sentiment_analysis.ipynb
│   ├── 04_document_summarization.ipynb
│   ├── 05_fraud_detection.ipynb
│   ├── 06_investment_insights.ipynb
│   └── 07_integrated_chatbot.ipynb
├── 📂 outputs/                 # Generated outputs
├── 📂 results/                 # Evaluation results & figures
├── 📂 src/                     # Source code modules
│   ├── sentiment_analyzer.py
│   ├── document_summarizer.py
│   ├── unified_fraud_risk_scorer.py
│   ├── investment_insight_generator.py
│   └── model_trainer.py
├── 📄 app.py                   # Streamlit dashboard
├── 📄 config.py                # Configuration settings
├── 📄 requirements.txt         # Python dependencies
├── 📄 LICENSE                  # MIT License
└── 📄 README.md                # This file
```

---

## 💻 Usage Examples

### Sentiment Analysis

```python
from transformers import pipeline

# Load trained model
sentiment = pipeline(
    "sentiment-analysis",
    model="models/final_model",
    tokenizer="models/final_model",
    return_all_scores=True
)

# Analyze financial text
result = sentiment("Apple reports record quarterly revenue growth of 15%")
# Output: [{'label': 'Positive', 'score': 0.89}]
```

### Document Summarization

```python
from src.document_summarizer import DocumentSummarizer

summarizer = DocumentSummarizer(model_path="models/summarizer")

text = """
Apple Inc. reported quarterly revenue of $123.9 billion, an increase of 11% 
year over year, and quarterly earnings per diluted share of $2.18. The Company 
also announced a dividend increase of 4% and an additional $110 billion for 
share repurchases. iPhone revenue reached $71.4 billion, up 17% from last year.
"""

summary = summarizer.summarize(text)
# Output: "Apple reported $123.9B revenue (+11% YoY), $2.18 EPS, with iPhone 
#          sales of $71.4B (+17%). Announced 4% dividend increase and $110B buyback."
```

### Fraud Detection

```python
from src.unified_fraud_risk_scorer import UnifiedFraudRiskScorer

scorer = UnifiedFraudRiskScorer()
scorer.load_models("models/best_fraud_scorer")

result = scorer.score_transaction(
    text="Urgent transfer of $50,000 to overseas account, sender balance now $0",
    amount=50000,
    old_balance_org=50000,
    new_balance_orig=0,
    trans_type="TRANSFER"
)

print(f"Risk Level: {result['risk_level']}")  # HIGH
print(f"Fraud Score: {result['risk_percentage']:.1%}")  # 98.2%
```

### Investment Analysis

```python
from src.investment_insight_generator import InvestmentInsightGenerator

generator = InvestmentInsightGenerator()
generator.load_modules()

insight = generator.generate_insight(
    company_name="Tesla Inc.",
    news_text="Tesla announces record vehicle deliveries for Q4.",
    document_text="Revenue $25.2B, automotive gross margin 17.6%, free cash flow $2.1B."
)

print(f"Recommendation: {insight['recommendation']}")  # HOLD
print(f"Overall Score: {insight['overall_score']:.1%}")  # 65.6%
print(f"Components:")
print(f"  - Sentiment: {insight['sentiment_score']:.1%}")  # 62.4%
print(f"  - Financial Health: {insight['document_score']:.1%}")  # 66.0%
print(f"  - Risk Assessment: {insight['risk_score']:.1%}")  # 70.0%
```

---

## ⚙️ Configuration

### Scoring Weights

```python
# In config.py or directly in InvestmentInsightGenerator
WEIGHTS = {
    'sentiment': 0.40,        # Market perception weight
    'document_quality': 0.35, # Fundamental analysis weight
    'risk_assessment': 0.25   # Risk factor weight
}
```

### Model Paths

```python
# In config.py
PATHS = {
    'sentiment_model': 'models/final_model',
    'summarizer_model': 'models/summarizer', 
    'fraud_model': 'models/best_fraud_scorer',
    'data_raw': 'data/raw',
}
```

### Recommendation Thresholds

```python
THRESHOLDS = {
    'strong_buy': 0.80,   # Score ≥ 0.80
    'buy': 0.70,          # Score 0.70-0.79
    'hold': 0.50,         # Score 0.50-0.69
    'weak_hold': 0.40,    # Score 0.40-0.49
    'sell': 0.00,         # Score < 0.40
}
```

---

## 🔧 Requirements

```txt
# Core ML/NLP
torch>=1.9.0
transformers>=4.20.0
sentence-transformers>=2.2.0
scikit-learn>=1.1.0

# Data Processing
pandas>=1.4.0
numpy>=1.21.0
datasets>=2.0.0

# Evaluation
rouge-score>=0.1.2

# Visualization
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.15.0

# Dashboard
streamlit>=1.28.0

# Utilities
jupyter>=1.0.0
joblib>=1.3.0
tqdm>=4.64.0
```

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add docstrings to all functions
- Include unit tests for new features
- Update README for significant changes

---

## 📚 Documentation

- **Notebooks**: Step-by-step walkthroughs in `/notebooks`
- **API Reference**: Docstrings in `/src` modules
- **Dashboard Guide**: Interactive help in Streamlit sidebar
- **Technical Report**: See accompanying LaTeX paper

---

## 🙏 Acknowledgments

### Datasets
- **Financial PhraseBank**: Malo et al., University of Helsinki
- **PaySim**: Lopez-Rojas & Axelsson, Blekinge Institute of Technology

### Frameworks & Libraries
- **Transformers**: Hugging Face
- **PyTorch**: Meta AI Research
- **Streamlit**: Streamlit Inc.
- **Sentence-Transformers**: UKP Lab, TU Darmstadt

### Academic References
- Devlin et al. (2019) - BERT
- Sanh et al. (2019) - DistilBERT
- Raffel et al. (2020) - T5
- Lin et al. (2017) - Focal Loss
- Reimers & Gurevych (2019) - Sentence-BERT

---

## 👥 Team

| Name | Role | Contact |
|------|------|---------|
| **Anisha Katiyar** | Co-Developer | ani14kay@umd.edu |
| **Siddhi Rohan Chakka** | Co-Developer | srohan@umd.edu |

**Course**: MSML641/DATA641 - Advanced NLP Applications  
**Program**: Masters in Data Science, University of Maryland

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

### 🚀 Quick Commands

```bash
# Clone → Install → Launch
git clone https://github.com/your-username/VesprAI.git && cd VesprAI
pip install -r requirements.txt
streamlit run app.py
```

---

**⭐ Star this repo if you find it useful!**

<br>

| Module | Status | Performance |
|:------:|:------:|:-----------:|
| Sentiment | ✅ Online | 93.8% Acc |
| Summarizer | ✅ Online | 0.35 ROUGE-L |
| Fraud | ✅ Online | 0.956 AUC |
| Insights | ✅ Online | Integrated |

<br>

*VesprAI — Where Financial Intelligence Meets Artificial Intelligence*

**Built with ❤️ at the University of Maryland**

</div>
