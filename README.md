# VesprAI - Financial Intelligence System

[![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/status-Production%20Ready-brightgreen.svg)](https://github.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg)](https://streamlit.io)

> **Complete NLP-based financial analysis system with 4 integrated modules, interactive chatbot, and Streamlit dashboard for comprehensive investment intelligence**

## 🎯 Project Overview

VesprAI is a sophisticated financial intelligence platform that combines cutting-edge Natural Language Processing with quantitative analysis to provide comprehensive investment insights. The system integrates multiple AI modules to analyze market sentiment, summarize financial documents, assess risks, and generate actionable investment recommendations through an intuitive conversational interface.

## ✨ What's New

- 🚀 **Streamlit Dashboard** - Full-featured web interface with 5 interactive tabs
- 💬 **Integrated Chatbot** - Conversational AI that routes queries to appropriate modules
- 📊 **Baseline Comparison** - Demonstrates value of trained models vs generic responses
- 🎯 **Enhanced Performance** - Sentiment accuracy: 99%, Fraud AUC: 0.95

## 🖥️ Live Dashboard

![VesprAI Dashboard](docs/dashboard_preview.png)

### Quick Launch
```bash
# Run the Streamlit dashboard
streamlit run app.py
```

The dashboard includes:
| Tab | Feature | Description |
|-----|---------|-------------|
| 💬 **Chatbot** | Interactive Q&A | Natural language queries with module routing |
| 📰 **Sentiment** | News Analysis | Real-time sentiment with confidence scores |
| 📝 **Summarizer** | Document Condensation | SEC filings and earnings report summaries |
| 🔒 **Fraud** | Risk Detection | Transaction fraud scoring with visualizations |
| 📈 **Insights** | Investment Analysis | Buy/Hold/Sell recommendations with radar charts |

## 🚀 Key Features

### 📊 **Module 1: News Sentiment Analyzer**
- **Technology**: DistilBERT fine-tuned on Financial PhraseBank
- **Performance**: **99.05% accuracy** on financial sentiment classification ✅
- **Capability**: Real-time analysis of financial news and market sentiment
- **Output**: Sentiment scores with confidence levels and market impact assessment

### 📄 **Module 2: Document Summarizer** 
- **Technology**: T5-small transformer for extractive and abstractive summarization
- **Performance**: ROUGE-L **0.35** on SEC filings and financial documents ✅
- **Capability**: Automated summarization of 10-K, 10-Q, earnings reports
- **Output**: Key insights extraction and executive summaries

### 🛡️ **Module 3: Unified Fraud Risk Scorer**
- **Technology**: Dual approach - Unsupervised (Isolation Forest) + Hybrid Supervised (NLP + Numeric)
- **Performance**: **0.9563 ROC-AUC** with hybrid approach ✅
- **Capability**: Transaction fraud detection and risk assessment
- **Output**: Risk scores, fraud probability, and risk factor identification

### 💡 **Module 4: Investment Insight Generator**
- **Technology**: Multi-module integration with weighted scoring algorithm
- **Performance**: Comprehensive analysis combining all previous modules
- **Capability**: End-to-end investment analysis and recommendation generation
- **Output**: BUY/HOLD/SELL recommendations with detailed rationale

### 💬 **Module 5: Integrated Financial Chatbot**
- **Technology**: Intent detection + module routing + response formatting
- **Interface**: Jupyter notebook (10_integrated_chatbot.ipynb) + Streamlit (app.py)
- **Capability**: Conversational Q&A with baseline comparison mode
- **Output**: Structured responses with module-specific insights and metrics

## 🏗️ Architecture

```
VesprAI System Architecture

📰 News Data ────────┐
                     ├──► Module 1: Sentiment Analysis ────┐
📊 Market Data ──────┘                                     │
                                                           ├──► Module 4: Investment Insights
📄 SEC Filings ─────┬──► Module 2: Document Summarizer ───┤     │
📋 Reports ─────────┘                                     │     ├──► Module 5: Chatbot Interface
                                                           │     │         │
💳 Transactions ────┬──► Module 3: Fraud Risk Scorer ─────┘     │         │
⚖️ Risk Data ──────┘                                           │         │
                                                                │         ▼
🎯 Final Output ◄───────────────────────────────────────────────┘    🖥️ Streamlit
• Comprehensive Investment Analysis                                   Dashboard
• Risk-Adjusted Recommendations  
• Natural Language Insights
• Visual Dashboard & Rankings
• Interactive Chat Interface
```

## 📋 Requirements

### System Requirements
- **Python**: 3.8+
- **RAM**: 8GB+ recommended
- **Storage**: 5GB+ for models and data
- **GPU**: Optional (CUDA/MPS-enabled for faster inference)

### Dependencies
```bash
torch>=1.9.0
transformers>=4.20.0
sentence-transformers>=2.2.0
scikit-learn>=1.1.0
pandas>=1.4.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
jupyter>=1.0.0
rouge-score>=0.1.2
datasets>=2.0.0
streamlit>=1.28.0
plotly>=5.15.0
joblib>=1.3.0
```

## 🛠️ Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/your-username/VesprAI.git
cd VesprAI
```

### 2. Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Additional dependencies for dashboard
pip install streamlit plotly
```

### 3. Data Download
```bash
# Download PaySim dataset for Module 3 (Fraud Detection)
# Option 1: Using Kaggle CLI
kaggle datasets download -d ealaxi/paysim1 -p data/raw/ --unzip

# Option 2: Manual download
# Visit: https://www.kaggle.com/datasets/ealaxi/paysim1
# Place the CSV in: data/raw/paysim.csv
```

### 4. Run Notebooks (Training)
```bash
# Start Jupyter notebook
jupyter notebook

# Run notebooks in order:
# 1. notebooks/01_data_exploration_fixed.ipynb
# 2. notebooks/02_model_training_FINAL.ipynb
# 3. notebooks/03_document_summarization.ipynb
# 4. notebooks/06_unified_fraud_detection.ipynb
# 5. notebooks/07_investment_insights.ipynb
# 6. notebooks/10_integrated_chatbot.ipynb (Final Integration)
```

### 5. Launch Dashboard
```bash
# Run Streamlit dashboard (from project root)
streamlit run app.py

# Opens browser at http://localhost:8501
```

## 📊 Performance Metrics

| Module | Metric | Target | Achieved | Status |
|--------|--------|--------|----------|--------|
| **Sentiment Analyzer** | Accuracy | ≥75% | **99.05%** | ✅ Exceeded |
| **Sentiment Analyzer** | F1 Score | ≥75% | **99.05%** | ✅ Exceeded |
| **Document Summarizer** | ROUGE-L | ≥30 | **0.35** | ✅ Met |
| **Fraud Risk Scorer** | ROC-AUC | ≥80% | **95.63%** | ✅ Exceeded |
| **Investment Insights** | Integration | Complete | **100%** | ✅ Complete |
| **Chatbot Interface** | Modules | 4/4 | **4/4** | ✅ Complete |

### Module Comparison: With Training vs Baseline

| Aspect | Baseline (No Training) | With Trained Modules |
|--------|------------------------|----------------------|
| **Sentiment** | ~33% random guess | 99.05% accuracy |
| **Fraud Detection** | ~50% AUC (random) | 95.63% AUC |
| **Response Quality** | Vague, generic | Specific scores & confidence |
| **Investment Advice** | "Consult advisor" | Quantified recommendations |

## 🎯 Usage Examples

### Option 1: Streamlit Dashboard (Recommended)
```bash
streamlit run app.py
```
Then use the interactive tabs for:
- Chatbot conversations
- Sentiment analysis with gauges
- Document summarization
- Fraud detection with risk meters
- Investment insights with radar charts

### Option 2: Jupyter Notebook
```python
# Run notebooks/10_integrated_chatbot.ipynb for interactive demo
```

### Option 3: Python API

#### Quick Sentiment Analysis
```python
from transformers import pipeline

# Load trained model
sentiment = pipeline(
    "sentiment-analysis",
    model="models/final_model",
    tokenizer="models/final_model",
    return_all_scores=True
)

result = sentiment("Apple reports record quarterly revenue growth of 15%")
print(result)
# [{'label': 'LABEL_2', 'score': 0.89}]  # LABEL_2 = Positive
```

#### Document Summarization
```python
from src.document_summarizer import DocumentSummarizer

summarizer = DocumentSummarizer(model_name="t5-small")
summary = summarizer.summarize("Long SEC filing text here...")
print(f"Summary: {summary}")
```

#### Fraud Detection
```python
from src.unified_fraud_risk_scorer import UnifiedFraudRiskScorer
import joblib

scorer = UnifiedFraudRiskScorer(model_name="all-MiniLM-L6-v2")

# Load trained weights
scorer.hybrid_classifier = joblib.load("models/best_fraud_scorer/best_classifier.joblib")
scorer.text_scaler = joblib.load("models/best_fraud_scorer/best_text_scaler.joblib")
scorer.numeric_scaler = joblib.load("models/best_fraud_scorer/best_numeric_scaler.joblib")

result = scorer.score_transaction(
    text="Transfer of $50000 from account A to B, sender balance now $0",
    method="hybrid",
    amount=50000,
    old_balance_org=50000,
    new_balance_orig=0,
    old_balance_dest=0,
    new_balance_dest=50000,
    trans_type="TRANSFER"
)
print(f"Risk Level: {result['risk_level']}, Score: {result['risk_percentage']:.1%}")
```

#### Comprehensive Investment Analysis
```python
from src.investment_insight_generator import InvestmentInsightGenerator

generator = InvestmentInsightGenerator()
generator.load_modules()

insight = generator.generate_insight(
    company_name="Apple Inc.",
    news_text="Apple reports strong quarterly earnings beating analyst expectations.",
    document_text="Revenue of $394B, net income $99.8B, strong cash position of $165B."
)

print(f"Recommendation: {insight['recommendation']}")
print(f"Confidence: {insight['overall_score']:.1%}")
print(f"Narrative: {insight['narrative']}")
```

## 📁 Project Structure

```
VesprAI/
├── 📂 src/                              # Source code
│   ├── sentiment_analyzer.py           # Module 1: Sentiment Analysis
│   ├── document_summarizer.py          # Module 2: Document Summarization
│   ├── unified_fraud_risk_scorer.py    # Module 3: Fraud Risk Assessment
│   ├── investment_insight_generator.py # Module 4: Investment Insights
│   └── model_trainer.py                # Training utilities
├── 📂 notebooks/                        # Jupyter demonstrations
│   ├── 01_data_exploration_fixed.ipynb  # Data exploration
│   ├── 02_model_training_FINAL.ipynb    # Sentiment model training
│   ├── 03_document_summarization.ipynb  # Summarizer demo
│   ├── 06_unified_fraud_detection.ipynb # Fraud detection training
│   ├── 07_investment_insights.ipynb     # Investment insights demo
│   └── 10_integrated_chatbot.ipynb      # 🆕 Integrated chatbot
├── 📂 models/                           # Trained models
│   ├── final_model/                     # Trained DistilBERT (99% acc)
│   ├── best_fraud_scorer/               # Trained fraud detector (0.95 AUC)
│   └── investment_insight_generator/    # Insight config
├── 📂 data/                             # Data directory
│   └── 📂 raw/                          # Raw datasets (paysim.csv)
├── app.py                               # 🆕 Streamlit Dashboard
├── config.py                            # Configuration settings
├── requirements.txt                     # Python dependencies
└── README.md                            # This file
```

## 🎨 Sample Output

### Streamlit Dashboard
```
🔧 System Status
├── Modules Loaded: 4/4 ✅
├── Sentiment Analyzer: DistilBERT (99% acc)
├── Document Summarizer: T5-small (ROUGE-L 0.35)
├── Fraud Detector: Hybrid ML (AUC 0.95)
└── Investment Insights: Integrated Pipeline
```

### Chatbot Response Example
```
🤔 You: What's the sentiment of: Tesla reported record deliveries

🤖 VesprAI [sentiment]:
📊 Sentiment Analysis Results

Overall Sentiment: Positive
Confidence: 71.5%

Detailed Scores:
• Positive: 71.5%
• Neutral: 18.2%
• Negative: 10.3%

Analysis: Based on our trained DistilBERT model (99% accuracy), 
this text expresses positive sentiment with 71.5% confidence.
```

### Investment Analysis Output
```
🏆 INVESTMENT RANKINGS:

Rank Company         Score    Recommendation Sentiment Quality  Risk    
─────────────────────────────────────────────────────────────────────
🥇 1  Apple          0.753    BUY            0.715     0.680    0.700   
🥈 2  Microsoft      0.721    BUY            0.680     0.720    0.750   
🥉 3  Amazon         0.656    HOLD           0.620     0.650    0.680   
📊 4  Tesla          0.634    HOLD           0.580     0.600    0.720   
```

## 🚀 Deployment Options

### Local Development
```bash
streamlit run app.py
```

### Production Deployment
```bash
# Using Streamlit Cloud
# 1. Push to GitHub
# 2. Connect at share.streamlit.io
# 3. Deploy with one click

# Using Docker
docker build -t vesprai .
docker run -p 8501:8501 vesprai
```

### API Deployment
```python
# Flask API example (see app.py for full Streamlit implementation)
from flask import Flask, jsonify, request
from src.investment_insight_generator import InvestmentInsightGenerator

app = Flask(__name__)
generator = InvestmentInsightGenerator()
generator.load_modules()

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    result = generator.generate_insight(
        data['company'], data['news'], data['document']
    )
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

## 🔧 Configuration

### Scoring Weight Adjustment
```python
# In src/investment_insight_generator.py
generator.weights = {
    'sentiment': 0.40,        # 40% weight on sentiment (default)
    'document_quality': 0.35, # 35% weight on document analysis  
    'risk_assessment': 0.25   # 25% weight on risk factors
}
```

### Model Paths (config.py)
```python
PATHS = {
    'final_model': Path('models/final_model'),
    'fraud_model': Path('models/best_fraud_scorer'),
    'train_dataset': Path('data/train_dataset'),
    # ... other paths
}
```

## 📈 Performance Optimization

- **GPU Acceleration**: Automatic MPS (Mac) / CUDA (NVIDIA) detection
- **Model Caching**: Streamlit `@st.cache_resource` for fast reloads
- **Batch Processing**: Process multiple queries efficiently
- **Response Time**: ~0.3-0.5s per query with cached models

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📚 Documentation

- **Notebooks**: Step-by-step walkthroughs in `/notebooks`
- **API Reference**: Docstrings in `/src` modules
- **Dashboard Guide**: Interactive help in Streamlit sidebar

## 🏆 Project Achievements

| Requirement | Target | Achieved | Status |
|-------------|--------|----------|--------|
| Sentiment F1 | ≥0.75 | 0.99 | ✅ +32% |
| Summarization ROUGE-L | ≥30 | 35 | ✅ +17% |
| Fraud AUC | ≥0.80 | 0.95 | ✅ +19% |
| RAG Chatbot | Complete | Integrated | ✅ |
| Streamlit Dashboard | Complete | 5 Tabs | ✅ |

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact & Support

- **Project Team**: Anisha Katiyar, Siddhi Rohan Chakka
- **Course**: MSML641 - Advanced NLP Applications
- **Program**: Masters in Data Science

### Issues & Support
- 🐛 **Bug Reports**: Open an issue with detailed description
- 💡 **Feature Requests**: Describe your use case and requirements
- 💬 **Questions**: Use GitHub Discussions

## 🙏 Acknowledgments

- **Financial PhraseBank Dataset**: Malo Pekka et al. (University of Helsinki)
- **PaySim Dataset**: E. A. Lopez-Rojas & S. Axelsson (Blekinge Institute of Technology)
- **Transformers Library**: Hugging Face team
- **Streamlit**: Streamlit Inc.
- **PyTorch Framework**: Meta AI Research

---

## 🚀 Quick Start Summary

```bash
# 1. Clone and setup
git clone <repository-url> && cd VesprAI
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Download data (PaySim for fraud detection)
kaggle datasets download -d ealaxi/paysim1 -p data/raw/ --unzip

# 3. Train models (run notebooks 01-07 in order)
jupyter notebook

# 4. Launch dashboard
streamlit run app.py

# 5. Open browser at http://localhost:8501
# ✅ VesprAI is ready!
```

**🎯 VesprAI delivers production-ready financial intelligence with state-of-the-art NLP integration!**

---

<div align="center">

### 📊 System Status

| Component | Status |
|-----------|--------|
| Sentiment Model | ✅ 99% Accuracy |
| Fraud Detector | ✅ 0.95 AUC |
| Summarizer | ✅ ROUGE-L 0.35 |
| Chatbot | ✅ Integrated |
| Dashboard | ✅ Live |

<br>

**Built with ❤️ for the future of financial AI**

<em>VesprAI - Where Financial Intelligence Meets Artificial Intelligence</em>

</div>
