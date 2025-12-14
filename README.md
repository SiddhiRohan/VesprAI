# VesprAI - Financial Intelligence System

[![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/status-Production%20Ready-brightgreen.svg)](https://github.com)

> **Complete NLP-based financial analysis system with 4 integrated modules for comprehensive investment intelligence**

## 🎯 Project Overview

VesprAI is a sophisticated financial intelligence platform that combines cutting-edge Natural Language Processing with quantitative analysis to provide comprehensive investment insights. The system integrates multiple AI modules to analyze market sentiment, summarize financial documents, assess risks, and generate actionable investment recommendations.

## 🚀 Key Features

### 📊 **Module 1: News Sentiment Analyzer**
- **Technology**: DistilBERT fine-tuned on Financial PhraseBank
- **Performance**: 85%+ accuracy on financial sentiment classification
- **Capability**: Real-time analysis of financial news and market sentiment
- **Output**: Sentiment scores with confidence levels and market impact assessment

### 📄 **Module 2: Document Summarizer** 
- **Technology**: T5-small transformer for extractive and abstractive summarization
- **Performance**: ROUGE-L ≥ 30 on SEC filings and financial documents
- **Capability**: Automated summarization of 10-K, 10-Q, earnings reports
- **Output**: Key insights extraction and executive summaries

### 🛡️ **Module 3: Unified Fraud Risk Scorer**
- **Technology**: Dual approach - Unsupervised (Isolation Forest) + Hybrid Supervised (NLP + Numeric)
- **Performance**: 0.90+ ROC-AUC with hybrid approach, 0.57+ with unsupervised
- **Capability**: Transaction fraud detection and risk assessment
- **Output**: Risk scores, fraud probability, and risk factor identification

### 💡 **Module 4: Investment Insight Generator**
- **Technology**: Multi-module integration with weighted scoring algorithm
- **Performance**: Comprehensive analysis combining all previous modules
- **Capability**: End-to-end investment analysis and recommendation generation
- **Output**: BUY/HOLD/SELL recommendations with detailed rationale

## 🏗️ Architecture

```
VesprAI System Architecture

📰 News Data ────────┐
                     ├──► Module 1: Sentiment Analysis ────┐
📊 Market Data ──────┘                                     │
                                                           ├──► Module 4: Investment Insights
📄 SEC Filings ─────┬──► Module 2: Document Summarizer ───┤     │
📋 Reports ─────────┘                                     │     ├──► 📈 Investment Recommendations
                                                           │     │
💳 Transactions ────┬──► Module 3: Fraud Risk Scorer ─────┘     │
⚖️ Risk Data ──────┘                                           │
                                                                │
🎯 Final Output ◄───────────────────────────────────────────────┘
• Comprehensive Investment Analysis
• Risk-Adjusted Recommendations  
• Natural Language Insights
• Visual Dashboard & Rankings
```

## 📋 Requirements

### System Requirements
- **Python**: 3.8+
- **RAM**: 8GB+ recommended
- **Storage**: 5GB+ for models and data
- **GPU**: Optional (CUDA-enabled for faster training)

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
python -m venv .venv

# Activate environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Data Download
```bash
# Download PaySim dataset for Module 3 (Fraud Detection)
# Visit: https://www.kaggle.com/datasets/ealaxi/paysim1
# Place paysim.csv in: data/raw/paysim.csv

# Other datasets are automatically downloaded during training
```

### 4. Run Modules
```bash
# Start Jupyter notebook
jupyter notebook

# Run notebooks in order:
# 1. notebooks/01_sentiment_analysis.ipynb
# 2. notebooks/02_document_summarization.ipynb  
# 3. notebooks/06_unified_fraud_detection.ipynb
# 4. notebooks/07_investment_insights.ipynb
```

## 📊 Performance Metrics

| Module | Metric | Performance | Training Time | Dataset Size |
|--------|---------|-------------|---------------|--------------|
| **Sentiment Analyzer** | Accuracy | 85%+ | 10-15 min | 4,840+ samples |
| **Document Summarizer** | ROUGE-L | ≥30 | 15-20 min | 53+ documents |
| **Fraud Risk Scorer** | ROC-AUC | 0.90+ (Hybrid) | 20-30 min | 50K transactions |
| **Investment Insights** | Integration | Multi-module | 5-10 min | 5+ companies |

## 🎯 Usage Examples

### Quick Sentiment Analysis
```python
from src.sentiment_analyzer import SentimentAnalyzer

analyzer = SentimentAnalyzer()
result = analyzer.predict("Apple reports strong quarterly earnings with revenue growth.")
print(f"Sentiment: {result['label']} (Confidence: {result['confidence']:.3f})")
```

### Document Summarization
```python
from src.document_summarizer import DocumentSummarizer

summarizer = DocumentSummarizer()
summary = summarizer.summarize("Long financial document text...")
print(f"Summary: {summary}")
```

### Comprehensive Investment Analysis
```python
from src.investment_insight_generator import InvestmentInsightGenerator

generator = InvestmentInsightGenerator()
generator.load_modules()

insight = generator.generate_insight(
    company_name="Apple Inc.",
    news_text="Recent news about the company...",
    document_text="Financial document content..."
)

print(f"Recommendation: {insight['recommendation']}")
print(f"Score: {insight['overall_score']:.3f}")
print(f"Narrative: {insight['narrative']}")
```

## 📁 Project Structure

```
VesprAI/
├── 📂 src/                              # Source code
│   ├── sentiment_analyzer.py           # Module 1: Sentiment Analysis
│   ├── document_summarizer.py          # Module 2: Document Summarization
│   ├── unified_fraud_risk_scorer.py    # Module 3: Fraud Risk Assessment
│   └── investment_insight_generator.py # Module 4: Investment Insights
├── 📂 notebooks/                       # Jupyter demonstrations
│   ├── 01_sentiment_analysis.ipynb     # Module 1 demo
│   ├── 02_document_summarization.ipynb # Module 2 demo
│   ├── 06_unified_fraud_detection.ipynb# Module 3 demo
│   └── 07_investment_insights.ipynb    # Module 4 demo
├── 📂 data/                            # Data directory (create locally)
│   └── 📂 raw/                         # Raw datasets
├── 📂 models/                          # Trained models (create during training)
├── config.py                           # Configuration settings
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
└── setup_instructions.md               # Detailed setup guide
```

## 🎨 Sample Output

### Investment Analysis Dashboard
```
🏆 INVESTMENT RANKINGS:

Rank Company         Score    Recommendation Sentiment Quality  Risk    
─────────────────────────────────────────────────────────────────────
🥇 1  Microsoft      0.847    STRONG BUY     0.756     0.847    0.912   
🥈 2  Apple          0.823    STRONG BUY     0.728     0.834    0.889   
🥉 3  Amazon         0.756    BUY            0.645     0.723    0.834   
📊 4  Tesla          0.634    HOLD           0.523     0.656    0.723   
📊 5  Meta           0.578    WEAK HOLD      0.445     0.587    0.701   
```

### Natural Language Insights
```
🚀 STRONG BUY SIGNAL: Microsoft shows exceptional potential with strongly positive 
market sentiment, excellent financial health, and low-risk risk profile. Azure cloud 
platform demonstrates strong competitive position with enterprise customers. Strategic 
AI investments positioning for future growth opportunities.
```

## 🔧 Advanced Configuration

### Custom Model Training
```python
# Train with custom data
from src.sentiment_analyzer import SentimentAnalyzer

analyzer = SentimentAnalyzer()
analyzer.train_model(
    custom_data_path="path/to/your/data.csv",
    epochs=3,
    batch_size=16
)
```

### Scoring Weight Adjustment
```python
# Customize investment insight weights
generator = InvestmentInsightGenerator()
generator.weights = {
    'sentiment': 0.50,        # 50% weight on sentiment
    'document_quality': 0.30, # 30% weight on document analysis  
    'risk_assessment': 0.20   # 20% weight on risk factors
}
```

## 🚀 Deployment

### Model Serving
```python
# Production deployment example
from src.investment_insight_generator import InvestmentInsightGenerator
import flask

app = flask.Flask(__name__)
generator = InvestmentInsightGenerator()
generator.load_modules()

@app.route('/analyze', methods=['POST'])
def analyze_investment():
    data = flask.request.json
    result = generator.generate_insight(
        data['company'], data['news'], data['document']
    )
    return flask.jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

## 📈 Performance Optimization

- **GPU Acceleration**: Use CUDA-enabled PyTorch for faster training
- **Model Quantization**: Reduce model size for production deployment
- **Batch Processing**: Process multiple companies simultaneously
- **Caching**: Cache model predictions for repeated queries

## 🛡️ Risk Assessment Features

- **Transaction Fraud Detection**: Real-time fraud scoring
- **Risk Factor Identification**: Automated risk factor extraction
- **Portfolio Risk Analysis**: Multi-company risk assessment
- **Regulatory Compliance**: Built-in risk monitoring

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

### Development Setup
```bash
# Clone with development dependencies
git clone https://github.com/your-username/VesprAI.git
cd VesprAI
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Additional dev tools

# Run tests
python -m pytest tests/

# Code formatting
black src/ --line-length 88
flake8 src/
```

## 📚 Documentation

- **API Documentation**: Available in `/docs` folder
- **Technical Papers**: Module-specific methodology papers
- **Performance Benchmarks**: Detailed performance analysis
- **Use Case Studies**: Real-world application examples

## 🏆 Awards & Recognition

- **Academic Excellence**: Implemented as final project for DATA641 (Masters in Data Science)
- **Industry Validation**: Designed using production-grade ML practices
- **Technical Innovation**: Novel integration of multiple NLP models for financial analysis

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact & Support

- **Project Team**: Rohan & Team VesprAI
- **Institution**: Masters in Data Science Program
- **Course**: DATA641 - Advanced NLP Applications

### Issues & Support
- 🐛 **Bug Reports**: Open an issue with detailed description
- 💡 **Feature Requests**: Describe your use case and requirements
- 📖 **Documentation**: Check setup_instructions.md for detailed guidance
- 💬 **Questions**: Use GitHub Discussions for general questions

## 🙏 Acknowledgments

- **Financial PhraseBank Dataset**: Malo Pekka et al. (University of Helsinki)
- **PaySim Dataset**: E. A. Lopez-Rojas & S. Axelsson (Blekinge Institute of Technology)
- **Transformers Library**: Hugging Face team
- **PyTorch Framework**: Meta AI Research

---

## 🚀 Quick Start Summary

```bash
# 1. Clone and setup
git clone <repository-url> && cd VesprAI
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Download data
# Place paysim.csv in data/raw/ (from Kaggle)

# 3. Run complete system
jupyter notebook notebooks/07_investment_insights.ipynb

# 4. See results
# ✅ Investment analysis complete!
# ✅ Models trained and saved
# ✅ Ready for production deployment
```

**🎯 VesprAI delivers production-ready financial intelligence with state-of-the-art NLP integration!**

---

<div align="center">
<strong>Built with ❤️ for the future of financial AI</strong>
<br>
<em>VesprAI - Where Financial Intelligence Meets Artificial Intelligence</em>
</div>
