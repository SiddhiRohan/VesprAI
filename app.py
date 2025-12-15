"""
VesprAI: Multi-Task Financial Intelligence Assistant
Streamlit Dashboard for Final Project Presentation

Run with: streamlit run app.py
"""

import streamlit as st
import sys
import os
from pathlib import Path
import time
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'src'))

# Page configuration
st.set_page_config(
    page_title="VesprAI - Financial Intelligence Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6B7280;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load all VesprAI models (cached for performance)"""
    models = {
        'sentiment': None,
        'summarizer': None,
        'fraud': None,
        'insights': None
    }
    status = {
        'sentiment': False,
        'summarizer': False,
        'fraud': False,
        'insights': False
    }
    
    # Load Sentiment Analyzer
    try:
        from transformers import pipeline
        from config import PATHS
        
        sentiment_model_path = PATHS.get('final_model', project_root / 'models' / 'final_model')
        if Path(sentiment_model_path).exists():
            models['sentiment'] = pipeline(
                "sentiment-analysis",
                model=str(sentiment_model_path),
                tokenizer=str(sentiment_model_path),
                return_all_scores=True
            )
            status['sentiment'] = True
    except Exception as e:
        st.sidebar.warning(f"Sentiment model not loaded: {e}")
    
    # Load Document Summarizer
    try:
        from src.document_summarizer import DocumentSummarizer
        models['summarizer'] = DocumentSummarizer(model_name="t5-small")
        status['summarizer'] = True
    except Exception as e:
        st.sidebar.warning(f"Summarizer not loaded: {e}")
    
    # Load Fraud Detector
    try:
        from src.unified_fraud_risk_scorer import UnifiedFraudRiskScorer
        import joblib
        
        models['fraud'] = UnifiedFraudRiskScorer(model_name="all-MiniLM-L6-v2")
        
        # Load trained weights
        fraud_model_path = project_root / 'notebooks' / 'models' / 'best_fraud_scorer'
        if not fraud_model_path.exists():
            fraud_model_path = project_root / 'models' / 'best_fraud_scorer'
        
        if (fraud_model_path / "best_classifier.joblib").exists():
            models['fraud'].hybrid_classifier = joblib.load(fraud_model_path / "best_classifier.joblib")
            models['fraud'].text_scaler = joblib.load(fraud_model_path / "best_text_scaler.joblib")
            models['fraud'].numeric_scaler = joblib.load(fraud_model_path / "best_numeric_scaler.joblib")
            models['fraud'].hybrid_trained = True
        status['fraud'] = True
    except Exception as e:
        st.sidebar.warning(f"Fraud detector not loaded: {e}")
    
    # Load Investment Insight Generator
    try:
        from src.investment_insight_generator import InvestmentInsightGenerator
        models['insights'] = InvestmentInsightGenerator()
        models['insights'].load_modules()
        status['insights'] = True
    except Exception as e:
        st.sidebar.warning(f"Insight generator not loaded: {e}")
    
    return models, status


# Sentiment label mapping
SENTIMENT_LABELS = {
    'LABEL_0': 'Negative',
    'LABEL_1': 'Neutral',
    'LABEL_2': 'Positive'
}


def analyze_sentiment(models, text):
    """Analyze sentiment of text"""
    if models['sentiment'] is None:
        return None
    
    result = models['sentiment'](text)
    scores = {}
    for item in result[0]:
        label = SENTIMENT_LABELS.get(item['label'], item['label'])
        scores[label] = item['score']
    
    best_label = max(scores, key=scores.get)
    return {
        'label': best_label,
        'confidence': scores[best_label],
        'scores': scores
    }


def summarize_document(models, text):
    """Summarize document"""
    if models['summarizer'] is None:
        return None
    
    summary = models['summarizer'].summarize(text)
    return {
        'summary': summary,
        'original_length': len(text),
        'summary_length': len(summary),
        'compression': len(summary) / len(text) if len(text) > 0 else 0
    }


def detect_fraud(models, text, amount=None):
    """Detect fraud risk"""
    if models['fraud'] is None:
        return None
    
    if amount:
        result = models['fraud'].score_transaction(
            text=text,
            method="hybrid",
            amount=amount,
            old_balance_org=amount * 2,
            new_balance_orig=amount,
            old_balance_dest=0,
            new_balance_dest=amount,
            trans_type="TRANSFER"
        )
    else:
        result = models['fraud'].score_transaction(text=text, method="unsupervised")
    
    return result


def generate_insight(models, company, news, document):
    """Generate investment insight"""
    if models['insights'] is None:
        return None
    
    return models['insights'].generate_insight(company, news, document)


def main():
    # Header
    st.markdown('<p class="main-header">🏦 VesprAI</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Multi-Task Financial Intelligence Assistant using NLP and RAG-based Conversational Reasoning</p>', unsafe_allow_html=True)
    
    # Load models
    with st.spinner("Loading VesprAI models..."):
        models, status = load_models()
    
    # Sidebar - System Status
    with st.sidebar:
        st.header("🔧 System Status")
        
        col1, col2 = st.columns(2)
        with col1:
            loaded = sum(status.values())
            st.metric("Modules Loaded", f"{loaded}/4")
        with col2:
            st.metric("Status", "✅ Ready" if loaded >= 3 else "⚠️ Partial")
        
        st.divider()
        
        st.subheader("Module Status")
        modules_info = [
            ("Sentiment Analyzer", status['sentiment'], "DistilBERT (99% acc)"),
            ("Document Summarizer", status['summarizer'], "T5-small (ROUGE-L 0.35)"),
            ("Fraud Detector", status['fraud'], "Hybrid ML (AUC 0.95)"),
            ("Investment Insights", status['insights'], "Integrated Pipeline")
        ]
        
        for name, loaded, desc in modules_info:
            icon = "✅" if loaded else "❌"
            st.write(f"{icon} **{name}**")
            if loaded:
                st.caption(f"   {desc}")
        
        st.divider()
        
        st.subheader("📊 Performance Metrics")
        metrics_df = pd.DataFrame({
            'Module': ['Sentiment', 'Fraud', 'Summarizer'],
            'Metric': ['Accuracy', 'AUC', 'ROUGE-L'],
            'Score': [0.99, 0.95, 0.35],
            'Target': [0.75, 0.80, 0.30]
        })
        
        for _, row in metrics_df.iterrows():
            achieved = "✅" if row['Score'] >= row['Target'] else "❌"
            st.write(f"{achieved} {row['Module']}: {row['Score']:.2f} (target: {row['Target']})")
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💬 Chatbot", 
        "📰 Sentiment Analysis", 
        "📝 Document Summarizer",
        "🔒 Fraud Detection",
        "📈 Investment Insights"
    ])
    
    # Tab 1: Chatbot Interface
    with tab1:
        st.header("💬 Financial Assistant Chatbot")
        st.write("Ask questions about sentiment, fraud, investments, or request document summaries.")
        
        # Chat interface
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask VesprAI anything about finance..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    response = process_chat_query(models, prompt)
                st.markdown(response)
            
            st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Quick action buttons
        st.divider()
        st.subheader("🚀 Quick Actions - Try These Examples")
        
        # Row 1: Sentiment Examples
        st.write("**📰 Sentiment Analysis**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📈 Bullish News"):
                query = "What's the sentiment of: Company reports record profits, exceeds analyst expectations, stock surges 15%"
                st.session_state.messages.append({"role": "user", "content": query})
                st.rerun()
        
        with col2:
            if st.button("📉 Bearish News"):
                query = "What's the sentiment of: Company misses earnings, announces layoffs, faces regulatory investigation"
                st.session_state.messages.append({"role": "user", "content": query})
                st.rerun()
        
        with col3:
            if st.button("⚖️ Neutral News"):
                query = "What's the sentiment of: Company maintains quarterly dividend, no changes to guidance"
                st.session_state.messages.append({"role": "user", "content": query})
                st.rerun()
        
        with col4:
            if st.button("🏦 Earnings Report"):
                query = "What's the sentiment of: Q3 revenue up 8% YoY, EPS of $2.15 beats estimates by $0.10"
                st.session_state.messages.append({"role": "user", "content": query})
                st.rerun()
        
        # Row 2: Fraud Detection Examples
        st.write("**🔒 Fraud Detection**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🚨 Suspicious Transfer"):
                query = "Is this suspicious? $50,000 transfer where sender balance went from $50,000 to $0, receiver balance unchanged"
                st.session_state.messages.append({"role": "user", "content": query})
                st.rerun()
        
        with col2:
            if st.button("✅ Normal Payment"):
                query = "Is this suspicious? $150 payment from account with $2000 balance to merchant account"
                st.session_state.messages.append({"role": "user", "content": query})
                st.rerun()
        
        with col3:
            if st.button("⚠️ Large Cash Out"):
                query = "Is this suspicious? $25,000 cash out to new account, sender balance depleted completely"
                st.session_state.messages.append({"role": "user", "content": query})
                st.rerun()
        
        with col4:
            if st.button("💳 Wire Transfer"):
                query = "Is this suspicious? International wire transfer of $10,000 to offshore account"
                st.session_state.messages.append({"role": "user", "content": query})
                st.rerun()
        
        # Row 3: Investment Insights Examples
        st.write("**📈 Investment Analysis**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🍎 Tech Giant"):
                query = "Should I invest in Apple stock?"
                st.session_state.messages.append({"role": "user", "content": query})
                st.rerun()
        
        with col2:
            if st.button("🚗 EV Company"):
                query = "Should I invest in Tesla stock?"
                st.session_state.messages.append({"role": "user", "content": query})
                st.rerun()
        
        with col3:
            if st.button("☁️ Cloud Leader"):
                query = "Should I invest in Microsoft stock?"
                st.session_state.messages.append({"role": "user", "content": query})
                st.rerun()
        
        with col4:
            if st.button("📦 E-Commerce"):
                query = "Should I invest in Amazon stock?"
                st.session_state.messages.append({"role": "user", "content": query})
                st.rerun()
        
        # Row 4: Summarization Examples
        st.write("**📝 Document Summarization**")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Earnings Summary"):
                query = "Summarize: The company reported quarterly revenue of $50.2 billion, representing a 12% increase year-over-year. Operating income grew 18% to $15.3 billion, driven by strong performance in cloud services and digital advertising. The company announced a new $10 billion share buyback program and increased its quarterly dividend by 5%. Management provided optimistic guidance for the upcoming quarter, citing continued demand for AI-related products and services."
                st.session_state.messages.append({"role": "user", "content": query})
                st.rerun()
        
        with col2:
            if st.button("📋 Risk Factors"):
                query = "Summarize: Key risk factors include exposure to foreign currency fluctuations, particularly in emerging markets where the company derives 30% of revenue. Supply chain disruptions remain a concern, with semiconductor shortages potentially impacting production timelines. Regulatory scrutiny in multiple jurisdictions could result in significant fines or operational restrictions. Competition from both established players and new entrants continues to pressure margins in core business segments."
                st.session_state.messages.append({"role": "user", "content": query})
                st.rerun()
    
    # Tab 2: Sentiment Analysis
    with tab2:
        st.header("📰 News Sentiment Analyzer")
        st.write("Analyze sentiment of financial news using our trained DistilBERT model (99% accuracy)")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            sentiment_text = st.text_area(
                "Enter financial news or text:",
                value="Apple reported strong quarterly earnings with revenue growth of 15% year-over-year, beating analyst expectations.",
                height=150
            )
            
            if st.button("Analyze Sentiment", type="primary"):
                if sentiment_text:
                    with st.spinner("Analyzing sentiment..."):
                        result = analyze_sentiment(models, sentiment_text)
                    
                    if result:
                        st.success(f"**Sentiment: {result['label']}** (Confidence: {result['confidence']:.1%})")
                        
                        # Create gauge chart
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=result['scores'].get('Positive', 0) * 100,
                            title={'text': "Positive Sentiment"},
                            gauge={
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "#22c55e"},
                                'steps': [
                                    {'range': [0, 33], 'color': "#fee2e2"},
                                    {'range': [33, 66], 'color': "#fef3c7"},
                                    {'range': [66, 100], 'color': "#dcfce7"}
                                ]
                            }
                        ))
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show all scores
                        scores_df = pd.DataFrame({
                            'Sentiment': list(result['scores'].keys()),
                            'Score': list(result['scores'].values())
                        })
                        st.bar_chart(scores_df.set_index('Sentiment'))
                    else:
                        st.error("Sentiment model not available")
        
        with col2:
            st.subheader("Model Info")
            st.info("""
            **Model**: DistilBERT (fine-tuned)
            
            **Training Data**: Financial PhraseBank
            
            **Accuracy**: 99.05%
            
            **F1 Score**: 99.05%
            
            **Classes**: Positive, Neutral, Negative
            """)
    
    # Tab 3: Document Summarizer
    with tab3:
        st.header("📝 Financial Document Summarizer")
        st.write("Summarize SEC filings, earnings reports, and financial documents using T5")
        
        doc_text = st.text_area(
            "Enter document text to summarize:",
            value="""Apple Inc. reported Q1 2025 revenue of $97.8 billion, up 8% year-over-year, driven by strong iPhone and Services growth. iPhone revenue was $49.2 billion, up 12% year-over-year. Services revenue reached an all-time high of $26.1 billion, up 14% year-over-year. Mac revenue was $8.2 billion and iPad revenue was $7.1 billion. Operating income increased to $31.2 billion and the company returned $27 billion to shareholders through dividends and share repurchases during the quarter.""",
            height=200
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Summarize Document", type="primary"):
                if doc_text:
                    with st.spinner("Generating summary..."):
                        result = summarize_document(models, doc_text)
                    
                    if result:
                        st.subheader("Summary")
                        st.success(result['summary'])
                        
                        # Stats
                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("Original", f"{result['original_length']} chars")
                        col_b.metric("Summary", f"{result['summary_length']} chars")
                        col_c.metric("Compression", f"{result['compression']:.1%}")
                    else:
                        st.error("Summarizer not available")
        
        with col2:
            st.subheader("Model Info")
            st.info("""
            **Model**: T5-small (pretrained)
            
            **ROUGE-L Score**: 0.35
            
            **Target**: ≥ 0.30 ✅
            
            **Use Case**: SEC filings, earnings reports
            """)
    
    # Tab 4: Fraud Detection
    with tab4:
        st.header("🔒 Fraud Risk Detection")
        st.write("Detect suspicious transactions using our hybrid NLP + ML model (AUC 0.95)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            transaction_desc = st.text_area(
                "Transaction Description:",
                value="A financial transaction where account C5555555 sent 50000.00 units to account C8888888. Sender balance changed from 50000.00 to 0.00.",
                height=100
            )
            
            transaction_amount = st.number_input(
                "Transaction Amount ($):",
                min_value=0.0,
                value=50000.0,
                step=1000.0
            )
            
            if st.button("Analyze Transaction", type="primary"):
                with st.spinner("Analyzing fraud risk..."):
                    result = detect_fraud(models, transaction_desc, transaction_amount)
                
                if result:
                    risk_level = result.get('risk_level', 'UNKNOWN')
                    risk_score = result.get('risk_score', result.get('risk_percentage', 0))
                    
                    # Color based on risk
                    if risk_level in ['HIGH', 'CRITICAL']:
                        st.error(f"🚨 **Risk Level: {risk_level}**")
                    elif risk_level == 'MEDIUM':
                        st.warning(f"⚠️ **Risk Level: {risk_level}**")
                    else:
                        st.success(f"✅ **Risk Level: {risk_level}**")
                    
                    # Risk gauge
                    risk_pct = float(risk_score) * 100 if risk_score <= 1 else risk_score
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=risk_pct,
                        title={'text': "Fraud Risk Score"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "#ef4444"},
                            'steps': [
                                {'range': [0, 30], 'color': "#dcfce7"},
                                {'range': [30, 70], 'color': "#fef3c7"},
                                {'range': [70, 100], 'color': "#fee2e2"}
                            ]
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Fraud detector not available")
        
        with col2:
            st.subheader("Model Info")
            st.info("""
            **Model**: Hybrid (NLP + Numeric)
            
            **Text Encoder**: all-MiniLM-L6-v2
            
            **Classifier**: Logistic Regression
            
            **AUC Score**: 0.9563
            
            **Target**: ≥ 0.80 ✅
            
            **Training Data**: PaySim dataset
            """)
            
            st.subheader("Red Flags")
            st.warning("""
            - Balance going to exactly $0
            - Large transfers to new accounts
            - Transactions just below limits
            - Rapid successive transfers
            """)
    
    # Tab 5: Investment Insights
    with tab5:
        st.header("📈 Investment Insight Generator")
        st.write("Get comprehensive investment analysis combining sentiment, financials, and risk assessment")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            company_name = st.text_input("Company Name:", value="Apple Inc.")
            
            news_text = st.text_area(
                "Recent News:",
                value="Apple reported strong quarterly earnings with iPhone sales exceeding expectations. The company beat analyst estimates with revenue growth of 8% year-over-year.",
                height=100
            )
            
            doc_text = st.text_area(
                "Financial Document:",
                value="Apple Inc. demonstrates solid financial performance with revenue of $394.3 billion and net income of $99.8 billion. The company maintains strong cash position of $165 billion.",
                height=100
            )
            
            if st.button("Generate Investment Insight", type="primary"):
                with st.spinner("Generating comprehensive analysis..."):
                    result = generate_insight(models, company_name, news_text, doc_text)
                
                if result:
                    # Recommendation banner
                    rec = result.get('recommendation', 'HOLD')
                    if 'BUY' in rec:
                        st.success(f"📈 **Recommendation: {rec}**")
                    elif 'SELL' in rec:
                        st.error(f"📉 **Recommendation: {rec}**")
                    else:
                        st.info(f"⚖️ **Recommendation: {rec}**")
                    
                    # Scores
                    col_a, col_b, col_c, col_d = st.columns(4)
                    col_a.metric("Overall Score", f"{result.get('overall_score', 0):.1%}")
                    col_b.metric("Sentiment", f"{result.get('sentiment_score', 0):.1%}")
                    col_c.metric("Financials", f"{result.get('document_score', 0):.1%}")
                    col_d.metric("Risk Score", f"{result.get('risk_score', 0):.1%}")
                    
                    # Narrative
                    st.subheader("Analysis Summary")
                    st.write(result.get('narrative', result.get('explanation', 'Analysis complete.')))
                    
                    # Radar chart of scores
                    categories = ['Sentiment', 'Financials', 'Risk', 'Overall']
                    values = [
                        result.get('sentiment_score', 0) * 100,
                        result.get('document_score', 0) * 100,
                        result.get('risk_score', 0) * 100,
                        result.get('overall_score', 0) * 100
                    ]
                    
                    fig = go.Figure(data=go.Scatterpolar(
                        r=values + [values[0]],
                        theta=categories + [categories[0]],
                        fill='toself',
                        fillcolor='rgba(99, 102, 241, 0.3)',
                        line=dict(color='rgb(99, 102, 241)')
                    ))
                    fig.update_layout(
                        polar=dict(radialaxis=dict(range=[0, 100])),
                        showlegend=False,
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Insight generator not available")
        
        with col2:
            st.subheader("Scoring Weights")
            weights_df = pd.DataFrame({
                'Component': ['Sentiment', 'Document Quality', 'Risk Assessment'],
                'Weight': [40, 35, 25]
            })
            
            fig = px.pie(weights_df, values='Weight', names='Component',
                        color_discrete_sequence=['#6366f1', '#8b5cf6', '#a855f7'])
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Recommendation Scale")
            st.write("""
            - **STRONG BUY**: Score ≥ 80%
            - **BUY**: Score ≥ 70%
            - **HOLD**: Score ≥ 60%
            - **WEAK HOLD**: Score ≥ 40%
            - **SELL**: Score < 40%
            """)


def process_chat_query(models, query):
    """Process chatbot query and route to appropriate module"""
    query_lower = query.lower()
    
    # Detect intent
    if any(kw in query_lower for kw in ['sentiment', 'feeling', 'mood', 'positive', 'negative']):
        result = analyze_sentiment(models, query)
        if result:
            return f"""**📊 Sentiment Analysis**

**Sentiment:** {result['label']}
**Confidence:** {result['confidence']:.1%}

**Detailed Scores:**
- Positive: {result['scores'].get('Positive', 0):.1%}
- Neutral: {result['scores'].get('Neutral', 0):.1%}
- Negative: {result['scores'].get('Negative', 0):.1%}

*Analysis by trained DistilBERT model (99% accuracy)*"""
    
    elif any(kw in query_lower for kw in ['fraud', 'suspicious', 'transaction', 'risk']):
        import re
        amount_match = re.search(r'\$?([\d,]+(?:\.\d{2})?)', query)
        amount = float(amount_match.group(1).replace(',', '')) if amount_match else None
        
        result = detect_fraud(models, query, amount)
        if result:
            risk_level = result.get('risk_level', 'UNKNOWN')
            risk_score = result.get('risk_score', result.get('risk_percentage', 0))
            return f"""**🔒 Fraud Risk Assessment**

**Risk Level:** {risk_level}
**Risk Score:** {float(risk_score):.1%}
**Fraud Prediction:** {'⚠️ SUSPICIOUS' if risk_level in ['HIGH', 'CRITICAL'] else '✅ APPEARS LEGITIMATE'}

*Analysis by hybrid fraud detection model (AUC 0.95)*"""
    
    elif any(kw in query_lower for kw in ['invest', 'buy', 'sell', 'stock', 'portfolio']):
        companies = ['Apple', 'Tesla', 'Microsoft', 'Amazon', 'Google', 'Meta']
        company = next((c for c in companies if c.lower() in query_lower), 'the company')
        
        result = generate_insight(
            models, 
            company,
            f"{company} shows positive market momentum with strong performance.",
            f"{company} demonstrates solid financial fundamentals."
        )
        if result:
            return f"""**📈 Investment Analysis: {company}**

**Recommendation:** {result.get('recommendation', 'HOLD')}
**Confidence:** {result.get('overall_score', 0):.1%}

**Component Scores:**
- Sentiment: {result.get('sentiment_score', 0):.1%}
- Financial Health: {result.get('document_score', 0):.1%}
- Risk Assessment: {result.get('risk_score', 0):.1%}

**Summary:** {result.get('narrative', result.get('explanation', 'Analysis complete.'))}"""
    
    elif any(kw in query_lower for kw in ['summarize', 'summary', 'brief']):
        result = summarize_document(models, query)
        if result:
            return f"""**📝 Document Summary**

{result['summary']}

*Compression: {result['compression']:.1%} | Original: {result['original_length']} chars*"""
    
    # Default response
    return """**🤖 VesprAI Assistant**

I can help you with:

1. **Sentiment Analysis** - "What's the sentiment of: [your text]"
2. **Fraud Detection** - "Is this transaction suspicious? [description]"
3. **Investment Insights** - "Should I invest in [company]?"
4. **Document Summary** - "Summarize: [your document]"

Try one of these queries!"""


if __name__ == "__main__":
    main()
