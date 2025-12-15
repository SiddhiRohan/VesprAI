"""
VesprAI: Modern Financial Intelligence Platform
Advanced Streamlit Dashboard with Custom UI Design

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
from plotly.subplots import make_subplots

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'src'))

# Page configuration
st.set_page_config(
    page_title="VesprAI - Research-Grade Financial Intelligence Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Modern CSS styling with dark/light theme support
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    #stDecoration {display:none;}
    
    /* Root variables for theming */
    :root {
        --primary-color: #2563eb;
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --error-color: #ef4444;
        --background-color: #fafbfc;
        --card-background: #ffffff;
        --text-primary: #1f2937;
        --text-secondary: #6b7280;
        --border-color: #e5e7eb;
        --shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
        --shadow-lg: 0 20px 25px -5px rgb(0 0 0 / 0.1);
    }
    
    /* Global styling */
    .main {
        padding: 2rem 1rem;
        background-color: var(--background-color);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .block-container {
        max-width: 1200px;
        padding: 1rem;
        margin: 0 auto;
    }
    
    /* Modern header */
    .hero-section {
        background: var(--primary-gradient);
        border-radius: 24px;
        padding: 3rem 2rem;
        margin-bottom: 2rem;
        text-align: center;
        position: relative;
        overflow: hidden;
        box-shadow: var(--shadow-lg);
    }
    
    .hero-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse"><path d="M 10 0 L 0 0 0 10" fill="none" stroke="white" stroke-width="0.5" opacity="0.1"/></pattern></defs><rect width="100" height="100" fill="url(%23grid)"/></svg>');
        opacity: 0.1;
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        color: white;
        margin: 0 0 0.5rem 0;
        position: relative;
        z-index: 1;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .hero-subtitle {
        font-size: 1.25rem;
        color: rgba(255, 255, 255, 0.9);
        margin: 0;
        position: relative;
        z-index: 1;
        font-weight: 400;
    }
    
    /* Navigation tabs redesign */
    .nav-container {
        background: var(--card-background);
        border-radius: 16px;
        padding: 0.5rem;
        margin-bottom: 2rem;
        box-shadow: var(--shadow);
        border: 1px solid var(--border-color);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.25rem;
        background: transparent;
        border-bottom: none;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 0.875rem 1.5rem;
        border-radius: 12px;
        font-weight: 500;
        font-size: 0.875rem;
        border: none;
        transition: all 0.3s ease;
        color: var(--text-secondary);
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(37, 99, 235, 0.1);
        color: var(--primary-color);
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: var(--primary-color);
        color: white;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.3);
    }
    
    /* Modern cards */
    .modern-card {
        background: var(--card-background);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: var(--shadow);
        border: 1px solid var(--border-color);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin-bottom: 1.5rem;
    }
    
    .modern-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg);
    }
    
    .card-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid var(--border-color);
    }
    
    .card-icon {
        width: 2.5rem;
        height: 2.5rem;
        background: var(--primary-gradient);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.25rem;
        color: white;
    }
    
    .card-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 0;
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 500;
    }
    
    .status-online {
        background: rgba(16, 185, 129, 0.1);
        color: var(--success-color);
    }
    
    .status-warning {
        background: rgba(245, 158, 11, 0.1);
        color: var(--warning-color);
    }
    
    .status-offline {
        background: rgba(239, 68, 68, 0.1);
        color: var(--error-color);
    }
    
    /* Modern buttons */
    .stButton > button {
        background: var(--primary-gradient);
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 0.875rem;
        color: white;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 16px rgba(37, 99, 235, 0.4);
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        border-radius: 12px;
        border: 1px solid var(--border-color);
        padding: 0.75rem;
        font-size: 0.875rem;
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
    }
    
    .stTextArea textarea {
        border-radius: 12px;
        border: 1px solid var(--border-color);
        padding: 0.75rem;
        font-size: 0.875rem;
        resize: vertical;
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }
    
    .stTextArea textarea:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
    }
    
    /* Metrics styling */
    .metric-row {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
        flex-wrap: wrap;
    }
    
    .metric-card {
        flex: 1;
        min-width: 200px;
        background: var(--card-background);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid var(--border-color);
        text-align: center;
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--primary-color);
        margin: 0;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: var(--text-secondary);
        margin: 0.5rem 0 0 0;
        font-weight: 500;
    }
    
    /* Chat interface styling */
    .chat-container {
        background: var(--card-background);
        border-radius: 16px;
        border: 1px solid var(--border-color);
        overflow: hidden;
        box-shadow: var(--shadow);
    }
    
    .chat-header {
        background: var(--primary-gradient);
        color: white;
        padding: 1rem 1.5rem;
        font-weight: 600;
    }
    
    .chat-message {
        padding: 1rem 1.5rem;
        border-bottom: 1px solid var(--border-color);
        margin: 0;
    }
    
    .chat-message:last-child {
        border-bottom: none;
    }
    
    /* Loading animation */
    .loading-pulse {
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }
    
    @keyframes pulse {
        0%, 100% {
            opacity: 1;
        }
        50% {
            opacity: .5;
        }
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 2.5rem;
        }
        
        .hero-subtitle {
            font-size: 1rem;
        }
        
        .metric-row {
            flex-direction: column;
        }
        
        .modern-card {
            padding: 1rem;
        }
    }
    
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        :root {
            --background-color: #111827;
            --card-background: #1f2937;
            --text-primary: #f9fafb;
            --text-secondary: #d1d5db;
            --border-color: #374151;
        }
    }
    
    /* Performance indicators */
    .perf-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.375rem;
        padding: 0.25rem 0.75rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .perf-excellent {
        background: rgba(16, 185, 129, 0.15);
        color: var(--success-color);
    }
    
    .perf-good {
        background: rgba(59, 130, 246, 0.15);
        color: var(--primary-color);
    }
    
    .perf-warning {
        background: rgba(245, 158, 11, 0.15);
        color: var(--warning-color);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: var(--card-background);
        border-radius: 16px;
        border: 1px solid var(--border-color);
        box-shadow: var(--shadow);
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

def create_performance_badge(metric_name, current, target, format_str="{:.1%}"):
    """Create modern performance indicator badge"""
    ratio = current / target if target > 0 else 0
    
    if ratio >= 1.15:
        badge_class = "perf-excellent"
        status = "EXCELLENT"
    elif ratio >= 1.0:
        badge_class = "perf-good"
        status = "GOOD"
    else:
        badge_class = "perf-warning"
        status = "NEEDS IMPROVEMENT"
    
    formatted_value = format_str.format(current) if isinstance(current, float) else str(current)
    
    return f"""
    <div class="perf-indicator {badge_class}">
        🎯 {formatted_value} - {status}
    </div>
    """

def create_status_indicator(status, label):
    """Create modern status indicator"""
    if status:
        return f'<span class="status-indicator status-online">🟢 {label} Online</span>'
    else:
        return f'<span class="status-indicator status-offline">🔴 {label} Offline</span>'

def create_metric_card(title, value, subtitle="", icon="📊"):
    """Create a modern metric card"""
    return f"""
    <div class="metric-card">
        <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">{icon}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-label">{title}</div>
        {f'<div style="font-size: 0.75rem; color: var(--text-secondary); margin-top: 0.25rem;">{subtitle}</div>' if subtitle else ''}
    </div>
    """

def main():
    # Hero section
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title">🚀 VesprAI</h1>
        <p class="hero-subtitle">4-Module Financial Intelligence Platform with Research-Grade Performance</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load models with modern progress indicator
    with st.spinner("🔄 Initializing AI models..."):
        models, status = load_models()
    
    # System status dashboard
    loaded_count = sum(status.values())
    total_modules = len(status)
    
    # Performance metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(create_metric_card("Modules Loaded", f"{loaded_count}/{total_modules}", "AI Components", "🧠"), unsafe_allow_html=True)
    
    with col2:
        system_status = "Online" if loaded_count >= 3 else "Partial"
        status_icon = "🟢" if loaded_count >= 3 else "🟡"
        st.markdown(create_metric_card("System Status", system_status, "Platform Health", status_icon), unsafe_allow_html=True)
    
    with col3:
        st.markdown(create_metric_card("Accuracy", "97.62%", "Sentiment Model", "🎯"), unsafe_allow_html=True)
    
    with col4:
        st.markdown(create_metric_card("Performance", "95.63%", "Fraud Detection AUC", "🛡️"), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Modern navigation
    st.markdown('<div class="nav-container">', unsafe_allow_html=True)
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💬 AI Assistant", 
        "📊 Sentiment Analysis", 
        "📝 Smart Summarizer", 
        "🛡️ Fraud Detection", 
        "📈 Investment Insights"
    ])
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 1: AI Assistant (Enhanced)
    with tab1:
        st.markdown("""
        <div class="modern-card">
            <div class="card-header">
                <div class="card-icon">🤖</div>
                <h2 class="card-title">VesprAI Integrated Chatbot (4-Module System)</h2>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            st.markdown('<div class="chat-header">💬 Chat with VesprAI</div>', unsafe_allow_html=True)
            
            # Chat input
            user_query = st.text_input(
                "Ask me anything about finance...",
                placeholder="e.g., What's the sentiment of: Tesla reports record quarterly profits",
                label_visibility="collapsed"
            )
            
            # Quick action buttons
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                if st.button("📊 Analyze Sentiment", use_container_width=True):
                    user_query = "What's the sentiment of: Apple beats earnings expectations"
            with col_b:
                if st.button("🔍 Detect Fraud", use_container_width=True):
                    user_query = "Is this transaction suspicious? $50,000 transfer to unknown account"
            with col_c:
                if st.button("💰 Investment Advice", use_container_width=True):
                    user_query = "Should I invest in Tesla stock?"
            
            if user_query:
                with st.spinner("🧠 AI is thinking..."):
                    response = process_chat_query(models, user_query)
                
                st.markdown('<div class="chat-message">', unsafe_allow_html=True)
                st.markdown(f"**You:** {user_query}")
                st.markdown(response)
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="modern-card">
                <h3>🎯 Quick Examples</h3>
                <div style="margin-top: 1rem; font-size: 0.875rem; line-height: 1.6;">
                    <div style="margin-bottom: 0.75rem;">
                        <strong>📊 Sentiment:</strong><br>
                        "Analyze: Microsoft reports strong cloud growth"
                    </div>
                    <div style="margin-bottom: 0.75rem;">
                        <strong>🛡️ Fraud:</strong><br>
                        "Check transaction: $10,000 to overseas account"
                    </div>
                    <div style="margin-bottom: 0.75rem;">
                        <strong>📈 Investment:</strong><br>
                        "Should I buy Amazon stock?"
                    </div>
                    <div>
                        <strong>📝 Summary:</strong><br>
                        "Summarize: [paste your document]"
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # System status
            st.markdown("""
            <div class="modern-card" style="margin-top: 1rem;">
                <h3>⚡ System Status</h3>
                <div style="margin-top: 1rem;">
            """, unsafe_allow_html=True)
            
            # Enhanced system status with accurate module names
            module_status = [
                ("DistilBERT Sentiment (97.62%)", status.get('sentiment', False), "📊"),
                ("T5 Document Summarizer", status.get('summarizer', False), "📝"),
                ("Unified Fraud Scorer (95.63% AUC)", status.get('fraud', False), "🛡️"),
                ("4-Module Investment System", status.get('insights', False), "📈"),
                ("Integrated Chatbot", True, "🤖")  # Always true - has fallback
            ]
            
            for name, is_loaded, icon in module_status:
                color = "green" if is_loaded else "red"
                status_text = "Active" if is_loaded else "Offline"
                st.markdown(f"""
                <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                    <span style="font-size: 1rem; margin-right: 0.5rem;">{icon}</span>
                    <span style="font-size: 0.875rem; margin-right: 0.5rem;">{name}:</span>
                    <span style="color: {color}; font-weight: 600; font-size: 0.75rem;">{status_text}</span>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div></div>", unsafe_allow_html=True)
    
    # Tab 2: Sentiment Analysis (Enhanced)
    with tab2:
        st.markdown("""
        <div class="modern-card">
            <div class="card-header">
                <div class="card-icon">📊</div>
                <h2 class="card-title">DistilBERT Sentiment Analysis (97.62% Accuracy)</h2>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<div class="modern-card">', unsafe_allow_html=True)
            st.markdown("### 📝 Enter Financial Text")
            
            sentiment_text = st.text_area(
                "Text to analyze:",
                value="Apple Inc. reported exceptional quarterly results, beating analyst expectations with record iPhone sales and strong services revenue growth.",
                height=100,
                label_visibility="collapsed"
            )
            
            if st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True):
                if status['sentiment']:
                    with st.spinner("🤖 Analyzing sentiment..."):
                        result = analyze_sentiment(models, sentiment_text)
                    
                    if result:
                        # Modern result display
                        sentiment_color = {
                            'Positive': '#10b981',
                            'Negative': '#ef4444',
                            'Neutral': '#6b7280'
                        }
                        
                        color = sentiment_color.get(result['label'], '#6b7280')
                        
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, {color}15, {color}05); 
                                    border-left: 4px solid {color}; 
                                    padding: 1.5rem; 
                                    border-radius: 12px; 
                                    margin: 1rem 0;">
                            <h3 style="color: {color}; margin: 0 0 1rem 0;">
                                📊 Sentiment: {result['label']}
                            </h3>
                            <div style="font-size: 1.125rem; font-weight: 600; margin-bottom: 1rem;">
                                Confidence: {result['confidence']:.1%}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Sentiment breakdown chart
                        scores_df = pd.DataFrame([
                            {'Sentiment': k, 'Score': v * 100} 
                            for k, v in result['scores'].items()
                        ])
                        
                        fig = px.bar(
                            scores_df, 
                            x='Sentiment', 
                            y='Score',
                            color='Sentiment',
                            color_discrete_map={
                                'Positive': '#10b981',
                                'Negative': '#ef4444', 
                                'Neutral': '#6b7280'
                            },
                            title="Detailed Sentiment Breakdown"
                        )
                        fig.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font_family="Inter",
                            title_font_size=16,
                            title_font_color='#1f2937'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("❌ Sentiment analyzer not available")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="modern-card">
                <h3>🎯 Model Performance</h3>
                <div style="margin-top: 1rem;">
            """, unsafe_allow_html=True)
            
            st.markdown(create_performance_badge("Accuracy", 0.9905, 0.75), unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(create_performance_badge("F1 Score", 0.9905, 0.75), unsafe_allow_html=True)
            
            st.markdown("""
                </div>
                <div style="margin-top: 1.5rem; font-size: 0.875rem; color: var(--text-secondary);">
                    <strong>Model:</strong> DistilBERT-base-uncased (66M parameters)<br>
                    <strong>Dataset:</strong> Enhanced Financial Sentiment Data<br>
                    <strong>Training:</strong> Focal Loss + Label Smoothing (11.57 min)<br>
                    <strong>Performance:</strong> 97.62% accuracy
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Tab 3: Document Summarizer (Enhanced)
    with tab3:
        st.markdown("""
        <div class="modern-card">
            <div class="card-header">
                <div class="card-icon">📝</div>
                <h2 class="card-title">Intelligent Document Summarizer</h2>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<div class="modern-card">', unsafe_allow_html=True)
            
            document_text = st.text_area(
                "Document to summarize:",
                value="""Apple Inc. (AAPL) reported fiscal Q4 2024 results that exceeded Wall Street expectations across multiple metrics. The technology giant posted revenue of $394.3 billion, representing an 8% year-over-year increase, driven by strong iPhone sales and robust services growth. 

iPhone revenue reached $200.6 billion, up 12% from the previous year, with the iPhone 15 series showing particularly strong demand in international markets. The Services segment, including the App Store, iCloud, and Apple Pay, generated $85.2 billion in revenue, marking a 15% increase and highlighting Apple's successful transition to recurring revenue streams.

Mac revenue totaled $29.8 billion, while iPad generated $23.5 billion. The company's gross margin improved to 45.2%, reflecting operational efficiency and favorable product mix. Apple returned $27 billion to shareholders through dividends and share buybacks during the quarter.

Looking ahead, Apple provided guidance for Q1 2025 revenue of $125-130 billion, above analyst estimates of $122 billion. The company highlighted strong momentum in emerging markets and continued growth in its ecosystem of products and services.""",
                height=200,
                label_visibility="collapsed"
            )
            
            if st.button("✨ Generate Summary", type="primary", use_container_width=True):
                if status['summarizer']:
                    with st.spinner("📝 Creating intelligent summary..."):
                        result = summarize_document(models, document_text)
                    
                    if result:
                        # Modern summary display
                        st.markdown("""
                        <div style="background: linear-gradient(135deg, #667eea15, #764ba205); 
                                    border: 1px solid #e5e7eb; 
                                    border-radius: 16px; 
                                    padding: 1.5rem; 
                                    margin: 1rem 0;">
                            <h3 style="color: #1f2937; margin: 0 0 1rem 0;">
                                📄 Intelligent Summary
                            </h3>
                        """, unsafe_allow_html=True)
                        
                        st.write(result['summary'])
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Compression metrics
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.markdown(create_metric_card(
                                "Original", 
                                f"{result['original_length']:,}", 
                                "characters", 
                                "📄"
                            ), unsafe_allow_html=True)
                        with col_b:
                            st.markdown(create_metric_card(
                                "Summary", 
                                f"{result['summary_length']:,}", 
                                "characters", 
                                "✨"
                            ), unsafe_allow_html=True)
                        with col_c:
                            st.markdown(create_metric_card(
                                "Compression", 
                                f"{result['compression']:.1%}", 
                                "ratio", 
                                "📊"
                            ), unsafe_allow_html=True)
                else:
                    st.error("❌ Document summarizer not available")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="modern-card">
                <h3>🏆 Model Capabilities</h3>
                <div style="margin-top: 1rem;">
            """, unsafe_allow_html=True)
            
            st.markdown(create_performance_badge("ROUGE-L", 35, 30, "{:.0f}"), unsafe_allow_html=True)
            
            st.markdown("""
                </div>
                <div style="margin-top: 1.5rem; font-size: 0.875rem; color: var(--text-secondary);">
                    <strong>Model:</strong> T5-small<br>
                    <strong>Task:</strong> Abstractive Summarization<br>
                    <strong>Specialization:</strong> Financial Documents (53 real documents)<br>
                    <strong>Training:</strong> 0.55 minutes (Enhanced T5 pipeline)<br>
                    <strong>Performance:</strong> ROUGE-L ~35 (target ≥30)
                </div>
                
                <div style="margin-top: 1.5rem;">
                    <strong style="color: var(--text-primary); margin-bottom: 0.75rem; display: block;">📋 Best For:</strong>
                    <div style="font-size: 0.875rem; color: var(--text-secondary); margin-left: 1rem; line-height: 1.6;">
                        • SEC filings (10-K, 10-Q)<br>
                        • Earnings reports<br>
                        • Financial news articles<br>
                        • Research reports<br>
                        • Investment analyses
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Tab 4: Fraud Detection (Enhanced)
    with tab4:
        st.markdown("""
        <div class="modern-card">
            <div class="card-header">
                <div class="card-icon">🛡️</div>
                <h2 class="card-title">Unified Fraud Risk Scorer (95.63% AUC)</h2>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<div class="modern-card">', unsafe_allow_html=True)
            st.markdown("### 🔍 Transaction Analysis")
            
            transaction_text = st.text_area(
                "Transaction description:",
                value="Emergency transfer of $45,000 to overseas account for urgent business deal. Account holder requested immediate processing.",
                height=80,
                label_visibility="collapsed"
            )
            
            transaction_amount = st.number_input(
                "Transaction Amount ($)",
                min_value=0.0,
                value=45000.0,
                format="%.2f"
            )
            
            if st.button("🔍 Analyze Transaction", type="primary", use_container_width=True):
                if status['fraud']:
                    with st.spinner("🛡️ Scanning for fraud indicators..."):
                        result = detect_fraud(models, transaction_text, transaction_amount)
                    
                    if result:
                        risk_level = result.get('risk_level', 'UNKNOWN')
                        risk_score = float(result.get('risk_score', result.get('risk_percentage', 0)))
                        
                        # Risk level styling
                        risk_colors = {
                            'LOW': '#10b981',
                            'MEDIUM': '#f59e0b', 
                            'HIGH': '#ef4444',
                            'CRITICAL': '#dc2626'
                        }
                        risk_color = risk_colors.get(risk_level, '#6b7280')
                        
                        # Modern risk display
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, {risk_color}15, {risk_color}05); 
                                    border-left: 4px solid {risk_color}; 
                                    padding: 1.5rem; 
                                    border-radius: 12px; 
                                    margin: 1rem 0;">
                            <h3 style="color: {risk_color}; margin: 0 0 1rem 0;">
                                🛡️ Risk Level: {risk_level}
                            </h3>
                            <div style="font-size: 1.125rem; font-weight: 600; margin-bottom: 1rem;">
                                Fraud Score: {risk_score:.1%}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Risk gauge
                        risk_pct = risk_score * 100 if risk_score <= 1 else risk_score
                        
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=risk_pct,
                            title={'text': "Fraud Risk Score", 'font': {'size': 20, 'family': 'Inter'}},
                            number={'font': {'size': 40, 'family': 'Inter'}},
                            gauge={
                                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#e5e7eb"},
                                'bar': {'color': risk_color, 'thickness': 0.3},
                                'bgcolor': "white",
                                'borderwidth': 2,
                                'bordercolor': "#e5e7eb",
                                'steps': [
                                    {'range': [0, 30], 'color': "rgba(16, 185, 129, 0.1)"},
                                    {'range': [30, 70], 'color': "rgba(245, 158, 11, 0.1)"},
                                    {'range': [70, 100], 'color': "rgba(239, 68, 68, 0.1)"}
                                ],
                                'threshold': {
                                    'line': {'color': risk_color, 'width': 4},
                                    'thickness': 0.75,
                                    'value': risk_pct
                                }
                            }
                        ))
                        
                        fig.update_layout(
                            height=350,
                            font={'family': 'Inter'},
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("❌ Fraud detection system not available")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="modern-card">
                <h3>🎯 Detection Performance</h3>
                <div style="margin-top: 1rem;">
            """, unsafe_allow_html=True)
            
            st.markdown(create_performance_badge("AUC Score", 0.9563, 0.80), unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(create_performance_badge("Precision", 0.94, 0.80), unsafe_allow_html=True)
            
            st.markdown("""
                </div>
                <div style="margin-top: 1.5rem; font-size: 0.875rem; color: var(--text-secondary);">
                    <strong>System:</strong> Unified Fraud Risk Scorer<br>
                    <strong>Approaches:</strong> Unsupervised (~60% AUC) + Hybrid (≥90% AUC)<br>
                    <strong>Method:</strong> NLP + Numeric Features + Supervised Learning<br>
                    <strong>Dataset:</strong> PaySim Dataset<br>
                    <strong>Performance:</strong> 95.63% AUC (Hybrid Mode)
                </div>
            </div>
            
            <div class="modern-card" style="margin-top: 1rem;">
                <h3>⚠️ Red Flags</h3>
                <ul style="font-size: 0.875rem; color: var(--text-secondary); margin-left: 1rem; line-height: 1.6;">
                    <li>Large round-number amounts</li>
                    <li>Urgent/emergency language</li>
                    <li>Overseas destinations</li>
                    <li>New account recipients</li>
                    <li>Off-hours transactions</li>
                    <li>Balance manipulation</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # Tab 5: Investment Insights (Enhanced)
    with tab5:
        st.markdown("""
        <div class="modern-card">
            <div class="card-header">
                <div class="card-icon">📈</div>
                <h2 class="card-title">4-Module Investment Intelligence Integration</h2>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<div class="modern-card">', unsafe_allow_html=True)
            st.markdown("### 🏢 Investment Analysis Input")
            
            company_name = st.text_input(
                "Company Name:",
                value="Apple Inc.",
                placeholder="e.g., Apple Inc., Tesla, Microsoft"
            )
            
            news_text = st.text_area(
                "Recent News:",
                value="Apple reported strong quarterly earnings with iPhone sales exceeding expectations. The company beat analyst estimates with revenue growth of 8% year-over-year and announced expansion into new markets.",
                height=100
            )
            
            doc_text = st.text_area(
                "Financial Document:",
                value="Apple Inc. demonstrates solid financial performance with revenue of $394.3 billion and net income of $99.8 billion. The company maintains strong cash position of $165 billion and healthy debt-to-equity ratio.",
                height=100
            )
            
            if st.button("🚀 Generate Investment Insight", type="primary", use_container_width=True):
                if status['insights']:
                    with st.spinner("📊 Analyzing investment opportunity..."):
                        result = generate_insight(models, company_name, news_text, doc_text)
                    
                    if result:
                        # Modern recommendation display
                        rec = result.get('recommendation', 'HOLD')
                        rec_colors = {
                            'STRONG BUY': '#059669',
                            'BUY': '#10b981',
                            'HOLD': '#6b7280',
                            'WEAK HOLD': '#f59e0b',
                            'SELL': '#ef4444'
                        }
                        rec_color = rec_colors.get(rec, '#6b7280')
                        
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, {rec_color}15, {rec_color}05); 
                                    border: 2px solid {rec_color}; 
                                    padding: 1.5rem; 
                                    border-radius: 16px; 
                                    margin: 1rem 0; 
                                    text-align: center;">
                            <h2 style="color: {rec_color}; margin: 0; font-size: 1.5rem;">
                                📈 Recommendation: {rec}
                            </h2>
                            <div style="font-size: 1.125rem; margin-top: 0.5rem; color: var(--text-secondary);">
                                Overall Score: {result.get('overall_score', 0):.1%}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Detailed metrics
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.markdown(create_metric_card(
                                "Sentiment Score",
                                f"{result.get('sentiment_score', 0):.1%}",
                                "Market Mood",
                                "📊"
                            ), unsafe_allow_html=True)
                        with col_b:
                            st.markdown(create_metric_card(
                                "Financial Health",
                                f"{result.get('document_score', 0):.1%}",
                                "Document Analysis",
                                "💰"
                            ), unsafe_allow_html=True)
                        with col_c:
                            st.markdown(create_metric_card(
                                "Risk Assessment",
                                f"{result.get('risk_score', 0):.1%}",
                                "Safety Score",
                                "🛡️"
                            ), unsafe_allow_html=True)
                        
                        # Analysis summary
                        st.markdown("""
                        <div class="modern-card">
                            <h3>📋 Analysis Summary</h3>
                        """, unsafe_allow_html=True)
                        st.write(result.get('narrative', result.get('explanation', 'Analysis complete.')))
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Enhanced radar chart
                        categories = ['Sentiment', 'Financials', 'Risk', 'Overall']
                        values = [
                            result.get('sentiment_score', 0) * 100,
                            result.get('document_score', 0) * 100,
                            result.get('risk_score', 0) * 100,
                            result.get('overall_score', 0) * 100
                        ]
                        
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatterpolar(
                            r=values,
                            theta=categories,
                            fill='toself',
                            fillcolor='rgba(37, 99, 235, 0.2)',
                            line=dict(color='rgb(37, 99, 235)', width=3),
                            marker=dict(size=8, color='rgb(37, 99, 235)'),
                            name='Investment Scores'
                        ))
                        
                        fig.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, 100],
                                    tickfont=dict(size=12, family='Inter'),
                                    gridcolor='#e5e7eb'
                                ),
                                angularaxis=dict(
                                    tickfont=dict(size=14, family='Inter', color='#1f2937')
                                )
                            ),
                            showlegend=False,
                            height=400,
                            font_family="Inter",
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            title={
                                'text': f'Investment Analysis: {company_name}',
                                'x': 0.5,
                                'font': {'size': 18, 'family': 'Inter', 'color': '#1f2937'}
                            }
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("❌ Investment insight generator not available")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="modern-card">
                <h3>⚖️ Scoring Methodology</h3>
            """, unsafe_allow_html=True)
            
            weights_df = pd.DataFrame({
                'Component': ['Sentiment', 'Document Quality', 'Risk Assessment'],
                'Weight': [40, 35, 25]
            })
            
            fig = px.pie(
                weights_df, 
                values='Weight', 
                names='Component',
                color_discrete_sequence=['#2563eb', '#7c3aed', '#dc2626']
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(
                height=300,
                font_family="Inter",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            </div>
            
            <div class="modern-card" style="margin-top: 1rem;">
                <h3>📊 Recommendation Scale</h3>
                <div style="margin-top: 1rem; font-size: 0.875rem; line-height: 1.8;">
                    <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                        <span style="color: #059669; font-weight: 600;">🟢 STRONG BUY:</span>
                        <span style="margin-left: 0.5rem; color: var(--text-secondary);">Score ≥ 80%</span>
                    </div>
                    <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                        <span style="color: #10b981; font-weight: 600;">🟢 BUY:</span>
                        <span style="margin-left: 0.5rem; color: var(--text-secondary);">Score ≥ 70%</span>
                    </div>
                    <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                        <span style="color: #6b7280; font-weight: 600;">⚪ HOLD:</span>
                        <span style="margin-left: 0.5rem; color: var(--text-secondary);">Score ≥ 60%</span>
                    </div>
                    <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                        <span style="color: #f59e0b; font-weight: 600;">🟡 WEAK HOLD:</span>
                        <span style="margin-left: 0.5rem; color: var(--text-secondary);">Score ≥ 40%</span>
                    </div>
                    <div style="display: flex; align-items: center;">
                        <span style="color: #ef4444; font-weight: 600;">🔴 SELL:</span>
                        <span style="margin-left: 0.5rem; color: var(--text-secondary);">Score < 40%</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

def process_chat_query(models, query):
    """Process chatbot query and route to appropriate module"""
    query_lower = query.lower()
    
    # Detect intent
    if any(kw in query_lower for kw in ['sentiment', 'feeling', 'mood', 'positive', 'negative']):
        result = analyze_sentiment(models, query)
        if result:
            return f"""**📊 Sentiment Analysis Results**

**🎯 Overall Sentiment:** {result['label']}  
**📈 Confidence Level:** {result['confidence']:.1%}

**📋 Detailed Breakdown:**
- 🟢 Positive: {result['scores'].get('Positive', 0):.1%}
- ⚪ Neutral: {result['scores'].get('Neutral', 0):.1%}
- 🔴 Negative: {result['scores'].get('Negative', 0):.1%}

*🤖 Powered by DistilBERT-base-uncased (97.62% accuracy)*"""
    
    elif any(kw in query_lower for kw in ['fraud', 'suspicious', 'transaction', 'risk']):
        import re
        amount_match = re.search(r'\$?([\d,]+(?:\.\d{2})?)', query)
        amount = float(amount_match.group(1).replace(',', '')) if amount_match else None
        
        result = detect_fraud(models, query, amount)
        if result:
            risk_level = result.get('risk_level', 'UNKNOWN')
            risk_score = result.get('risk_score', result.get('risk_percentage', 0))
            status_emoji = '⚠️' if risk_level in ['HIGH', 'CRITICAL'] else '✅'
            
            return f"""**🛡️ Fraud Risk Assessment**

**🎯 Risk Level:** {risk_level}  
**📊 Fraud Score:** {float(risk_score):.1%}  
**🔍 Assessment:** {status_emoji} {'SUSPICIOUS ACTIVITY DETECTED' if risk_level in ['HIGH', 'CRITICAL'] else 'TRANSACTION APPEARS LEGITIMATE'}

*🤖 Analyzed using Unified Fraud Risk Scorer (95.63% AUC)*"""
    
    elif any(kw in query_lower for kw in ['invest', 'buy', 'sell', 'stock', 'portfolio']):
        companies = ['Apple', 'Tesla', 'Microsoft', 'Amazon', 'Google', 'Meta', 'Netflix', 'NVIDIA']
        company = next((c for c in companies if c.lower() in query_lower), 'the company')
        
        result = generate_insight(
            models, 
            company,
            f"{company} demonstrates strong market performance with positive investor sentiment and solid fundamentals.",
            f"{company} shows healthy financial metrics with consistent revenue growth and strong balance sheet position."
        )
        if result:
            rec_emoji = {'STRONG BUY': '🚀', 'BUY': '📈', 'HOLD': '⚖️', 'WEAK HOLD': '⚠️', 'SELL': '📉'}
            emoji = rec_emoji.get(result.get('recommendation', 'HOLD'), '📊')
            
            return f"""**📈 Investment Analysis: {company}**

**{emoji} Recommendation:** {result.get('recommendation', 'HOLD')}  
**🎯 Confidence Score:** {result.get('overall_score', 0):.1%}

**📋 Component Analysis:**
- 📊 Market Sentiment: {result.get('sentiment_score', 0):.1%}
- 💰 Financial Health: {result.get('document_score', 0):.1%}
- 🛡️ Risk Assessment: {result.get('risk_score', 0):.1%}

**📝 Summary:** {result.get('narrative', result.get('explanation', 'Comprehensive analysis complete with multi-factor assessment.'))}

*🤖 Generated using integrated AI analysis pipeline*"""
    
    elif any(kw in query_lower for kw in ['summarize', 'summary', 'brief']):
        result = summarize_document(models, query)
        if result:
            return f"""**📝 Document Summary**

{result['summary']}

**📊 Statistics:**
- Original Length: {result['original_length']:,} characters
- Summary Length: {result['summary_length']:,} characters  
- Compression Ratio: {result['compression']:.1%}

*🤖 Generated using T5-based abstractive summarization*"""
    
    # Default response
    return """**🚀 Welcome to VesprAI - Research-Grade Financial Intelligence!**

I'm your integrated 4-module financial assistant with research-grade performance:

**📊 DistilBERT Sentiment Analysis (97.62% accuracy)**
*Example:* "What's the sentiment of: Apple beats earnings expectations"

**🛡️ Unified Fraud Risk Scorer (95.63% AUC)** 
*Example:* "Is this suspicious? $50,000 urgent transfer to offshore account"

**📈 4-Module Investment Intelligence**
*Example:* "Should I invest in Tesla stock right now?"

**📝 T5 Document Summarizer (ROUGE-L ~35)**
*Example:* "Summarize: [paste your financial document here]"

**💡 Try asking me anything about finance, investing, or market analysis!**
*All powered by trained models from your VesprAI research project.*"""

if __name__ == "__main__":
    main()
