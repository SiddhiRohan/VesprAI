"""
VesprAI Investment Insight Generator - Module 4
Integrates sentiment analysis, document summarization, and fraud detection for investment insights
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import json
import joblib
import time
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InvestmentInsightGenerator:
    """
    VesprAI Investment Insight Generator
    Combines outputs from all modules to generate comprehensive investment insights
    """
    
    def __init__(self):
        self.sentiment_analyzer = None
        self.document_summarizer = None
        self.fraud_scorer = None
        
        # Insight scoring weights
        self.weights = {
            'sentiment': 0.40,     # 40% weight on market sentiment
            'document_quality': 0.35,  # 35% weight on document analysis
            'risk_assessment': 0.25    # 25% weight on risk factors
        }
        
        # Insight templates
        self.templates = self._load_templates()
        
        logger.info("Initialized Investment Insight Generator")
    
    def _load_templates(self) -> Dict[str, str]:
        """Load insight generation templates"""
        return {
            'excellent': "🚀 STRONG BUY SIGNAL: {company} shows exceptional potential with {sentiment_desc} market sentiment, {doc_desc} financial health, and {risk_desc} risk profile. {key_insights}",
            'good': "✅ POSITIVE OUTLOOK: {company} demonstrates solid fundamentals with {sentiment_desc} market reception, {doc_desc} business metrics, and {risk_desc} risk management. {key_insights}",
            'neutral': "⚖️ HOLD/NEUTRAL: {company} presents a balanced investment profile with {sentiment_desc} market sentiment, {doc_desc} financial position, and {risk_desc} risk factors. {key_insights}",
            'cautious': "⚠️ PROCEED WITH CAUTION: {company} shows concerning signals with {sentiment_desc} market sentiment, {doc_desc} financial indicators, and {risk_desc} risk elements. {key_insights}",
            'negative': "🔴 AVOID/SELL: {company} exhibits significant risks with {sentiment_desc} market reception, {doc_desc} financial health, and {risk_desc} risk profile. {key_insights}"
        }
    
    def load_modules(self, module_paths: Dict[str, str] = None):
        """Load trained models from previous modules"""
        try:
            # Import modules (adjust paths as needed)
            import sys
            from pathlib import Path
            
            # Add src to path if not already there
            src_path = Path.cwd() / "src"
            if src_path.exists():
                sys.path.append(str(src_path))
            
            # Load Module 1: Sentiment Analyzer
            try:
                from sentiment_analyzer import SentimentAnalyzer
                self.sentiment_analyzer = SentimentAnalyzer()
                logger.info("✅ Loaded Module 1: Sentiment Analyzer")
            except ImportError:
                logger.warning("⚠️ Could not import sentiment analyzer - will use fallback")
            
            # Load Module 2: Document Summarizer  
            try:
                from document_summarizer import DocumentSummarizer
                self.document_summarizer = DocumentSummarizer()
                logger.info("✅ Loaded Module 2: Document Summarizer")
            except ImportError:
                logger.warning("⚠️ Could not import document summarizer - will use fallback")
            
            # Load Module 3: Fraud Risk Scorer
            try:
                from unified_fraud_risk_scorer import UnifiedFraudRiskScorer
                self.fraud_scorer = UnifiedFraudRiskScorer()
                logger.info("✅ Loaded Module 3: Fraud Risk Scorer")
            except ImportError:
                logger.warning("⚠️ Could not import fraud scorer - will use fallback")
                
        except Exception as e:
            logger.error(f"Error loading modules: {e}")
            logger.info("Will use fallback scoring methods")
    
    def analyze_sentiment(self, news_text: str) -> Dict[str, Any]:
        """Analyze sentiment from news text"""
        try:
            if self.sentiment_analyzer:
                # Use real sentiment analyzer
                result = self.sentiment_analyzer.predict(news_text)
                if isinstance(result, dict):
                    sentiment_score = result.get('confidence', 0.5)
                    sentiment_label = result.get('label', 'neutral')
                else:
                    # Handle different return formats
                    sentiment_score = float(result) if hasattr(result, '__float__') else 0.5
                    sentiment_label = 'positive' if sentiment_score > 0.6 else 'negative' if sentiment_score < 0.4 else 'neutral'
            else:
                # Fallback: Simple keyword-based sentiment
                sentiment_score, sentiment_label = self._fallback_sentiment(news_text)
            
            return {
                'score': sentiment_score,
                'label': sentiment_label,
                'confidence': sentiment_score,
                'text_sample': news_text[:100] + "..." if len(news_text) > 100 else news_text
            }
            
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            return self._fallback_sentiment(news_text)
    
    def _fallback_sentiment(self, text: str) -> Dict[str, Any]:
        """Fallback sentiment analysis using keywords"""
        positive_words = ['good', 'great', 'excellent', 'strong', 'growth', 'profit', 'success', 'beat', 'exceed', 'positive', 'bullish', 'up']
        negative_words = ['bad', 'poor', 'weak', 'loss', 'decline', 'fall', 'negative', 'bearish', 'down', 'concern', 'risk', 'warning']
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            score = 0.6 + (pos_count - neg_count) * 0.1
            label = 'positive'
        elif neg_count > pos_count:
            score = 0.4 - (neg_count - pos_count) * 0.1
            label = 'negative'
        else:
            score = 0.5
            label = 'neutral'
        
        score = max(0.0, min(1.0, score))
        
        return {
            'score': score,
            'label': label,
            'confidence': score,
            'text_sample': text[:100] + "..." if len(text) > 100 else text
        }
    
    def analyze_document(self, document_text: str) -> Dict[str, Any]:
        """Analyze document quality and extract insights"""
        try:
            if self.document_summarizer:
                # Use real document summarizer
                summary = self.document_summarizer.summarize(document_text)
                
                # Calculate quality metrics
                quality_score = min(1.0, len(summary) / 100)  # Simple quality metric
                
                return {
                    'summary': summary,
                    'quality_score': quality_score,
                    'key_insights': self._extract_key_insights(summary),
                    'document_length': len(document_text)
                }
            else:
                # Fallback document analysis
                return self._fallback_document_analysis(document_text)
                
        except Exception as e:
            logger.warning(f"Document analysis failed: {e}")
            return self._fallback_document_analysis(document_text)
    
    def _fallback_document_analysis(self, text: str) -> Dict[str, Any]:
        """Fallback document analysis"""
        # Simple extractive summarization
        sentences = text.split('. ')[:3]  # Take first 3 sentences
        summary = '. '.join(sentences) + '.'
        
        # Simple quality scoring based on document characteristics
        quality_indicators = ['revenue', 'profit', 'earnings', 'growth', 'financial', 'performance']
        quality_score = min(1.0, sum(1 for word in quality_indicators if word.lower() in text.lower()) / 10)
        
        key_insights = self._extract_key_insights(summary)
        
        return {
            'summary': summary,
            'quality_score': quality_score,
            'key_insights': key_insights,
            'document_length': len(text)
        }
    
    def _extract_key_insights(self, text: str) -> List[str]:
        """Extract key insights from text"""
        insights = []
        text_lower = text.lower()
        
        # Financial indicators
        if 'revenue' in text_lower or 'sales' in text_lower:
            insights.append("Revenue/sales performance highlighted")
        if 'profit' in text_lower or 'earnings' in text_lower:
            insights.append("Profitability metrics discussed")
        if 'growth' in text_lower:
            insights.append("Growth trends identified")
        if 'risk' in text_lower or 'challenge' in text_lower:
            insights.append("Risk factors mentioned")
        if 'market' in text_lower:
            insights.append("Market conditions analyzed")
            
        return insights[:3] if insights else ["General business information provided"]
    
    def assess_risk(self, document_text: str) -> Dict[str, Any]:
        """Assess risk factors from document"""
        try:
            if self.fraud_scorer:
                # Use real fraud risk scorer for risk assessment
                risk_result = self.fraud_scorer.score_transaction(document_text, method="unsupervised")
                
                return {
                    'risk_score': 1.0 - risk_result.get('risk_percentage', 50) / 100,  # Invert for investment risk
                    'risk_level': 'low' if risk_result.get('risk_level') == 'LOW' else 'medium',
                    'risk_factors': self._identify_risk_factors(document_text)
                }
            else:
                # Fallback risk assessment
                return self._fallback_risk_assessment(document_text)
                
        except Exception as e:
            logger.warning(f"Risk assessment failed: {e}")
            return self._fallback_risk_assessment(document_text)
    
    def _fallback_risk_assessment(self, text: str) -> Dict[str, Any]:
        """Fallback risk assessment"""
        risk_indicators = ['debt', 'lawsuit', 'investigation', 'decline', 'loss', 'warning', 'concern', 'volatility']
        positive_indicators = ['stable', 'growth', 'strong', 'solid', 'healthy']
        
        text_lower = text.lower()
        risk_count = sum(1 for word in risk_indicators if word in text_lower)
        positive_count = sum(1 for word in positive_indicators if word in text_lower)
        
        # Calculate risk score (higher = less risky)
        risk_score = max(0.2, min(1.0, 0.7 - (risk_count * 0.1) + (positive_count * 0.05)))
        
        risk_level = 'low' if risk_score > 0.7 else 'medium' if risk_score > 0.4 else 'high'
        
        return {
            'risk_score': risk_score,
            'risk_level': risk_level,
            'risk_factors': self._identify_risk_factors(text)
        }
    
    def _identify_risk_factors(self, text: str) -> List[str]:
        """Identify specific risk factors"""
        risks = []
        text_lower = text.lower()
        
        risk_mapping = {
            'debt': "High debt levels",
            'lawsuit': "Legal proceedings",
            'competition': "Competitive pressures",
            'regulation': "Regulatory risks",
            'market volatility': "Market volatility",
            'economic': "Economic uncertainty"
        }
        
        for keyword, risk_desc in risk_mapping.items():
            if keyword in text_lower:
                risks.append(risk_desc)
                
        return risks[:3] if risks else ["Standard business risks"]
    
    def calculate_overall_score(self, sentiment_result: Dict, document_result: Dict, risk_result: Dict) -> float:
        """Calculate weighted overall investment score"""
        sentiment_score = sentiment_result['score']
        document_score = document_result['quality_score']
        risk_score = risk_result['risk_score']
        
        overall_score = (
            sentiment_score * self.weights['sentiment'] +
            document_score * self.weights['document_quality'] +
            risk_score * self.weights['risk_assessment']
        )
        
        return round(overall_score, 3)
    
    def generate_insight(self, company_name: str, news_text: str, document_text: str) -> Dict[str, Any]:
        """Generate comprehensive investment insight"""
        logger.info(f"Generating investment insight for {company_name}")
        
        start_time = time.time()
        
        # Analyze all components
        sentiment_result = self.analyze_sentiment(news_text)
        document_result = self.analyze_document(document_text)
        risk_result = self.assess_risk(document_text)
        
        # Calculate overall score
        overall_score = self.calculate_overall_score(sentiment_result, document_result, risk_result)
        
        # Generate narrative insight
        insight_narrative = self._generate_narrative(
            company_name, overall_score, sentiment_result, document_result, risk_result
        )
        
        processing_time = time.time() - start_time
        
        return {
            'company': company_name,
            'overall_score': overall_score,
            'recommendation': self._get_recommendation(overall_score),
            'narrative': insight_narrative,
            'components': {
                'sentiment': sentiment_result,
                'document': document_result,
                'risk': risk_result
            },
            'processing_time': round(processing_time, 2),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def _get_recommendation(self, score: float) -> str:
        """Get investment recommendation based on score"""
        if score >= 0.8:
            return "STRONG BUY"
        elif score >= 0.7:
            return "BUY"
        elif score >= 0.6:
            return "HOLD"
        elif score >= 0.4:
            return "WEAK HOLD"
        else:
            return "SELL"
    
    def _generate_narrative(self, company: str, score: float, sentiment: Dict, document: Dict, risk: Dict) -> str:
        """Generate natural language investment insight"""
        # Determine template based on score
        if score >= 0.8:
            template_key = 'excellent'
        elif score >= 0.65:
            template_key = 'good'
        elif score >= 0.5:
            template_key = 'neutral'
        elif score >= 0.35:
            template_key = 'cautious'
        else:
            template_key = 'negative'
        
        # Generate descriptions
        sentiment_desc = self._describe_sentiment(sentiment)
        doc_desc = self._describe_document(document)
        risk_desc = self._describe_risk(risk)
        key_insights = '. '.join(document['key_insights'])
        
        template = self.templates[template_key]
        
        return template.format(
            company=company,
            sentiment_desc=sentiment_desc,
            doc_desc=doc_desc,
            risk_desc=risk_desc,
            key_insights=key_insights
        )
    
    def _describe_sentiment(self, sentiment: Dict) -> str:
        """Describe sentiment in narrative form"""
        score = sentiment['score']
        if score >= 0.7:
            return "strongly positive"
        elif score >= 0.6:
            return "positive"
        elif score >= 0.4:
            return "neutral"
        else:
            return "negative"
    
    def _describe_document(self, document: Dict) -> str:
        """Describe document analysis in narrative form"""
        score = document['quality_score']
        if score >= 0.7:
            return "excellent"
        elif score >= 0.5:
            return "solid"
        elif score >= 0.3:
            return "adequate"
        else:
            return "concerning"
    
    def _describe_risk(self, risk: Dict) -> str:
        """Describe risk assessment in narrative form"""
        score = risk['risk_score']
        if score >= 0.7:
            return "low-risk"
        elif score >= 0.5:
            return "moderate-risk"
        else:
            return "high-risk"
    
    def batch_analyze(self, companies_data: Dict[str, Dict[str, str]]) -> Dict[str, Dict]:
        """Analyze multiple companies at once"""
        results = {}
        
        for company_name, data in companies_data.items():
            try:
                news_text = data.get('news', '')
                document_text = data.get('document', '')
                
                result = self.generate_insight(company_name, news_text, document_text)
                results[company_name] = result
                
            except Exception as e:
                logger.error(f"Error analyzing {company_name}: {e}")
                results[company_name] = {'error': str(e)}
        
        return results
    
    def save_model(self, model_dir: str = "models/investment_insight_generator"):
        """Save the insight generator configuration"""
        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config = {
            'weights': self.weights,
            'templates': self.templates,
            'model_type': 'investment_insight_generator',
            'version': '1.0'
        }
        
        with open(model_path / "insight_config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Investment Insight Generator configuration saved to {model_path}")
        return str(model_path)
    
    def load_model(self, model_dir: str):
        """Load saved configuration"""
        model_path = Path(model_dir)
        
        with open(model_path / "insight_config.json", "r") as f:
            config = json.load(f)
        
        self.weights = config['weights']
        self.templates = config['templates']
        
        logger.info(f"Configuration loaded from {model_path}")

if __name__ == "__main__":
    # Quick test
    generator = InvestmentInsightGenerator()
    generator.load_modules()
    
    # Test with sample data
    sample_news = "Apple reports strong quarterly earnings with revenue growth of 15% year-over-year."
    sample_document = "Apple Inc. continues to demonstrate solid financial performance with strong revenue growth across all segments."
    
    result = generator.generate_insight("Apple Inc.", sample_news, sample_document)
    print(json.dumps(result, indent=2))
