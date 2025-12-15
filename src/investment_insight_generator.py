"""
VesprAI Investment Insight Generator - Module 4 (FIXED)
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
    
    # Sentiment label mapping for trained DistilBERT model
    SENTIMENT_LABELS = {
        'LABEL_0': 'negative',
        'LABEL_1': 'neutral', 
        'LABEL_2': 'positive'
    }
    
    def __init__(self):
        self.sentiment_pipeline = None
        self.document_summarizer = None
        self.fraud_scorer = None
        
        # Track which modules are loaded
        self.modules_loaded = {
            'sentiment': False,
            'summarizer': False,
            'fraud': False
        }
        
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
    
    def load_modules(self, sentiment_model_path: str = None, fraud_model_path: str = None):
        """Load trained models from previous modules"""
        
        # Determine paths - check both project root and notebooks folder
        project_root = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
        
        # Default paths
        if sentiment_model_path is None:
            # Try multiple possible locations
            possible_sentiment_paths = [
                project_root / 'models' / 'final_model',
                Path('models') / 'final_model',
                Path('../models/final_model'),
            ]
            for p in possible_sentiment_paths:
                if p.exists():
                    sentiment_model_path = p
                    break
        
        if fraud_model_path is None:
            possible_fraud_paths = [
                Path('models') / 'best_fraud_scorer',  # Relative to notebooks/
                project_root / 'models' / 'best_fraud_scorer',
                Path('../models/best_fraud_scorer'),
            ]
            for p in possible_fraud_paths:
                if p.exists():
                    fraud_model_path = p
                    break
        
        # Load Module 1: Sentiment Analyzer (Trained DistilBERT)
        try:
            from transformers import pipeline
            
            if sentiment_model_path and Path(sentiment_model_path).exists():
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model=str(sentiment_model_path),
                    tokenizer=str(sentiment_model_path),
                    return_all_scores=True
                )
                self.modules_loaded['sentiment'] = True
                logger.info(f"✅ Loaded Module 1: Sentiment Analyzer from {sentiment_model_path}")
            else:
                logger.warning(f"⚠️ Sentiment model not found at {sentiment_model_path}")
        except Exception as e:
            logger.warning(f"⚠️ Could not load sentiment analyzer: {e}")
        
        # Load Module 2: Document Summarizer (T5-small pretrained)
        try:
            from document_summarizer import DocumentSummarizer
            self.document_summarizer = DocumentSummarizer(model_name="t5-small")
            self.modules_loaded['summarizer'] = True
            logger.info("✅ Loaded Module 2: Document Summarizer (T5-small)")
        except ImportError:
            try:
                # Try with src prefix
                import sys
                sys.path.append(str(project_root / 'src'))
                from document_summarizer import DocumentSummarizer
                self.document_summarizer = DocumentSummarizer(model_name="t5-small")
                self.modules_loaded['summarizer'] = True
                logger.info("✅ Loaded Module 2: Document Summarizer (T5-small)")
            except Exception as e:
                logger.warning(f"⚠️ Could not load document summarizer: {e}")
        
        # Load Module 3: Fraud Risk Scorer (Trained Hybrid Model)
        try:
            from unified_fraud_risk_scorer import UnifiedFraudRiskScorer
            self.fraud_scorer = UnifiedFraudRiskScorer(model_name="all-MiniLM-L6-v2")
            
            # Load trained weights if available
            if fraud_model_path and Path(fraud_model_path).exists():
                classifier_path = Path(fraud_model_path) / "best_classifier.joblib"
                if classifier_path.exists():
                    self.fraud_scorer.hybrid_classifier = joblib.load(classifier_path)
                    self.fraud_scorer.text_scaler = joblib.load(Path(fraud_model_path) / "best_text_scaler.joblib")
                    self.fraud_scorer.numeric_scaler = joblib.load(Path(fraud_model_path) / "best_numeric_scaler.joblib")
                    self.fraud_scorer.hybrid_trained = True
                    logger.info(f"✅ Loaded Module 3: Fraud Detector (Trained Hybrid, AUC ~0.95)")
                else:
                    logger.warning(f"⚠️ Fraud model weights not found, using base scorer")
            
            self.modules_loaded['fraud'] = True
        except ImportError:
            try:
                import sys
                sys.path.append(str(project_root / 'src'))
                from unified_fraud_risk_scorer import UnifiedFraudRiskScorer
                self.fraud_scorer = UnifiedFraudRiskScorer(model_name="all-MiniLM-L6-v2")
                self.modules_loaded['fraud'] = True
                logger.info("✅ Loaded Module 3: Fraud Risk Scorer (Base)")
            except Exception as e:
                logger.warning(f"⚠️ Could not load fraud scorer: {e}")
        
        # Summary
        loaded_count = sum(self.modules_loaded.values())
        logger.info(f"📊 Modules loaded: {loaded_count}/3")
    
    def analyze_sentiment(self, news_text: str) -> Dict[str, Any]:
        """Analyze sentiment from news text using trained DistilBERT"""
        try:
            if self.sentiment_pipeline and self.modules_loaded['sentiment']:
                # Use trained sentiment model
                result = self.sentiment_pipeline(news_text)
                
                # Parse results - pipeline returns list of list of dicts
                scores = {}
                for item in result[0]:
                    label = self.SENTIMENT_LABELS.get(item['label'], item['label'])
                    scores[label] = item['score']
                
                # Find best prediction
                best_label = max(scores, key=scores.get)
                best_score = scores[best_label]
                
                # Convert to numeric score (0-1 scale)
                # positive = high, negative = low
                sentiment_score = (
                    scores.get('positive', 0) * 1.0 +
                    scores.get('neutral', 0) * 0.5 +
                    scores.get('negative', 0) * 0.0
                )
                
                return {
                    'score': sentiment_score,
                    'label': best_label,
                    'confidence': best_score,
                    'all_scores': scores,
                    'text_sample': news_text[:100] + "..." if len(news_text) > 100 else news_text,
                    'model_used': 'trained_distilbert'
                }
            else:
                # Fallback: Simple keyword-based sentiment
                return self._fallback_sentiment(news_text)
            
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            return self._fallback_sentiment(news_text)
    
    def _fallback_sentiment(self, text: str) -> Dict[str, Any]:
        """Fallback sentiment analysis using keywords"""
        positive_words = ['good', 'great', 'excellent', 'strong', 'growth', 'profit', 'success', 'beat', 'exceed', 'positive', 'bullish', 'up', 'record', 'optimism']
        negative_words = ['bad', 'poor', 'weak', 'loss', 'decline', 'fall', 'negative', 'bearish', 'down', 'concern', 'risk', 'warning', 'miss', 'disappoint']
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            score = min(1.0, 0.6 + (pos_count - neg_count) * 0.05)
            label = 'positive'
        elif neg_count > pos_count:
            score = max(0.0, 0.4 - (neg_count - pos_count) * 0.05)
            label = 'negative'
        else:
            score = 0.5
            label = 'neutral'
        
        return {
            'score': score,
            'label': label,
            'confidence': abs(score - 0.5) * 2,  # Confidence based on distance from neutral
            'all_scores': {'positive': score if label == 'positive' else 0, 
                          'neutral': score if label == 'neutral' else 0,
                          'negative': score if label == 'negative' else 0},
            'text_sample': text[:100] + "..." if len(text) > 100 else text,
            'model_used': 'keyword_fallback'
        }
    
    def analyze_document(self, document_text: str) -> Dict[str, Any]:
        """Analyze document quality and extract insights"""
        try:
            if self.document_summarizer and self.modules_loaded['summarizer']:
                # Use T5 document summarizer
                summary = self.document_summarizer.summarize(document_text)
                
                # Calculate quality metrics based on content
                quality_score = self._calculate_document_quality(document_text, summary)
                
                return {
                    'summary': summary,
                    'quality_score': quality_score,
                    'key_insights': self._extract_key_insights(summary),
                    'document_length': len(document_text),
                    'model_used': 't5_summarizer'
                }
            else:
                # Fallback document analysis
                return self._fallback_document_analysis(document_text)
                
        except Exception as e:
            logger.warning(f"Document analysis failed: {e}")
            return self._fallback_document_analysis(document_text)
    
    def _calculate_document_quality(self, original: str, summary: str) -> float:
        """Calculate document quality score based on content analysis"""
        quality_indicators = {
            'positive': ['revenue growth', 'profit', 'earnings', 'strong', 'increase', 'success', 'beat', 'exceeded'],
            'negative': ['loss', 'decline', 'decrease', 'weak', 'concern', 'risk', 'warning', 'miss']
        }
        
        text_lower = original.lower()
        pos_count = sum(1 for phrase in quality_indicators['positive'] if phrase in text_lower)
        neg_count = sum(1 for phrase in quality_indicators['negative'] if phrase in text_lower)
        
        # Base score
        base_score = 0.5 + (pos_count - neg_count) * 0.08
        
        # Bonus for comprehensive documents
        if len(original) > 500:
            base_score += 0.1
        
        return max(0.0, min(1.0, base_score))
    
    def _fallback_document_analysis(self, text: str) -> Dict[str, Any]:
        """Fallback document analysis"""
        # Simple extractive summarization
        sentences = text.split('. ')[:3]
        summary = '. '.join(sentences) + '.'
        
        quality_score = self._calculate_document_quality(text, summary)
        key_insights = self._extract_key_insights(summary)
        
        return {
            'summary': summary,
            'quality_score': quality_score,
            'key_insights': key_insights,
            'document_length': len(text),
            'model_used': 'fallback'
        }
    
    def _extract_key_insights(self, text: str) -> List[str]:
        """Extract key insights from text"""
        insights = []
        text_lower = text.lower()
        
        # Financial metrics
        if any(word in text_lower for word in ['revenue', 'sales', 'income']):
            insights.append("Financial metrics reported")
        
        # Growth indicators
        if any(word in text_lower for word in ['growth', 'increase', 'up', 'higher']):
            insights.append("Positive growth trends")
        
        # Profitability
        if any(word in text_lower for word in ['profit', 'margin', 'earnings']):
            insights.append("Profitability metrics included")
        
        # Forward looking
        if any(word in text_lower for word in ['outlook', 'guidance', 'forecast', 'expect']):
            insights.append("Forward guidance provided")
        
        # Risk factors
        if any(word in text_lower for word in ['risk', 'challenge', 'concern', 'competition']):
            insights.append("Risk factors identified")
        
        return insights if insights else ["Standard financial disclosure"]
    
    def assess_risk(self, document_text: str) -> Dict[str, Any]:
        """Assess investment risk from document"""
        try:
            if self.fraud_scorer and self.modules_loaded['fraud']:
                # Use fraud scorer for risk assessment (inverse of fraud risk)
                result = self.fraud_scorer.score_transaction(
                    text=document_text,
                    method="unsupervised"
                )
                
                # Convert fraud risk to investment safety score (inverse)
                fraud_risk = result.get('risk_score', 0.5)
                safety_score = 1.0 - fraud_risk
                
                return {
                    'risk_score': safety_score,
                    'risk_level': 'LOW' if safety_score > 0.7 else 'MEDIUM' if safety_score > 0.4 else 'HIGH',
                    'risk_factors': self._identify_risk_factors(document_text),
                    'model_used': 'trained_fraud_scorer'
                }
            else:
                return self._fallback_risk_assessment(document_text)
                
        except Exception as e:
            logger.warning(f"Risk assessment failed: {e}")
            return self._fallback_risk_assessment(document_text)
    
    def _fallback_risk_assessment(self, text: str) -> Dict[str, Any]:
        """Fallback risk assessment"""
        risk_factors = self._identify_risk_factors(text)
        
        # Score based on number of risk factors
        risk_score = max(0.3, 0.8 - len(risk_factors) * 0.1)
        
        return {
            'risk_score': risk_score,
            'risk_level': 'LOW' if risk_score > 0.7 else 'MEDIUM' if risk_score > 0.4 else 'HIGH',
            'risk_factors': risk_factors,
            'model_used': 'fallback'
        }
    
    def _identify_risk_factors(self, text: str) -> List[str]:
        """Identify risk factors in text"""
        risks = []
        text_lower = text.lower()
        
        risk_mapping = {
            'competition': "Competitive pressure",
            'regulatory': "Regulatory challenges",
            'debt': "Debt concerns",
            'litigation': "Legal risks",
            'supply chain': "Supply chain risks",
            'market volatility': "Market volatility",
            'economic': "Economic uncertainty",
            'decline': "Performance decline",
            'loss': "Financial losses"
        }
        
        for keyword, risk_desc in risk_mapping.items():
            if keyword in text_lower:
                risks.append(risk_desc)
                
        return risks[:3] if risks else ["Standard business risks"]
    
    def calculate_overall_score(self, sentiment_result: Dict, document_result: Dict, risk_result: Dict) -> float:
        """Calculate weighted overall investment score"""
        sentiment_score = sentiment_result.get('score', 0.5)
        document_score = document_result.get('quality_score', 0.5)
        risk_score = risk_result.get('risk_score', 0.5)
        
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
            'confidence': overall_score,
            'narrative': insight_narrative,
            'sentiment_score': sentiment_result.get('score', 0),
            'document_score': document_result.get('quality_score', 0),
            'risk_score': risk_result.get('risk_score', 0),
            'explanation': insight_narrative,
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
        key_insights = '. '.join(document.get('key_insights', ['Analysis complete']))
        
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
        score = sentiment.get('score', 0.5)
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
        score = document.get('quality_score', 0.5)
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
        score = risk.get('risk_score', 0.5)
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
            'version': '2.0',  # Updated version
            'modules_status': self.modules_loaded
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
    sample_news = "Apple reports strong quarterly earnings with revenue growth of 15% year-over-year. The company beat analyst expectations."
    sample_document = "Apple Inc. continues to demonstrate solid financial performance with strong revenue growth across all segments. Services revenue reached record highs."
    
    result = generator.generate_insight("Apple Inc.", sample_news, sample_document)
    print(json.dumps(result, indent=2))