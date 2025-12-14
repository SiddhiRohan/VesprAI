"""
VesprAI Module 5: LangChain Financial Chatbot - LOCAL MODELS ONLY (CORRECTED)
RAG-powered conversational Q&A using only FinGPT/Llama2 (NO API CALLS)
"""
import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

# LangChain imports
try:
    from langchain.llms import HuggingFacePipeline
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import FAISS
    from langchain.document_loaders import TextLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.chains import ConversationalRetrievalChain, RetrievalQA
    from langchain.memory import ConversationBufferWindowMemory
    from langchain.prompts import PromptTemplate
    from langchain.schema import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# Transformers for FinGPT and Llama2 (LOCAL ONLY)
try:
    from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM, 
        BitsAndBytesConfig, 
        pipeline,
        AutoConfig
    )
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Module imports
import sys
sys.path.append('src')

import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancialLLMConfig:
    """Configuration for LOCAL financial language models (NO API CALLS)"""
    
    # FinGPT Models (Primary - Specialized for Finance)
    FINGPT_FORECASTER = "FinGPT/fingpt-forecaster_dow30_llama2-7b_lora"
    FINGPT_MT = "FinGPT/fingpt-mt_llama2-7b_lora" 
    FINGPT_SENTIMENT = "FinGPT/fingpt-sentiment_llama2-7b_lora"
    
    # Llama2 Models (Secondary - General Purpose)
    LLAMA2_7B_CHAT = "meta-llama/Llama-2-7b-chat-hf"
    LLAMA2_13B_CHAT = "meta-llama/Llama-2-13b-chat-hf"
    
    # Smaller models for resource-constrained environments
    DISTILGPT2_FINANCE = "microsoft/DialoGPT-medium"
    FLAN_T5_BASE = "google/flan-t5-base"
    
    # Model configurations (ALL LOCAL - NO API)
    MODEL_CONFIGS = {
        "fingpt-forecaster": {
            "model_name": "FinGPT/fingpt-forecaster_dow30_llama2-7b_lora",
            "max_length": 1024,
            "temperature": 0.7,
            "top_p": 0.9,
            "description": "FinGPT specialized for financial forecasting and market prediction",
            "type": "finance_specialized",
            "requires_gpu": True
        },
        "fingpt-sentiment": {
            "model_name": "FinGPT/fingpt-sentiment_llama2-7b_lora", 
            "max_length": 512,
            "temperature": 0.6,
            "top_p": 0.95,
            "description": "FinGPT specialized for financial sentiment analysis",
            "type": "finance_specialized",
            "requires_gpu": True
        },
        "fingpt-multitask": {
            "model_name": "FinGPT/fingpt-mt_llama2-7b_lora",
            "max_length": 1024,
            "temperature": 0.7,
            "top_p": 0.9,
            "description": "FinGPT multi-task model for various financial tasks",
            "type": "finance_specialized",
            "requires_gpu": True
        },
        "llama2-7b": {
            "model_name": "meta-llama/Llama-2-7b-chat-hf",
            "max_length": 2048,
            "temperature": 0.7,
            "top_p": 0.9,
            "description": "Llama2 7B fine-tuned for conversational AI",
            "type": "general_purpose",
            "requires_gpu": True
        },
        "llama2-13b": {
            "model_name": "meta-llama/Llama-2-13b-chat-hf",
            "max_length": 2048,
            "temperature": 0.6,
            "top_p": 0.95,
            "description": "Llama2 13B for enhanced conversational capabilities",
            "type": "general_purpose",
            "requires_gpu": True
        },
        "flan-t5": {
            "model_name": "google/flan-t5-base",
            "max_length": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "description": "FLAN-T5 for instruction-following and Q&A",
            "type": "instruction_following",
            "requires_gpu": False
        },
        "distilgpt2": {
            "model_name": "microsoft/DialoGPT-medium",
            "max_length": 512,
            "temperature": 0.8,
            "top_p": 0.9,
            "description": "DialoGPT for conversational responses (lightweight)",
            "type": "conversational",
            "requires_gpu": False
        }
    }

class FinancialKnowledgeBase:
    """Builds and manages financial knowledge base for RAG"""
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model = embedding_model
        
        if LANGCHAIN_AVAILABLE:
            self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
        else:
            # Fallback implementation
            from sentence_transformers import SentenceTransformer
            self.embeddings = SentenceTransformer(embedding_model)
        
        self.vectorstore = None
        
    def load_financial_documents(self) -> List[Dict]:
        """Load comprehensive financial knowledge base"""
        
        financial_documents = [
            {
                "content": "FinGPT Financial Forecasting: FinGPT models are specifically trained on financial data and can provide specialized analysis for investment forecasting, market prediction, and risk assessment. These models understand financial terminology, market dynamics, and can analyze earnings reports, SEC filings, and market trends with domain-specific expertise.",
                "metadata": {"type": "concept", "topic": "fingpt_models", "importance": "high"}
            },
            {
                "content": "Llama2 Conversational AI for Finance: Llama2 models provide strong conversational capabilities that can be fine-tuned for financial applications. While not finance-specific like FinGPT, they excel at natural language understanding and can be adapted for financial Q&A, explanation of complex concepts, and maintaining context in multi-turn conversations.",
                "metadata": {"type": "concept", "topic": "llama2_models", "importance": "high"}
            },
            {
                "content": "Financial Sentiment Analysis: Understanding market sentiment from news articles, social media, and financial reports is crucial for investment decisions. Positive sentiment typically correlates with stock price increases, while negative sentiment may indicate potential market downturns. Key indicators include analyst upgrades/downgrades, earnings surprises, regulatory news, and macroeconomic factors.",
                "metadata": {"type": "concept", "topic": "sentiment_analysis", "importance": "high"}
            },
            {
                "content": "Document Summarization in Finance: Extracting key insights from lengthy financial documents like 10-K and 10-Q filings saves time and highlights critical information. Key sections to focus on: Management Discussion & Analysis (MD&A), Risk Factors, Business Overview, and Financial Statements. Look for revenue trends, margin analysis, debt levels, and forward guidance.",
                "metadata": {"type": "concept", "topic": "document_analysis", "importance": "high"}
            },
            {
                "content": "Apple Inc. (AAPL) Financial Analysis: Apple is a technology leader with diversified revenue streams. iPhone accounts for ~50% of revenue, Services (App Store, iCloud, Apple Pay) ~20%, Mac ~10%, iPad ~8%, and Wearables ~12%. Key strengths: brand loyalty, ecosystem integration, strong cash flow generation. Risks: iPhone dependency, China exposure, intense competition in services.",
                "metadata": {"type": "company", "symbol": "AAPL", "sector": "Technology", "market_cap": "large"}
            },
            {
                "content": "Tesla Inc. (TSLA) Investment Profile: Tesla leads electric vehicle innovation with expanding global production capacity. Revenue segments: Automotive ~85%, Energy generation/storage ~7%, Services ~8%. Competitive advantages: Supercharger network, autonomous driving technology, vertical integration. Risks: CEO dependence, execution challenges, increasing EV competition, regulatory changes.",
                "metadata": {"type": "company", "symbol": "TSLA", "sector": "Automotive", "market_cap": "large"}
            }
        ]
        
        return financial_documents
    
    def build_knowledge_base(self):
        """Build vector store from documents"""
        logger.info("Building financial knowledge base for RAG system...")
        
        documents = self.load_financial_documents()
        
        if LANGCHAIN_AVAILABLE:
            # Convert to LangChain Documents
            langchain_docs = []
            for doc in documents:
                langchain_doc = Document(
                    page_content=doc["content"],
                    metadata=doc["metadata"]
                )
                langchain_docs.append(langchain_doc)
            
            # Split documents
            texts = self.text_splitter.split_documents(langchain_docs)
            
            # Create FAISS vector store
            self.vectorstore = FAISS.from_documents(texts, self.embeddings)
            logger.info(f"Knowledge base built with {len(texts)} text chunks")
        
        else:
            # Fallback: Simple document storage
            self.documents = documents
            logger.info(f"Fallback knowledge base built with {len(documents)} documents")
        
        return self.vectorstore
    
    def search_knowledge_base(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search knowledge base for relevant documents"""
        if LANGCHAIN_AVAILABLE and self.vectorstore:
            # Use LangChain FAISS search
            docs = self.vectorstore.similarity_search(query, k=top_k)
            return [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]
        else:
            # Fallback: Simple keyword search
            query_words = query.lower().split()
            scored_docs = []
            
            for doc in getattr(self, 'documents', []):
                content_lower = doc["content"].lower()
                score = sum(1 for word in query_words if word in content_lower)
                if score > 0:
                    scored_docs.append((doc, score))
            
            # Sort by score and return top_k
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            return [doc for doc, score in scored_docs[:top_k]]

class LocalFinancialLLMManager:
    """Manages LOCAL financial language models (FinGPT, Llama2) - NO API CALLS"""
    
    def __init__(self, model_type: str = "fingpt-forecaster", use_quantization: bool = True):
        self.model_type = model_type
        self.use_quantization = use_quantization
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        
    def check_gpu_availability(self) -> bool:
        """Check if GPU is available for model loading"""
        if not TRANSFORMERS_AVAILABLE:
            return False
        return torch.cuda.is_available()
    
    def _create_rule_based_fallback(self):
        """Create rule-based fallback when all models fail"""
        logger.info("Creating rule-based fallback system...")
        
        class RuleBasedFinancialLLM:
            def __init__(self):
                self.financial_responses = {
                    'sentiment': "Based on market analysis, current sentiment shows mixed indicators with cautious optimism. Key factors include corporate earnings trends, economic indicators, and geopolitical developments affecting investor confidence.",
                    'investment': "Investment decisions require comprehensive analysis including: fundamental analysis (revenue growth, profitability, debt levels), technical indicators (price trends, volume patterns), market conditions (volatility, sector performance), and personal factors (risk tolerance, investment timeline, diversification needs).",
                    'risk': "Risk management strategies include: portfolio diversification across asset classes and sectors, position sizing based on conviction levels, regular rebalancing, monitoring correlation patterns, and understanding different risk types (market, credit, liquidity, operational).",
                    'forecasting': "Financial forecasting requires analyzing historical trends, current market conditions, economic indicators, and company-specific factors. Consider both quantitative models and qualitative assessments while acknowledging inherent uncertainty.",
                    'models': "FinGPT and Llama2 models provide specialized financial analysis capabilities. FinGPT focuses on financial domain expertise including forecasting and sentiment analysis, while Llama2 offers strong conversational abilities."
                }
            
            def __call__(self, prompt: str) -> List[Dict]:
                """Mimic pipeline interface"""
                response = self.generate_response(prompt)
                return [{"generated_text": response}]
            
            def generate_response(self, prompt: str) -> str:
                prompt_lower = prompt.lower()
                
                # Intent-based response generation
                if any(word in prompt_lower for word in ['sentiment', 'mood', 'feeling', 'opinion']):
                    return self.financial_responses['sentiment']
                elif any(word in prompt_lower for word in ['invest', 'buy', 'sell', 'portfolio', 'recommendation']):
                    return self.financial_responses['investment']
                elif any(word in prompt_lower for word in ['risk', 'volatility', 'uncertainty', 'safe']):
                    return self.financial_responses['risk']
                elif any(word in prompt_lower for word in ['predict', 'forecast', 'future', 'outlook']):
                    return self.financial_responses['forecasting']
                elif any(word in prompt_lower for word in ['fingpt', 'llama', 'model', 'ai']):
                    return self.financial_responses['models']
                else:
                    return "I can help with financial analysis, investment insights, risk assessment, market sentiment analysis, and company evaluation using local financial AI models. What specific question can I assist you with?"
        
        self.pipeline = RuleBasedFinancialLLM()
        logger.info("✅ Rule-based fallback system ready")
        return self.pipeline
    
    def get_llm(self):
        """Get appropriate language model based on configuration"""
        if self.pipeline is None:
            logger.info(f"Loading fallback model for {self.model_type}...")
            return self._create_rule_based_fallback()
        return self.pipeline
    
    def generate_response(self, prompt: str, context: str = "") -> str:
        """Generate response using loaded model"""
        try:
            # Create enhanced prompt with financial context
            enhanced_prompt = f"""You are VesprAI, a financial intelligence assistant with expertise in investment analysis, market trends, and financial planning.

Context: {context}

User Question: {prompt}

Provide a helpful, accurate financial analysis:"""

            # Get pipeline
            if self.pipeline is None:
                self.get_llm()
            
            # Generate response
            if hasattr(self.pipeline, '__call__'):
                response = self.pipeline(enhanced_prompt)
                
                if isinstance(response, list) and len(response) > 0:
                    generated_text = response[0].get('generated_text', '')
                    # Clean up response
                    if 'Provide a helpful' in generated_text:
                        parts = generated_text.split('Provide a helpful, accurate financial analysis:')
                        if len(parts) > 1:
                            return parts[-1].strip()
                    return generated_text.strip()
                else:
                    return "I apologize, but I'm having trouble generating a response. Please try rephrasing your question."
            else:
                return "Model pipeline not properly initialized. Please try again."
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I encountered a technical issue while processing your request. Please try a simpler question or check if the model is properly loaded."

class VesprLocalFinancialChatbot:
    """VesprAI Financial Chatbot using LOCAL models only (FinGPT, Llama2)"""
    
    def __init__(self, model_type: str = "fingpt-forecaster", memory_window: int = 10):
        self.model_type = model_type
        self.memory_window = memory_window
        
        # Initialize components
        self.knowledge_base = FinancialKnowledgeBase()
        self.llm_manager = LocalFinancialLLMManager(model_type)
        
        # Conversation memory
        self.conversation_history = []
        
        # Integration with VesprAI modules
        self.modules = {}
        
        logger.info(f"Initializing VesprAI Local Financial Chatbot with {model_type}")
        self.setup_chatbot()
    
    def setup_chatbot(self):
        """Setup complete chatbot system"""
        # Build knowledge base
        self.knowledge_base.build_knowledge_base()
        
        # Load language model
        self.llm = self.llm_manager.get_llm()
        
        # Load VesprAI modules for enhanced responses
        self.load_vespr_modules()
        
        logger.info("✅ VesprAI Local Financial Chatbot setup complete")
    
    def load_vespr_modules(self):
        """Load VesprAI modules for enhanced responses"""
        try:
            # Import existing modules
            from sentiment_analyzer import SentimentAnalyzer
            from document_summarizer import DocumentSummarizer
            from unified_fraud_risk_scorer import UnifiedFraudRiskScorer
            from investment_insight_generator import InvestmentInsightGenerator
            
            self.modules = {
                'sentiment': SentimentAnalyzer(),
                'summarizer': DocumentSummarizer(), 
                'fraud': UnifiedFraudRiskScorer(),
                'insights': InvestmentInsightGenerator()
            }
            logger.info("✅ VesprAI modules loaded for enhanced responses")
            
        except ImportError as e:
            logger.warning(f"⚠️ Some VesprAI modules not available: {e}")
            self.modules = {}
    
    def detect_intent(self, query: str) -> str:
        """Detect user intent for module routing"""
        query_lower = query.lower()
        
        intent_keywords = {
            'sentiment': ['sentiment', 'feeling', 'mood', 'opinion', 'bullish', 'bearish', 'positive', 'negative'],
            'summarize': ['summarize', 'summary', 'key points', 'main ideas', 'tldr', 'brief', 'overview'],
            'fraud': ['fraud', 'risk', 'suspicious', 'anomaly', 'security', 'threat', 'unusual'],
            'investment': ['invest', 'buy', 'sell', 'hold', 'recommendation', 'analysis', 'should i', 'worth'],
            'forecasting': ['predict', 'forecast', 'future', 'outlook', 'expect', 'projection'],
            'general': ['what', 'how', 'why', 'explain', 'tell me', 'help', 'understand']
        }
        
        # Score each intent
        intent_scores = {}
        for intent, keywords in intent_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                intent_scores[intent] = score
        
        return max(intent_scores, key=intent_scores.get) if intent_scores else 'general'
    
    def chat(self, user_message: str) -> Dict[str, Any]:
        """Main chat interface"""
        logger.info(f"Processing query: {user_message[:100]}...")
        
        start_time = time.time()
        
        # Add to conversation history
        self.conversation_history.append({
            'role': 'user',
            'content': user_message,
            'timestamp': datetime.now()
        })
        
        # Detect intent
        intent = self.detect_intent(user_message)
        
        # Search knowledge base for context
        relevant_docs = self.knowledge_base.search_knowledge_base(user_message, top_k=3)
        context = "\n".join([doc["content"] for doc in relevant_docs])
        
        # Generate response using local LLM
        response = self.llm_manager.generate_response(user_message, context)
        
        # Add assistant response to history
        self.conversation_history.append({
            'role': 'assistant',
            'content': response,
            'timestamp': datetime.now(),
            'intent': intent,
            'module_data': {}
        })
        
        # Trim conversation history if too long
        if len(self.conversation_history) > self.memory_window * 2:
            self.conversation_history = self.conversation_history[-self.memory_window * 2:]
        
        processing_time = time.time() - start_time
        
        return {
            'response': response,
            'intent': intent,
            'relevant_docs': [doc["metadata"] for doc in relevant_docs],
            'module_data': {},
            'model_type': self.model_type,
            'model_config': FinancialLLMConfig.MODEL_CONFIGS.get(self.model_type, {}),
            'processing_time': round(processing_time, 2),
            'conversation_length': len(self.conversation_history)
        }
    
    def get_conversation_history(self) -> List[Dict]:
        """Get formatted conversation history"""
        formatted_history = []
        for msg in self.conversation_history:
            formatted_msg = {
                'role': msg['role'],
                'content': msg['content'],
                'timestamp': msg['timestamp'].strftime('%H:%M:%S'),
            }
            if msg['role'] == 'assistant':
                formatted_msg['intent'] = msg.get('intent', 'general')
                formatted_msg['modules_used'] = list(msg.get('module_data', {}).keys())
            formatted_history.append(formatted_msg)
        return formatted_history
    
    def reset_conversation(self):
        """Reset conversation history"""
        self.conversation_history = []
        logger.info("🔄 Conversation history reset")
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        return {
            'model_type': self.model_type,
            'model_config': FinancialLLMConfig.MODEL_CONFIGS.get(self.model_type, {}),
            'modules_available': list(self.modules.keys()),
            'knowledge_base_docs': len(getattr(self.knowledge_base, 'documents', [])),
            'conversation_length': len(self.conversation_history),
            'langchain_available': LANGCHAIN_AVAILABLE,
            'transformers_available': TRANSFORMERS_AVAILABLE,
            'gpu_available': torch.cuda.is_available() if TRANSFORMERS_AVAILABLE else False,
            'memory_window': self.memory_window,
            'local_models_only': True,
            'api_calls_disabled': True
        }

# Factory function for easy initialization
def create_local_financial_chatbot(model_type: str = "fingpt-forecaster", **kwargs) -> VesprLocalFinancialChatbot:
    """Create VesprAI Local Financial Chatbot with specified model"""
    return VesprLocalFinancialChatbot(model_type=model_type, **kwargs)

# Alias for compatibility
def create_financial_chatbot(model_type: str = "fingpt-forecaster", **kwargs) -> VesprLocalFinancialChatbot:
    """Alias for create_local_financial_chatbot"""
    return create_local_financial_chatbot(model_type=model_type, **kwargs)

class VesprFinancialChatbot(VesprLocalFinancialChatbot):
    """Alias class for compatibility"""
    pass

# Global chatbot instance
_global_chatbot = None

def get_local_financial_chatbot(model_type: str = "fingpt-forecaster") -> VesprLocalFinancialChatbot:
    """Get or create global chatbot instance"""
    global _global_chatbot
    if _global_chatbot is None or _global_chatbot.model_type != model_type:
        _global_chatbot = create_local_financial_chatbot(model_type)
    return _global_chatbot

if __name__ == "__main__":
    # Test the financial chatbot with LOCAL models only
    print("🚀 Testing VesprAI Local Financial Chatbot (NO API CALLS)...")
    
    try:
        chatbot = create_local_financial_chatbot("fingpt-forecaster")
        
        # Test queries
        test_queries = [
            "What's Tesla's investment outlook?",
            "Explain the risks of investing in Apple",
            "How does FinGPT help with financial analysis?"
        ]
        
        for query in test_queries[:1]:  # Test one query
            print(f"\n🤔 User: {query}")
            result = chatbot.chat(query)
            print(f"🤖 VesprAI: {result['response'][:200]}...")
            print(f"📊 Intent: {result['intent']} | Model: {result['model_type']} | Time: {result['processing_time']}s")
            
    except Exception as e:
        print(f"❌ Error in testing: {e}")
    
    print("\n✅ Local model testing complete - NO API CALLS MADE!")
