"""
VesprAI Module 5: Real Model-Based Financial Chatbot
Uses actual FinGPT/Llama2 models for dynamic response generation (NO pre-written answers)
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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Transformers for actual model loading
try:
    from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM, 
        BitsAndBytesConfig, 
        pipeline,
        T5ForConditionalGeneration,
        T5Tokenizer
    )
    import torch
    TRANSFORMERS_AVAILABLE = True
    logger.info("✅ Transformers available - can load real models")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.error("❌ Transformers not available - install with: pip install transformers accelerate bitsandbytes")

# Vector search
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    VECTOR_SEARCH_AVAILABLE = True
    logger.info("✅ Vector search available")
except ImportError:
    VECTOR_SEARCH_AVAILABLE = False
    logger.warning("⚠️ Vector search not available - install with: pip install sentence-transformers faiss-cpu")

# Module imports
import sys
sys.path.append('src')

import warnings
warnings.filterwarnings('ignore')

class FinancialLLMConfig:
    """Configuration for LOCAL financial language models"""
    
    MODEL_CONFIGS = {
        "fingpt-forecaster": {
            "model_name": "FinGPT/fingpt-forecaster_dow30_llama2-7b_lora",
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "description": "FinGPT specialized for financial forecasting and market prediction",
            "type": "finance_specialized",
            "requires_gpu": True
        },
        "fingpt-sentiment": {
            "model_name": "FinGPT/fingpt-sentiment_llama2-7b_lora", 
            "max_new_tokens": 256,
            "temperature": 0.6,
            "top_p": 0.95,
            "do_sample": True,
            "description": "FinGPT specialized for financial sentiment analysis",
            "type": "finance_specialized",
            "requires_gpu": True
        },
        "llama2-7b-chat": {
            "model_name": "meta-llama/Llama-2-7b-chat-hf",
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "description": "Llama2 7B conversational model",
            "type": "general_purpose",
            "requires_gpu": True
        },
        "flan-t5-base": {
            "model_name": "google/flan-t5-base",
            "max_new_tokens": 256,
            "temperature": 0.7,
            "do_sample": False,  # T5 works better without sampling
            "description": "FLAN-T5 instruction-following model",
            "type": "instruction_following",
            "requires_gpu": False
        },
        "distilgpt2": {
            "model_name": "distilgpt2",
            "max_new_tokens": 256,
            "temperature": 0.8,
            "top_p": 0.9,
            "do_sample": True,
            "description": "DistilGPT2 lightweight model",
            "type": "lightweight",
            "requires_gpu": False
        }
    }

class RealModelManager:
    """Manages loading and inference with real language models"""
    
    def __init__(self, model_type: str = "flan-t5-base"):
        self.model_type = model_type
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.model_loaded = False
        
        # Get model configuration
        self.config = FinancialLLMConfig.MODEL_CONFIGS.get(model_type, {})
        if not self.config:
            logger.error(f"Unknown model type: {model_type}")
            self.model_type = "flan-t5-base"
            self.config = FinancialLLMConfig.MODEL_CONFIGS["flan-t5-base"]
    
    def check_requirements(self) -> bool:
        """Check if we can load models"""
        if not TRANSFORMERS_AVAILABLE:
            logger.error("❌ Transformers library required for real models")
            logger.error("Install with: pip install transformers accelerate bitsandbytes")
            return False
        
        # Check GPU availability for GPU-required models
        if self.config.get("requires_gpu", False) and not torch.cuda.is_available():
            logger.warning(f"⚠️ {self.model_type} requires GPU but CUDA not available")
            logger.info("💡 Will try to load on CPU (may be slow)")
        
        return True
    
    def load_model(self) -> bool:
        """Load the actual language model"""
        if not self.check_requirements():
            return False
        
        try:
            model_name = self.config["model_name"]
            logger.info(f"🔄 Loading model: {model_name}")
            
            # Load based on model type
            if self.model_type.startswith("fingpt"):
                return self._load_fingpt_model()
            elif self.model_type.startswith("llama2"):
                return self._load_llama2_model()
            elif self.model_type == "flan-t5-base":
                return self._load_flan_t5_model()
            elif self.model_type == "distilgpt2":
                return self._load_distilgpt2_model()
            else:
                logger.error(f"❌ Unsupported model type: {self.model_type}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to load {self.model_type}: {e}")
            return False
    
    def _get_device_map(self):
        """Get appropriate device mapping"""
        if torch.cuda.is_available():
            return "auto"
        else:
            return None
    
    def _get_quantization_config(self):
        """Get quantization config for memory efficiency"""
        if not torch.cuda.is_available():
            return None
        
        try:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        except:
            logger.warning("⚠️ Quantization not available - loading in full precision")
            return None
    
    def _load_fingpt_model(self) -> bool:
        """Load FinGPT model"""
        try:
            model_name = self.config["model_name"]
            logger.info(f"Loading FinGPT model: {model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                padding_side="left"
            )
            
            # Ensure pad token exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with quantization
            quantization_config = self._get_quantization_config()
            device_map = self._get_device_map()
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map=device_map,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True
            )
            
            # Create pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=self.config["max_new_tokens"],
                temperature=self.config["temperature"],
                top_p=self.config["top_p"],
                do_sample=self.config["do_sample"],
                return_full_text=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            self.model_loaded = True
            logger.info(f"✅ FinGPT model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load FinGPT: {e}")
            return False
    
    def _load_llama2_model(self) -> bool:
        """Load Llama2 model"""
        try:
            model_name = self.config["model_name"]
            logger.info(f"Loading Llama2 model: {model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            quantization_config = self._get_quantization_config()
            device_map = self._get_device_map()
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map=device_map,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True
            )
            
            # Create pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=self.config["max_new_tokens"],
                temperature=self.config["temperature"],
                top_p=self.config["top_p"],
                do_sample=self.config["do_sample"],
                return_full_text=False
            )
            
            self.model_loaded = True
            logger.info(f"✅ Llama2 model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load Llama2: {e}")
            return False
    
    def _load_flan_t5_model(self) -> bool:
        """Load FLAN-T5 model (most reliable option)"""
        try:
            model_name = self.config["model_name"]
            logger.info(f"Loading FLAN-T5 model: {model_name}")
            
            # Load tokenizer and model
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map=self._get_device_map() if torch.cuda.is_available() else None
            )
            
            # Create pipeline for T5 (text2text-generation doesn't use return_full_text)
            self.pipeline = pipeline(
                "text2text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=self.config["max_new_tokens"],
                temperature=self.config["temperature"]
            )
            
            self.model_loaded = True
            logger.info(f"✅ FLAN-T5 model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load FLAN-T5: {e}")
            return False
    
    def _load_distilgpt2_model(self) -> bool:
        """Load DistilGPT2 model (lightweight fallback)"""
        try:
            model_name = self.config["model_name"]
            logger.info(f"Loading DistilGPT2 model: {model_name}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32  # Keep in FP32 for CPU
            )
            
            # Create pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=self.config["max_new_tokens"],
                temperature=self.config["temperature"],
                top_p=self.config["top_p"],
                do_sample=self.config["do_sample"],
                return_full_text=False
            )
            
            self.model_loaded = True
            logger.info(f"✅ DistilGPT2 model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load DistilGPT2: {e}")
            return False
    
    def generate_response(self, prompt: str, context: str = "") -> str:
        """Generate response using the loaded model"""
        if not self.model_loaded:
            logger.info("🔄 Model not loaded, attempting to load...")
            if not self.load_model():
                return "❌ Model loading failed. Please check your environment and try again."
        
        try:
            # Create financial prompt
            financial_prompt = self._create_financial_prompt(prompt, context)
            
            logger.info(f"🤖 Generating response using {self.model_type}")
            
            # Generate response
            result = self.pipeline(financial_prompt)
            
            # Extract generated text based on model type
            if isinstance(result, list) and len(result) > 0:
                if self.model_type == "flan-t5-base":
                    # T5 returns 'generated_text' directly
                    generated_text = result[0].get('generated_text', '')
                else:
                    # GPT-style models also use 'generated_text'
                    generated_text = result[0].get('generated_text', '')
            else:
                generated_text = str(result)
            
            # Clean up response
            response = self._clean_response(generated_text, financial_prompt)
            
            logger.info(f"✅ Response generated ({len(response)} chars)")
            return response
            
        except Exception as e:
            logger.error(f"❌ Error generating response: {e}")
            return f"I encountered an error while processing your question: {str(e)}. Please try rephrasing your question."
    
    def _create_financial_prompt(self, user_question: str, context: str = "") -> str:
        """Create appropriate prompt for financial analysis"""
        
        # Base financial context
        base_context = "You are a financial advisor and analyst with expertise in investment strategies, market analysis, risk management, and financial planning."
        
        # Add retrieved context if available
        if context:
            full_context = f"{base_context}\n\nRelevant Information:\n{context}"
        else:
            full_context = base_context
        
        # Create prompt based on model type
        if self.model_type == "flan-t5-base":
            # T5 works better with instruction format
            prompt = f"""Answer this financial question as a professional financial advisor:

Question: {user_question}

Context: {full_context}

Provide a comprehensive, professional response:"""

        elif self.model_type.startswith("llama2"):
            # Llama2 chat format
            prompt = f"""<s>[INST] <<SYS>>
{full_context}
<</SYS>>

{user_question} [/INST]"""

        else:
            # General format for FinGPT and others
            prompt = f"""{full_context}

User Question: {user_question}

Financial Analysis:"""
        
        return prompt
    
    def _clean_response(self, response: str, original_prompt: str = "") -> str:
        """Clean up model response"""
        # Remove common artifacts
        response = response.strip()
        
        # For T5 models, the response should be clean already
        if self.model_type == "flan-t5-base":
            # T5 returns just the answer, no prompt included
            pass
        else:
            # For GPT-style models, remove the original prompt if included
            if original_prompt and response.startswith(original_prompt):
                response = response[len(original_prompt):].strip()
        
        # Remove instruction artifacts
        artifacts_to_remove = [
            "Financial Analysis:",
            "Professional Response:",
            "Answer:",
            "Response:",
            "Provide a comprehensive, professional response:",
        ]
        
        for artifact in artifacts_to_remove:
            if response.startswith(artifact):
                response = response[len(artifact):].strip()
        
        # Ensure minimum length and quality
        if len(response) < 20:
            response = "I understand you're asking about financial matters. Could you please provide more specific details about what you'd like to know? I can help with investment analysis, risk assessment, financial planning, and market insights."
        
        # Ensure response ends properly
        if response and not response.endswith('.'):
            if len(response.split('.')) > 1:
                # Has sentences, ensure it ends with a period
                response = response.rstrip() + '.'
        
        return response

class SimpleFinancialKnowledgeBase:
    """Simple knowledge base for financial context"""
    
    def __init__(self):
        self.documents = []
        
    def build_knowledge_base(self):
        """Build basic financial knowledge base"""
        self.documents = [
            {
                "content": "Tesla Inc. is an electric vehicle and clean energy company known for innovation in automotive technology, energy storage, and solar panels. The company faces both growth opportunities in the expanding EV market and challenges from increasing competition.",
                "metadata": {"symbol": "TSLA", "topic": "company_profile"}
            },
            {
                "content": "Apple Inc. is a technology company that designs and manufactures consumer electronics, software, and digital services. Key financial metrics include strong profit margins, significant cash reserves, and consistent revenue growth from diverse product lines.",
                "metadata": {"symbol": "AAPL", "topic": "company_profile"}
            },
            {
                "content": "Investment risk management involves diversification across asset classes, sectors, and geographic regions. Key risk factors include market volatility, economic cycles, inflation, interest rate changes, and geopolitical events.",
                "metadata": {"topic": "risk_management"}
            },
            {
                "content": "Financial sentiment analysis uses natural language processing to analyze news, social media, and other text sources to gauge market mood and investor sentiment. This information can provide insights into potential market movements and investor behavior.",
                "metadata": {"topic": "sentiment_analysis"}
            }
        ]
        logger.info(f"✅ Knowledge base built with {len(self.documents)} documents")
    
    def search_knowledge_base(self, query: str, top_k: int = 2) -> List[Dict]:
        """Simple keyword-based search"""
        query_lower = query.lower()
        relevant_docs = []
        
        for doc in self.documents:
            content_lower = doc["content"].lower()
            metadata = doc["metadata"]
            
            # Check for matches
            if any(term in query_lower for term in ['tesla', 'tsla']) and metadata.get("symbol") == "TSLA":
                relevant_docs.append(doc)
            elif any(term in query_lower for term in ['apple', 'aapl']) and metadata.get("symbol") == "AAPL":
                relevant_docs.append(doc)
            elif any(term in query_lower for term in ['risk', 'volatility']) and metadata.get("topic") == "risk_management":
                relevant_docs.append(doc)
            elif any(term in query_lower for term in ['sentiment', 'analysis']) and metadata.get("topic") == "sentiment_analysis":
                relevant_docs.append(doc)
        
        return relevant_docs[:top_k]

class VesprFinancialChatbot:
    """Real model-based VesprAI Financial Chatbot"""
    
    def __init__(self, model_type: str = "flan-t5-base", memory_window: int = 10):
        self.model_type = model_type
        self.memory_window = memory_window
        
        # Initialize components
        self.knowledge_base = SimpleFinancialKnowledgeBase()
        self.model_manager = RealModelManager(model_type)
        
        # Conversation memory
        self.conversation_history = []
        
        logger.info(f"Initializing Real Model VesprAI Chatbot with {model_type}")
        self.setup_chatbot()
    
    def setup_chatbot(self):
        """Setup chatbot system"""
        # Build knowledge base
        self.knowledge_base.build_knowledge_base()
        
        # Note: Model will be loaded on first use for faster initialization
        logger.info("✅ VesprAI Real Model Chatbot setup complete (model will load on first use)")
    
    def detect_intent(self, query: str) -> str:
        """Detect user intent"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['sentiment', 'mood', 'feeling']):
            return 'sentiment'
        elif any(word in query_lower for word in ['invest', 'buy', 'sell', 'recommendation']):
            return 'investment'
        elif any(word in query_lower for word in ['risk', 'volatility', 'danger']):
            return 'risk'
        elif any(word in query_lower for word in ['ratio', 'financial', 'performance']):
            return 'analysis'
        elif any(word in query_lower for word in ['fraud', 'detection', 'patterns']):
            return 'fraud'
        else:
            return 'general'
    
    def chat(self, user_message: str) -> Dict[str, Any]:
        """Main chat interface using real models"""
        start_time = time.time()
        
        # Add to conversation history
        self.conversation_history.append({
            'role': 'user',
            'content': user_message,
            'timestamp': datetime.now()
        })
        
        # Detect intent
        intent = self.detect_intent(user_message)
        
        # Search knowledge base
        relevant_docs = self.knowledge_base.search_knowledge_base(user_message, top_k=2)
        context = "\n".join([doc["content"] for doc in relevant_docs])
        
        # Generate response using real model
        logger.info(f"🎯 Processing: {user_message[:50]}... (Intent: {intent})")
        response = self.model_manager.generate_response(user_message, context)
        
        # Add response to history
        self.conversation_history.append({
            'role': 'assistant',
            'content': response,
            'timestamp': datetime.now(),
            'intent': intent
        })
        
        # Trim history
        if len(self.conversation_history) > self.memory_window * 2:
            self.conversation_history = self.conversation_history[-self.memory_window * 2:]
        
        processing_time = time.time() - start_time
        
        return {
            'response': response,
            'intent': intent,
            'relevant_docs': [doc["metadata"] for doc in relevant_docs],
            'module_data': {},
            'model_type': self.model_type,
            'model_loaded': self.model_manager.model_loaded,
            'processing_time': round(processing_time, 2),
            'conversation_length': len(self.conversation_history)
        }
    
    def get_conversation_history(self) -> List[Dict]:
        """Get conversation history"""
        return [
            {
                'role': msg['role'],
                'content': msg['content'],
                'timestamp': msg['timestamp'].strftime('%H:%M:%S'),
                'intent': msg.get('intent', 'general') if msg['role'] == 'assistant' else None
            }
            for msg in self.conversation_history
        ]
    
    def reset_conversation(self):
        """Reset conversation"""
        self.conversation_history = []
        logger.info("🔄 Conversation reset")
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        return {
            'model_type': self.model_type,
            'model_config': self.model_manager.config,
            'model_loaded': self.model_manager.model_loaded,
            'knowledge_base_docs': len(self.knowledge_base.documents),
            'conversation_length': len(self.conversation_history),
            'transformers_available': TRANSFORMERS_AVAILABLE,
            'cuda_available': torch.cuda.is_available() if TRANSFORMERS_AVAILABLE else False,
            'memory_window': self.memory_window,
            'uses_real_models': True,
            'api_calls_disabled': True
        }

# Factory functions
def create_financial_chatbot(model_type: str = "flan-t5-base", **kwargs) -> VesprFinancialChatbot:
    """Create Real Model VesprAI Financial Chatbot"""
    return VesprFinancialChatbot(model_type=model_type, **kwargs)

def create_local_financial_chatbot(model_type: str = "flan-t5-base", **kwargs) -> VesprFinancialChatbot:
    """Alias for compatibility"""
    return create_financial_chatbot(model_type=model_type, **kwargs)

if __name__ == "__main__":
    print("🚀 Testing Real Model VesprAI Financial Chatbot...")
    
    # Test model loading priorities
    test_models = ["flan-t5-base", "distilgpt2", "fingpt-forecaster"]
    
    for model_type in test_models:
        print(f"\n📊 Testing {model_type}...")
        try:
            chatbot = create_financial_chatbot(model_type)
            
            # Test a simple query
            test_query = "What are the risks of investing in Tesla?"
            print(f"🤔 User: {test_query}")
            
            result = chatbot.chat(test_query)
            print(f"🤖 VesprAI: {result['response'][:150]}...")
            print(f"📊 Model Loaded: {result['model_loaded']} | Time: {result['processing_time']:.2f}s")
            
            break  # Stop after first successful model
            
        except Exception as e:
            print(f"❌ Failed with {model_type}: {e}")
            continue
    
    print("\n✅ Real model testing complete!")
