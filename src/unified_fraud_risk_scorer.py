"""
VesprAI Unified Fraud Risk Scorer - Module 3 (Complete System)
Includes both unsupervised NLP-only and hybrid supervised approaches
"""
import torch
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import logging
from pathlib import Path
import json
import joblib
import time
import os
from typing import Tuple, List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Import config
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import PATHS

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnifiedFraudRiskScorer:
    """
    VesprAI Unified Fraud Risk Scorer with both approaches:
    1. Unsupervised NLP-only (Isolation Forest)
    2. Hybrid Supervised (NLP + Numeric + Classification)
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Unsupervised components
        self.contamination = 0.005  # Optimized contamination rate
        self.unsupervised_detector = None
        self.unsupervised_scaler = StandardScaler()
        self.unsupervised_pca = PCA(n_components=0.95)
        
        # Hybrid components
        self.hybrid_classifier = None
        self.text_scaler = StandardScaler()
        self.numeric_scaler = StandardScaler()
        
        logger.info(f"Initialized UnifiedFraudRiskScorer with {model_name}")
        self._load_encoder()
        self._setup_models()
    
    def _load_encoder(self):
        """Load Sentence-BERT encoder"""
        self.encoder = SentenceTransformer(self.model_name)
        logger.info(f"Loaded Sentence-BERT encoder")
    
    def _setup_models(self):
        """Setup both unsupervised and supervised models"""
        # Unsupervised model
        self.unsupervised_detector = IsolationForest(
            n_estimators=500,
            contamination=self.contamination,
            max_samples=0.8,
            max_features=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        # Hybrid supervised model
        self.hybrid_classifier = LogisticRegression(
            max_iter=2000,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            C=0.1
        )
        
        logger.info("Initialized both unsupervised and hybrid models")
    
    def load_local_paysim_dataset(self) -> str:
        """Load local PaySim dataset"""
        possible_paths = [
            Path("data/raw/paysim.csv"),
            Path("../data/raw/paysim.csv"), 
            Path("../../data/raw/paysim.csv"),
            Path(r"C:\Users\siddh\Downloads\DATA641 Final (Vespr)\data\raw\paysim.csv"),
            Path.cwd() / "data" / "raw" / "paysim.csv",
            Path.cwd().parent / "data" / "raw" / "paysim.csv"
        ]
        
        for paysim_path in possible_paths:
            if paysim_path.exists():
                logger.info(f"✅ Found PaySim dataset: {paysim_path}")
                return str(paysim_path)
        
        raise FileNotFoundError("PaySim dataset not found. Please place paysim.csv in data/raw/ directory.")
    
    def load_paysim_data(self, sample_size: int = 50000) -> pd.DataFrame:
        """Load and preprocess PaySim data"""
        logger.info("Loading PaySim data...")
        
        # Load CSV
        data_path = self.load_local_paysim_dataset()
        df = pd.read_csv(data_path)
        logger.info(f"Loaded PaySim dataset: {len(df):,} transactions")
        
        # Sample if needed
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
            logger.info(f"Sampled to {len(df):,} transactions")
        
        # Create transaction narratives
        df["text"] = (
            "A financial transaction where account "
            + df["nameOrig"] + " sent " + df["amount"].astype(str)
            + " units to account " + df["nameDest"] 
            + " using transaction type " + df["type"]
            + ". Sender balance changed from " + df["oldbalanceOrg"].astype(str)
            + " to " + df["newbalanceOrig"].astype(str)
            + ". Receiver balance changed from " + df["oldbalanceDest"].astype(str)
            + " to " + df["newbalanceDest"].astype(str) + "."
        )
        
        # Clean data
        df = df.dropna()
        df["is_fraud"] = df["isFraud"].astype(int)
        
        logger.info(f"Final dataset: {len(df):,} transactions")
        logger.info(f"Fraud rate: {df['is_fraud'].mean():.4f}")
        
        return df
    
    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare training and test datasets"""
        df = self.load_paysim_data()
        
        # Split with stratification
        train_df, test_df = train_test_split(
            df, test_size=0.2, stratify=df['is_fraud'], random_state=42
        )
        
        logger.info(f"Training: {len(train_df):,} samples")
        logger.info(f"Testing: {len(test_df):,} samples")
        
        return train_df, test_df
    
    def encode_transactions(self, texts: List[str]) -> np.ndarray:
        """Encode transactions with Sentence-BERT"""
        logger.info(f"Encoding {len(texts):,} transactions...")
        embeddings = self.encoder.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings
    
    def extract_numeric_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract numeric features for hybrid approach"""
        # Core numeric features
        numeric_features = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
        
        df_features = df.copy()
        
        # Derived features
        df_features['sender_balance_change'] = df_features['oldbalanceOrg'] - df_features['newbalanceOrig']
        df_features['receiver_balance_change'] = df_features['newbalanceDest'] - df_features['oldbalanceDest']
        df_features['amount_to_sender_balance'] = df_features['amount'] / (df_features['oldbalanceOrg'] + 1)
        df_features['sender_balance_remaining_ratio'] = df_features['newbalanceOrig'] / (df_features['oldbalanceOrg'] + 1)
        
        # Zero balance indicators
        df_features['sender_zero_after'] = (df_features['newbalanceOrig'] == 0).astype(int)
        df_features['receiver_zero_after'] = (df_features['newbalanceDest'] == 0).astype(int)
        
        # Transaction type encoding
        all_types = ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']  # All PaySim transaction types
        for trans_type in all_types:
            df_features[f'type_{trans_type}'] = (df_features['type'] == trans_type).astype(int)

        
        # Combine features
        feature_columns = (numeric_features + 
                  ['sender_balance_change', 'receiver_balance_change',
                   'amount_to_sender_balance', 'sender_balance_remaining_ratio',
                   'sender_zero_after', 'receiver_zero_after'] +
                  [f'type_{t}' for t in all_types])
        
        features = df_features[feature_columns].fillna(0)
        
        return features.values
    
    def train_unsupervised(self, train_df: pd.DataFrame) -> Dict[str, Any]:
        """Train unsupervised NLP-only model"""
        logger.info("Training Unsupervised NLP-only Model...")
        
        start_time = time.time()
        
        # Filter to normal transactions only
        normal_df = train_df[train_df['is_fraud'] == 0].copy()
        logger.info(f"Training on {len(normal_df):,} normal transactions")
        
        # Encode and enhance embeddings
        embeddings = self.encode_transactions(normal_df['text'].tolist())
        embeddings_scaled = self.unsupervised_scaler.fit_transform(embeddings)
        embeddings_pca = self.unsupervised_pca.fit_transform(embeddings_scaled)
        
        # Train Isolation Forest
        self.unsupervised_detector.fit(embeddings_pca)
        
        training_time = time.time() - start_time
        
        return {
            "method": "unsupervised_nlp_only",
            "training_time": training_time,
            "training_samples": len(normal_df),
            "pca_components": self.unsupervised_pca.n_components_,
            "explained_variance": self.unsupervised_pca.explained_variance_ratio_.sum()
        }
    
    def train_hybrid(self, train_df: pd.DataFrame) -> Dict[str, Any]:
        """Train hybrid supervised model"""
        logger.info("Training Hybrid Supervised Model...")
        
        start_time = time.time()
        
        # Extract features
        text_embeddings = self.encode_transactions(train_df['text'].tolist())
        numeric_features = self.extract_numeric_features(train_df)
        
        # Scale features
        text_embeddings_scaled = self.text_scaler.fit_transform(text_embeddings)
        numeric_features_scaled = self.numeric_scaler.fit_transform(numeric_features)
        
        # Combine features
        combined_features = np.hstack([text_embeddings_scaled, numeric_features_scaled])
        labels = train_df['is_fraud'].values
        
        # Train classifier
        self.hybrid_classifier.fit(combined_features, labels)
        
        training_time = time.time() - start_time
        
        # Training evaluation
        train_predictions = self.hybrid_classifier.predict_proba(combined_features)[:, 1]
        train_auc = roc_auc_score(labels, train_predictions)
        
        return {
            "method": "hybrid_supervised",
            "training_time": training_time,
            "training_samples": len(train_df),
            "training_auc": train_auc,
            "text_features": text_embeddings_scaled.shape[1],
            "numeric_features": numeric_features_scaled.shape[1],
            "total_features": combined_features.shape[1]
        }
    
    def predict_unsupervised(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict using unsupervised model"""
        embeddings = self.encode_transactions(df['text'].tolist())
        embeddings_scaled = self.unsupervised_scaler.transform(embeddings)
        embeddings_pca = self.unsupervised_pca.transform(embeddings_scaled)
        
        scores = -self.unsupervised_detector.score_samples(embeddings_pca)
        labels = self.unsupervised_detector.predict(embeddings_pca)
        
        return scores, labels
    
    def predict_hybrid(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict using hybrid model"""
        text_embeddings = self.encode_transactions(df['text'].tolist())
        numeric_features = self.extract_numeric_features(df)
        
        text_embeddings_scaled = self.text_scaler.transform(text_embeddings)
        numeric_features_scaled = self.numeric_scaler.transform(numeric_features)
        
        combined_features = np.hstack([text_embeddings_scaled, numeric_features_scaled])
        
        probabilities = self.hybrid_classifier.predict_proba(combined_features)[:, 1]
        predictions = self.hybrid_classifier.predict(combined_features)
        
        return probabilities, predictions
    
    def evaluate_model(self, test_df: pd.DataFrame, method: str = "both") -> Dict[str, Any]:
        """Evaluate model(s) performance"""
        results = {}
        
        if method in ["unsupervised", "both"]:
            logger.info("Evaluating Unsupervised Model...")
            scores, pred_labels = self.predict_unsupervised(test_df)
            true_labels = test_df['is_fraud'].values
            
            binary_preds = (pred_labels == -1).astype(int)
            auc = roc_auc_score(true_labels, scores)
            cm = confusion_matrix(true_labels, binary_preds)
            report = classification_report(true_labels, binary_preds, output_dict=True, zero_division=0)
            
            results["unsupervised"] = {
                "roc_auc": auc,
                "precision": report.get('1', {}).get('precision', 0.0),
                "recall": report.get('1', {}).get('recall', 0.0),
                "f1_score": report.get('1', {}).get('f1-score', 0.0),
                "accuracy": report['accuracy'],
                "confusion_matrix": cm.tolist()
            }
        
        if method in ["hybrid", "both"]:
            logger.info("Evaluating Hybrid Model...")
            probabilities, predictions = self.predict_hybrid(test_df)
            true_labels = test_df['is_fraud'].values
            
            auc = roc_auc_score(true_labels, probabilities)
            cm = confusion_matrix(true_labels, predictions)
            report = classification_report(true_labels, predictions, output_dict=True, zero_division=0)
            
            results["hybrid"] = {
                "roc_auc": auc,
                "precision": report.get('1', {}).get('precision', 0.0),
                "recall": report.get('1', {}).get('recall', 0.0),
                "f1_score": report.get('1', {}).get('f1-score', 0.0),
                "accuracy": report['accuracy'],
                "confusion_matrix": cm.tolist()
            }
        
        return results
    
    def save_models(self, model_dir: str = None):
        """Save both models"""
        if model_dir is None:
            model_dir = PATHS.get('fraud_model', Path('models/unified_fraud_scorer'))
        
        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Save unsupervised components
        if self.unsupervised_detector:
            joblib.dump(self.unsupervised_detector, model_path / "unsupervised_detector.joblib")
            joblib.dump(self.unsupervised_scaler, model_path / "unsupervised_scaler.joblib")
            joblib.dump(self.unsupervised_pca, model_path / "unsupervised_pca.joblib")
        
        # Save hybrid components
        if self.hybrid_classifier:
            joblib.dump(self.hybrid_classifier, model_path / "hybrid_classifier.joblib")
            joblib.dump(self.text_scaler, model_path / "text_scaler.joblib")
            joblib.dump(self.numeric_scaler, model_path / "numeric_scaler.joblib")
        
        # Save configuration
        config = {
            "model_name": self.model_name,
            "contamination": self.contamination,
            "approaches": ["unsupervised_nlp_only", "hybrid_supervised"],
            "target_performance": {
                "unsupervised": "0.55-0.65 AUC",
                "hybrid": "0.90+ AUC"
            }
        }
        
        with open(model_path / "unified_config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Unified models saved to {model_path}")
        return str(model_path)
    
    def score_transaction(self, text: str, method: str = "hybrid", **kwargs) -> Dict[str, Any]:
        """Score single transaction with specified method"""
        
        if method == "unsupervised":
            df_single = pd.DataFrame({"text": [text]})
            scores, labels = self.predict_unsupervised(df_single)
            
            risk_score = scores[0]
            is_suspicious = labels[0] == -1
            risk_percentage = min(100, max(0, risk_score * 20))
            
            return {
                "text": text,
                "method": "unsupervised_nlp_only",
                "risk_score": float(risk_score),
                "risk_percentage": float(risk_percentage),
                "is_suspicious": bool(is_suspicious),
                "risk_level": "HIGH" if is_suspicious else "LOW"
            }
        
        elif method == "hybrid":
            # For hybrid, need numeric features
            required_fields = ["amount", "old_balance_org", "new_balance_orig", 
                             "old_balance_dest", "new_balance_dest", "trans_type"]
            
            if not all(field in kwargs for field in required_fields):
                return {"error": f"Hybrid method requires: {required_fields}"}
            
            temp_df = pd.DataFrame({
                'text': [text],
                'amount': [kwargs['amount']],
                'oldbalanceOrg': [kwargs['old_balance_org']],
                'newbalanceOrig': [kwargs['new_balance_orig']],
                'oldbalanceDest': [kwargs['old_balance_dest']],
                'newbalanceDest': [kwargs['new_balance_dest']],
                'type': [kwargs['trans_type']],
                'nameOrig': ['C1234567'],
                'nameDest': ['M7654321']
            })
            
            probabilities, predictions = self.predict_hybrid(temp_df)
            
            risk_score = probabilities[0]
            risk_percentage = risk_score * 100
            
            return {
                "text": text,
                "method": "hybrid_supervised",
                "risk_score": float(risk_score),
                "risk_percentage": float(risk_percentage),
                "is_fraud": bool(predictions[0]),
                "risk_level": "HIGH" if risk_percentage > 50 else "LOW"
            }

# For backward compatibility
FraudRiskScorer = UnifiedFraudRiskScorer

if __name__ == "__main__":
    # Test unified approach
    scorer = UnifiedFraudRiskScorer()
    train_df, test_df = scorer.prepare_data()
    
    # Train both models
    unsup_results = scorer.train_unsupervised(train_df)
    hybrid_results = scorer.train_hybrid(train_df)
    
    print(f"Unsupervised training: {unsup_results}")
    print(f"Hybrid training: {hybrid_results}")
    
    # Evaluate both
    eval_results = scorer.evaluate_model(test_df, method="both")
    print(f"Evaluation results: {eval_results}")
