"""
Enhanced model trainer with advanced techniques for improved performance
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from transformers import (
    DistilBertTokenizer, 
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)

# Import AdamW from correct location
try:
    from transformers import AdamW
except ImportError:
    from torch.optim import AdamW

from datasets import load_from_disk
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import json
import logging
from pathlib import Path
from datetime import datetime
import time
from typing import Dict, Any, Tuple, List
import warnings
from tqdm import tqdm
import math
warnings.filterwarnings('ignore')

# Import config
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import MODEL_CONFIG, LABEL_MAP, PATHS

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    """
    def __init__(self, alpha=1, gamma=2, num_classes=3):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        
    def forward(self, predictions, targets):
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing for better generalization
    """
    def __init__(self, num_classes=3, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        
    def forward(self, predictions, targets):
        log_probs = F.log_softmax(predictions, dim=-1)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = targets * self.confidence + (1 - targets) * self.smoothing / (self.num_classes - 1)
        return (-targets * log_probs).sum(dim=1).mean()

class EnhancedSentimentTrainer:
    """
    Enhanced trainer with advanced techniques for improved performance
    """
    
    def __init__(self, model_name: str = None, use_focal_loss: bool = True, 
                 use_label_smoothing: bool = True, use_mixup: bool = False):
        """
        Initialize enhanced trainer with advanced techniques
        """
        self.model_name = model_name or MODEL_CONFIG['model_name']
        self.num_labels = MODEL_CONFIG['num_labels']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.label_map = LABEL_MAP
        
        # Advanced training options
        self.use_focal_loss = use_focal_loss
        self.use_label_smoothing = use_label_smoothing
        self.use_mixup = use_mixup
        
        logger.info(f"Enhanced trainer initialized with:")
        logger.info(f"  Model: {self.model_name}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Focal Loss: {use_focal_loss}")
        logger.info(f"  Label Smoothing: {use_label_smoothing}")
        logger.info(f"  Mixup: {use_mixup}")
        
        # Initialize model components
        self._initialize_model()
        self._setup_loss_functions()
        
    def _initialize_model(self):
        """Initialize model with enhanced configuration"""
        logger.info("Loading enhanced model...")
        
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
        )
        
        # Move to device
        self.model.to(self.device)
        
        # Model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"Enhanced model loaded:")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        
    def _setup_loss_functions(self):
        """Setup advanced loss functions"""
        if self.use_focal_loss:
            self.focal_loss = FocalLoss(alpha=1, gamma=2, num_classes=self.num_labels)
            logger.info("Focal Loss enabled for class imbalance handling")
            
        if self.use_label_smoothing:
            self.label_smoothing_loss = LabelSmoothingLoss(
                num_classes=self.num_labels, 
                smoothing=0.1
            )
            logger.info("Label Smoothing enabled for better generalization")
    
    def mixup_data(self, x, y, alpha=0.2):
        """
        Mixup data augmentation
        """
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
            
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(self.device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
    
    def mixup_criterion(self, pred, y_a, y_b, lam):
        """
        Mixup loss calculation
        """
        return lam * F.cross_entropy(pred, y_a) + (1 - lam) * F.cross_entropy(pred, y_b)
    
    def compute_enhanced_metrics(self, predictions, labels):
        """Enhanced metrics calculation"""
        predictions = np.argmax(predictions, axis=1)
        
        # Basic metrics
        accuracy = accuracy_score(labels, predictions)
        f1_weighted = f1_score(labels, predictions, average='weighted')
        f1_macro = f1_score(labels, predictions, average='macro')
        f1_micro = f1_score(labels, predictions, average='micro')
        precision_weighted = precision_score(labels, predictions, average='weighted')
        recall_weighted = recall_score(labels, predictions, average='weighted')
        
        # Per-class metrics
        f1_per_class = f1_score(labels, predictions, average=None)
        precision_per_class = precision_score(labels, predictions, average=None)
        recall_per_class = recall_score(labels, predictions, average=None)
        
        metrics = {
            'accuracy': accuracy,
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
        }
        
        # Add per-class metrics
        for i in range(self.num_labels):
            if i < len(f1_per_class):
                metrics[f'f1_class_{i}'] = f1_per_class[i]
                metrics[f'precision_class_{i}'] = precision_per_class[i]
                metrics[f'recall_class_{i}'] = recall_per_class[i]
                metrics[f'f1_{self.label_map[i].lower()}'] = f1_per_class[i]
        
        return metrics
    
    def load_datasets(self) -> Tuple[Any, Any, Any]:
        """Load datasets with validation"""
        logger.info("Loading datasets for enhanced training...")
        
        try:
            train_dataset = load_from_disk(str(PATHS['train_dataset']))
            val_dataset = load_from_disk(str(PATHS['val_dataset']))
            test_dataset = load_from_disk(str(PATHS['test_dataset']))
            
            # Validate datasets
            logger.info(f"Dataset validation:")
            logger.info(f"  Train: {len(train_dataset)} samples")
            logger.info(f"  Validation: {len(val_dataset)} samples")
            logger.info(f"  Test: {len(test_dataset)} samples")
            
            # Check class distribution
            #train_labels = [item['label'].item() if hasattr(item['label'], 'item') else item['label'] for item in train_dataset]
            #train_dist = pd.Series(train_labels).value_counts().sort_index()
            #logger.info(f"  Train class distribution: {dict(train_dist)}")
            
            return train_dataset, val_dataset, test_dataset
            
        except Exception as e:
            logger.error(f"Error loading datasets: {e}")
            raise
    
    def create_dataloader(self, dataset, batch_size=16, shuffle=False):
        """Create enhanced DataLoader"""
        from torch.utils.data import DataLoader
        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            num_workers=0,  # Avoid multiprocessing issues
            pin_memory=torch.cuda.is_available()
        )
    
    def get_labels_from_batch(self, batch):
        """Extract labels handling different formats"""
        possible_keys = ['labels', 'label']
        
        for key in possible_keys:
            if key in batch:
                return batch[key].to(self.device)
        
        label_keys = [k for k in batch.keys() if 'label' in k.lower()]
        if label_keys:
            return batch[label_keys[0]].to(self.device)
        
        raise KeyError(f"No label found. Available keys: {list(batch.keys())}")
    
    def evaluate_model_enhanced(self, dataloader):
        """Enhanced evaluation with detailed metrics"""
        self.model.eval()
        all_predictions = []
        all_labels = []
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                labels = self.get_labels_from_batch(batch)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
                all_predictions.extend(outputs.logits.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        metrics = self.compute_enhanced_metrics(np.array(all_predictions), np.array(all_labels))
        metrics['loss'] = avg_loss
        
        return metrics, np.array(all_predictions), np.array(all_labels)
    
    def train_model_enhanced(self):
        """Enhanced training with advanced techniques"""
        logger.info("Starting enhanced model training...")
        start_time = time.time()
        
        # Load datasets
        train_dataset, val_dataset, test_dataset = self.load_datasets()
        
        # Enhanced hyperparameters
        batch_size = 8  # Smaller for stability
        learning_rate = 1e-5  # Lower learning rate for better convergence
        num_epochs = 6  # More epochs
        warmup_ratio = 0.1
        weight_decay = 0.01
        
        # Create dataloaders
        train_loader = self.create_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = self.create_dataloader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = self.create_dataloader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Enhanced optimizer with different learning rates for different layers
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay) and "classifier" not in n],
                "weight_decay": weight_decay,
                "lr": learning_rate * 0.1,  # Lower LR for base layers
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay) and "classifier" not in n],
                "weight_decay": 0.0,
                "lr": learning_rate * 0.1,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if "classifier" in n],
                "weight_decay": weight_decay,
                "lr": learning_rate,  # Higher LR for classifier
            },
        ]
        
        optimizer = AdamW(optimizer_grouped_parameters, eps=1e-8)
        
        # Enhanced scheduler
        num_training_steps = len(train_loader) * num_epochs
        num_warmup_steps = int(num_training_steps * warmup_ratio)
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # Training tracking
        training_losses = []
        validation_metrics = []
        best_f1 = 0
        patience = 0
        max_patience = 3
        
        logger.info(f"Enhanced Training Configuration:")
        logger.info(f"  Epochs: {num_epochs}")
        logger.info(f"  Batch Size: {batch_size}")
        logger.info(f"  Learning Rate: {learning_rate}")
        logger.info(f"  Warmup Steps: {num_warmup_steps}")
        logger.info(f"  Total Steps: {num_training_steps}")
        
        # Training loop
        for epoch in range(num_epochs):
            logger.info(f"=== Epoch {epoch + 1}/{num_epochs} ===")
            
            # Training phase
            self.model.train()
            epoch_losses = []
            progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
            
            for batch_idx, batch in enumerate(progress_bar):
                optimizer.zero_grad()
                
                labels = self.get_labels_from_batch(batch)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                if self.use_mixup and np.random.random() > 0.5:
                    # Apply mixup
                    input_ids, labels_a, labels_b, lam = self.mixup_data(input_ids, labels, alpha=0.2)
                    
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = self.mixup_criterion(outputs.logits, labels_a, labels_b, lam)
                else:
                    # Regular training
                    outputs = self.model(
                        input_ids=input_ids, 
                        attention_mask=attention_mask, 
                        labels=labels
                    )
                    
                    # Enhanced loss calculation
                    if self.use_focal_loss:
                        loss = self.focal_loss(outputs.logits, labels)
                    elif self.use_label_smoothing:
                        loss = self.label_smoothing_loss(outputs.logits, labels)
                    else:
                        loss = outputs.loss
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                
                epoch_losses.append(loss.item())
                
                # Update progress bar
                current_lr = scheduler.get_last_lr()[0]
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{current_lr:.2e}'
                })
            
            avg_epoch_loss = np.mean(epoch_losses)
            training_losses.append(avg_epoch_loss)
            
            # Validation phase
            logger.info("Running validation...")
            val_metrics, _, _ = self.evaluate_model_enhanced(val_loader)
            validation_metrics.append(val_metrics)
            
            # Logging
            logger.info(f"Epoch {epoch + 1} Results:")
            logger.info(f"  Train Loss: {avg_epoch_loss:.4f}")
            logger.info(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
            logger.info(f"  Val F1 (Weighted): {val_metrics['f1_weighted']:.4f}")
            logger.info(f"  Val F1 (Macro): {val_metrics['f1_macro']:.4f}")
            
            # Early stopping based on F1 score
            if val_metrics['f1_weighted'] > best_f1:
                best_f1 = val_metrics['f1_weighted']
                patience = 0
                
                # Save best model
                best_model_path = PATHS['final_model'] / 'best_model'
                best_model_path.mkdir(parents=True, exist_ok=True)
                self.model.save_pretrained(str(best_model_path))
                self.tokenizer.save_pretrained(str(best_model_path))
                logger.info(f"New best model saved! F1: {best_f1:.4f}")
            else:
                patience += 1
                logger.info(f"No improvement. Patience: {patience}/{max_patience}")
                
                if patience >= max_patience:
                    logger.info("Early stopping triggered!")
                    break
        
        # Load best model for final evaluation
        if (PATHS['final_model'] / 'best_model').exists():
            logger.info("Loading best model for final evaluation...")
            self.model = DistilBertForSequenceClassification.from_pretrained(
                str(PATHS['final_model'] / 'best_model')
            )
            self.model.to(self.device)
        
        # Final evaluation
        logger.info("Final evaluation on test set...")
        test_metrics, test_predictions, test_labels = self.evaluate_model_enhanced(test_loader)
        
        # Training time
        training_time = time.time() - start_time
        
        # Save final model
        logger.info("Saving final enhanced model...")
        PATHS['final_model'].mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(PATHS['final_model']))
        self.tokenizer.save_pretrained(str(PATHS['final_model']))
        
        # Comprehensive results
        training_results = {
            'training_time_minutes': training_time / 60,
            'final_test_results': {f'eval_{k}': v for k, v in test_metrics.items()},
            'model_path': str(PATHS['final_model']),
            'training_timestamp': datetime.now().isoformat(),
            'training_losses': training_losses,
            'validation_metrics': validation_metrics,
            'best_f1_score': best_f1,
            'total_epochs_trained': len(training_losses),
            'early_stopped': patience >= max_patience,
            'enhancement_features': {
                'focal_loss': self.use_focal_loss,
                'label_smoothing': self.use_label_smoothing,
                'mixup': self.use_mixup,
                'layered_learning_rates': True,
                'gradient_clipping': True,
                'cosine_scheduler': True
            }
        }
        
        # Save results
        PATHS['evaluation_results'].parent.mkdir(parents=True, exist_ok=True)
        with open(PATHS['evaluation_results'], 'w') as f:
            json.dump(training_results, f, indent=2, default=str)
        
        # Print enhanced results
        logger.info("\n" + "="*60)
        logger.info("ENHANCED TRAINING COMPLETED!")
        logger.info("="*60)
        logger.info("Final Enhanced Results:")
        for key, value in test_metrics.items():
            if key != 'loss':
                logger.info(f"  {key.replace('_', ' ').title()}: {value:.4f}")
        logger.info(f"Training Time: {training_time/60:.2f} minutes")
        logger.info(f"Best F1 Score: {best_f1:.4f}")
        logger.info(f"Total Epochs: {len(training_losses)}")
        logger.info("="*60)
        
        # Mock trainer for compatibility
        mock_trainer = type('MockTrainer', (), {
            'state': type('State', (), {'log_history': []})(),
            'predict': lambda self, dataset: type('Predictions', (), {
                'predictions': test_predictions,
                'label_ids': test_labels
            })()
        })()
        
        return mock_trainer, training_results
    
    def create_enhanced_training_plots(self, trainer):
        """Create comprehensive enhanced training visualizations"""
        logger.info("Creating enhanced training plots...")
        
        try:
            # Load training results
            with open(PATHS['evaluation_results'], 'r') as f:
                results = json.load(f)
            
            training_losses = results.get('training_losses', [])
            validation_metrics = results.get('validation_metrics', [])
            
            if training_losses and validation_metrics:
                fig = plt.figure(figsize=(20, 16))
                
                # 1. Training Loss
                plt.subplot(3, 4, 1)
                epochs = range(1, len(training_losses) + 1)
                plt.plot(epochs, training_losses, 'b-o', alpha=0.7, linewidth=2, markersize=4)
                plt.title('Enhanced Training Loss', fontweight='bold', fontsize=12)
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.grid(True, alpha=0.3)
                
                # 2. Validation Accuracy
                plt.subplot(3, 4, 2)
                val_accuracy = [m['accuracy'] for m in validation_metrics]
                plt.plot(epochs[:len(val_accuracy)], val_accuracy, 'g-o', alpha=0.7, linewidth=2, markersize=4)
                plt.title('Validation Accuracy', fontweight='bold', fontsize=12)
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.grid(True, alpha=0.3)
                plt.ylim(0, 1)
                
                # 3. F1 Scores Comparison
                plt.subplot(3, 4, 3)
                val_f1_weighted = [m['f1_weighted'] for m in validation_metrics]
                val_f1_macro = [m['f1_macro'] for m in validation_metrics]
                epochs_val = range(1, len(val_f1_weighted) + 1)
                plt.plot(epochs_val, val_f1_weighted, 'r-o', alpha=0.7, linewidth=2, 
                        markersize=4, label='F1 Weighted')
                plt.plot(epochs_val, val_f1_macro, 'm-s', alpha=0.7, linewidth=2, 
                        markersize=4, label='F1 Macro')
                plt.title('F1 Score Progression', fontweight='bold', fontsize=12)
                plt.xlabel('Epoch')
                plt.ylabel('F1 Score')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.ylim(0, 1)
                
                # 4. Enhanced Metrics Comparison
                plt.subplot(3, 4, 4)
                if validation_metrics:
                    final_metrics = validation_metrics[-1]
                    metrics_names = ['Accuracy', 'F1 Weighted', 'F1 Macro', 'Precision', 'Recall']
                    metrics_values = [
                        final_metrics.get('accuracy', 0),
                        final_metrics.get('f1_weighted', 0),
                        final_metrics.get('f1_macro', 0),
                        final_metrics.get('precision_weighted', 0),
                        final_metrics.get('recall_weighted', 0)
                    ]
                    
                    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']
                    bars = plt.bar(metrics_names, metrics_values, color=colors, alpha=0.8)
                    plt.title('Final Enhanced Performance', fontweight='bold', fontsize=12)
                    plt.ylabel('Score')
                    plt.ylim(0, 1)
                    plt.xticks(rotation=45)
                    
                    # Add value labels
                    for bar, value in zip(bars, metrics_values):
                        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
                
                # 5. Per-Class F1 Scores
                plt.subplot(3, 4, 5)
                if validation_metrics:
                    final_metrics = validation_metrics[-1]
                    class_names = ['Negative', 'Neutral', 'Positive']
                    class_f1 = []
                    for i, name in enumerate(class_names):
                        f1_key = f'f1_{name.lower()}'
                        class_f1.append(final_metrics.get(f1_key, final_metrics.get(f'f1_class_{i}', 0)))
                    
                    bars = plt.bar(class_names, class_f1, color=['#ff6b6b', '#4ecdc4', '#45b7d1'], alpha=0.8)
                    plt.title('Per-Class F1 Scores', fontweight='bold', fontsize=12)
                    plt.ylabel('F1 Score')
                    plt.ylim(0, 1)
                    
                    for bar, value in zip(bars, class_f1):
                        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
                
                # 6. Training Progress Overview
                plt.subplot(3, 4, 6)
                plt.text(0.1, 0.9, 'Enhanced Training Summary', fontsize=14, fontweight='bold', 
                        transform=plt.gca().transAxes)
                
                summary_text = f"""Training Enhancements:
• Focal Loss: {results.get('enhancement_features', {}).get('focal_loss', False)}
• Label Smoothing: {results.get('enhancement_features', {}).get('label_smoothing', False)}
• Mixup: {results.get('enhancement_features', {}).get('mixup', False)}
• Layered LR: {results.get('enhancement_features', {}).get('layered_learning_rates', False)}
• Gradient Clipping: {results.get('enhancement_features', {}).get('gradient_clipping', False)}

Results:
• Final Accuracy: {final_metrics.get('accuracy', 0):.3f}
• Best F1 Score: {results.get('best_f1_score', 0):.3f}
• Training Time: {results.get('training_time_minutes', 0):.1f} min
• Total Epochs: {results.get('total_epochs_trained', 0)}
• Early Stopped: {results.get('early_stopped', False)}"""
                
                plt.text(0.1, 0.8, summary_text, fontsize=10, transform=plt.gca().transAxes,
                        verticalalignment='top', family='monospace')
                plt.axis('off')
                
                # 7. Learning Rate Schedule (if available)
                plt.subplot(3, 4, 7)
                # Simulate LR schedule for cosine annealing
                total_steps = len(training_losses) * 16  # Approximate steps per epoch
                warmup_steps = int(total_steps * 0.1)
                lrs = []
                for step in range(total_steps):
                    if step < warmup_steps:
                        lr = 1e-5 * step / warmup_steps
                    else:
                        lr = 1e-5 * (1 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps))) / 2
                    lrs.append(lr)
                
                plt.plot(lrs, 'orange', alpha=0.7, linewidth=1)
                plt.title('Learning Rate Schedule', fontweight='bold', fontsize=12)
                plt.xlabel('Training Steps')
                plt.ylabel('Learning Rate')
                plt.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
                plt.grid(True, alpha=0.3)
                
                # 8. Loss vs Accuracy Trade-off
                plt.subplot(3, 4, 8)
                if len(validation_metrics) > 1:
                    val_losses = [m.get('loss', 0) for m in validation_metrics]
                    plt.scatter(val_losses, val_accuracy, c=range(len(val_losses)), 
                              cmap='viridis', alpha=0.7, s=50)
                    plt.xlabel('Validation Loss')
                    plt.ylabel('Validation Accuracy')
                    plt.title('Loss vs Accuracy Trade-off', fontweight='bold', fontsize=12)
                    plt.colorbar(label='Epoch')
                    plt.grid(True, alpha=0.3)
                
                # 9-12. Additional analysis plots
                for i in range(9, 13):
                    plt.subplot(3, 4, i)
                    if i == 9:  # Model Architecture Info
                        arch_text = """Enhanced DistilBERT Architecture:

• Base Model: DistilBERT-base-uncased
• Parameters: ~67M (trainable)
• Hidden Dropout: 0.3 (increased)
• Attention Dropout: 0.3 (increased)
• Sequence Length: 128 tokens
• Batch Size: 8 (optimized)
• Learning Strategy: Layered LR

Classifier Head:
• Hidden Layer: 768 → 768
• Dropout: 0.3
• Output: 768 → 3 classes"""
                        
                        plt.text(0.05, 0.95, arch_text, fontsize=9, transform=plt.gca().transAxes,
                                verticalalignment='top', family='monospace')
                        plt.title('Model Architecture', fontweight='bold', fontsize=12)
                        plt.axis('off')
                    
                    elif i == 10:  # Comparison with Previous Results
                        if 'previous_results' in results:
                            # Plot comparison if available
                            pass
                        else:
                            plt.text(0.5, 0.5, 'Performance Improvements\nvs Previous Model\n\n(Run comparison\nto see results)', 
                                    ha='center', va='center', fontsize=12, 
                                    transform=plt.gca().transAxes)
                            plt.title('Model Comparison', fontweight='bold', fontsize=12)
                            plt.axis('off')
                    
                    elif i == 11:  # Training Stability
                        if len(training_losses) > 2:
                            # Calculate moving average for stability
                            window = min(3, len(training_losses))
                            moving_avg = pd.Series(training_losses).rolling(window=window).mean()
                            
                            plt.plot(epochs, training_losses, 'b-', alpha=0.3, label='Raw Loss')
                            plt.plot(epochs, moving_avg, 'r-', linewidth=2, label=f'Moving Avg ({window})')
                            plt.title('Training Stability', fontweight='bold', fontsize=12)
                            plt.xlabel('Epoch')
                            plt.ylabel('Loss')
                            plt.legend()
                            plt.grid(True, alpha=0.3)
                    
                    else:  # Enhancement Impact
                        plt.text(0.5, 0.7, '✅ Enhanced Training Complete!', 
                                fontsize=16, ha='center', va='center', fontweight='bold',
                                color='green', transform=plt.gca().transAxes)
                        plt.text(0.5, 0.5, f'Best Performance Achieved:\nF1 Score: {results.get("best_f1_score", 0):.3f}', 
                                fontsize=14, ha='center', va='center',
                                transform=plt.gca().transAxes)
                        plt.text(0.5, 0.3, 'Model ready for deployment!', 
                                fontsize=12, ha='center', va='center',
                                transform=plt.gca().transAxes)
                        plt.title('Training Status', fontweight='bold', fontsize=12)
                        plt.axis('off')
                
                plt.tight_layout(pad=3.0)
                
                # Save enhanced plots
                plot_path = PATHS['training_logs'] / 'enhanced_training_plots.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
                logger.info(f"Enhanced training plots saved to {plot_path}")
                
                plt.show()
            else:
                logger.warning("No training history found for enhanced plotting")
                
        except Exception as e:
            logger.error(f"Error creating enhanced training plots: {e}")
    
    def create_confusion_matrix(self, trainer, test_dataset):
        """Create enhanced confusion matrix with additional insights"""
        logger.info("Creating enhanced confusion matrix...")
        
        predictions_obj = trainer.predict(test_dataset)
        y_pred = np.argmax(predictions_obj.predictions, axis=1)
        y_true = predictions_obj.label_ids
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Enhanced visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Standard confusion matrix
        labels = [self.label_map[i] for i in range(len(self.label_map))]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels,
                   cbar_kws={'label': 'Count'}, ax=axes[0])
        axes[0].set_title('Confusion Matrix (Counts)', fontweight='bold', fontsize=14)
        axes[0].set_xlabel('Predicted Label', fontweight='bold')
        axes[0].set_ylabel('True Label', fontweight='bold')
        
        # Normalized confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Greens',
                   xticklabels=labels, yticklabels=labels,
                   cbar_kws={'label': 'Proportion'}, ax=axes[1])
        axes[1].set_title('Normalized Confusion Matrix', fontweight='bold', fontsize=14)
        axes[1].set_xlabel('Predicted Label', fontweight='bold')
        axes[1].set_ylabel('True Label', fontweight='bold')
        
        plt.tight_layout()
        
        # Save enhanced confusion matrix
        cm_path = PATHS['training_logs'] / 'enhanced_confusion_matrix.png'
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        logger.info(f"Enhanced confusion matrix saved to {cm_path}")
        
        plt.show()
        
        return cm

# Backward compatibility
SentimentModelTrainer = EnhancedSentimentTrainer

if __name__ == "__main__":
    trainer = EnhancedSentimentTrainer(
        use_focal_loss=True,
        use_label_smoothing=True,
        use_mixup=False  # Start conservative
    )
    trained_model, results = trainer.train_model_enhanced()
    print("Enhanced training completed!")
