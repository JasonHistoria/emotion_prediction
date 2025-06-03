"""
Chinese Poetry Emotion Classification System
Updated implementation with flexible multi-label support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForMaskedLM,
    DataCollatorForLanguageModeling, TrainingArguments, Trainer
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set Chinese font for matplotlib
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class UnlabeledPoetryDataset(Dataset):
    """Dataset for unsupervised pretraining on unlabeled poems"""
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        # Tokenize with padding and truncation
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }


class LabeledPoetryDataset(Dataset):
    """Dataset for supervised fine-tuning on labeled poems"""
    
    def __init__(self, texts: List[str], labels: Optional[List[List[int]]] = None, 
                 tokenizer=None, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }
        
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        
        return item


class PoetryEmotionClassifier(nn.Module):
    """Enhanced multi-label emotion classifier with attention visualization"""
    
    def __init__(self, model_name: str = 'bert-base-chinese', 
                 num_labels: int = 13, dropout_rate: float = 0.3):
        super().__init__()
        
        self.num_labels = num_labels
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Attention pooling layer
        self.attention_weights = nn.Linear(self.bert.config.hidden_size, 1)
        
        # Multi-label classification head
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize classifier weights"""
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        nn.init.xavier_uniform_(self.attention_weights.weight)
        nn.init.zeros_(self.attention_weights.bias)
    
    def forward(self, input_ids, attention_mask, labels=None, return_attention=True):
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Use last hidden state
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Compute attention scores
        attention_scores = self.attention_weights(sequence_output)  # [batch_size, seq_len, 1]
        attention_scores = attention_scores.squeeze(-1)  # [batch_size, seq_len]
        
        # Mask padding tokens
        attention_scores = attention_scores.masked_fill(
            attention_mask == 0, -float('inf')
        )
        
        # Apply softmax to get attention weights
        attention_probs = F.softmax(attention_scores, dim=-1)  # [batch_size, seq_len]
        
        # Weighted pooling
        weighted_output = torch.bmm(
            attention_probs.unsqueeze(1),  # [batch_size, 1, seq_len]
            sequence_output  # [batch_size, seq_len, hidden_size]
        ).squeeze(1)  # [batch_size, hidden_size]
        
        # Apply dropout and classification
        pooled_output = self.dropout(weighted_output)
        logits = self.classifier(pooled_output)
        
        output_dict = {'logits': logits}
        
        if return_attention:
            output_dict['attention_weights'] = attention_probs
        
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            output_dict['loss'] = loss
        
        return output_dict


class OptimizedPoetryTrainer:
    """Optimized trainer with automatic emotion detection"""
    
    def __init__(self, base_model_name: str = 'bert-base-chinese'):
        self.base_model_name = base_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.emotion_names = []  # Will be automatically detected
        print(f"Using device: {self.device}")
    
    def detect_emotions_from_csv(self, csv_path: str) -> List[str]:
        """Automatically detect emotion columns from CSV"""
        df = pd.read_csv(csv_path, encoding='utf-8')
        
        # Common non-emotion columns to exclude
        exclude_columns = ['content', 'poem', 'text', 'title', 'author', 'dynasty', 
                          'id', 'index', 'label', 'category', 'source', 'year']
        
        # Find emotion columns (numeric columns that aren't excluded)
        emotion_columns = []
        for col in df.columns:
            if col.lower() not in [x.lower() for x in exclude_columns]:
                # Check if column contains binary values (0/1) or numeric values
                try:
                    unique_vals = df[col].dropna().unique()
                    if len(unique_vals) <= 10 and all(isinstance(x, (int, float)) or str(x).isdigit() for x in unique_vals):
                        emotion_columns.append(col)
                except:
                    continue
        
        print(f"Detected emotion columns: {emotion_columns}")
        return emotion_columns
    
    def pretrain_on_unlabeled_data(self, unlabeled_texts: List[str], 
                                   output_dir: str = './pretrained_poetry_model',
                                   num_epochs: int = 3, batch_size: int = 32):
        """
        Pretrain BERT on unlabeled poetry using Masked Language Modeling
        """
        print("Starting unsupervised pretraining...")
        
        # Create MLM model
        model = AutoModelForMaskedLM.from_pretrained(self.base_model_name)
        model.to(self.device)
        
        # Create dataset
        dataset = UnlabeledPoetryDataset(unlabeled_texts, self.tokenizer)
        
        # Data collator for MLM
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=0.15
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            save_steps=500,
            save_total_limit=2,
            prediction_loss_only=True,
            logging_steps=100,
            logging_dir=f'{output_dir}/logs',
            warmup_steps=500,
            weight_decay=0.01,
            fp16=torch.cuda.is_available(),
            report_to=[],  # Disable wandb/tensorboard
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset,
        )
        
        # Train
        trainer.train()
        
        # Save the pretrained model
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"Pretraining completed. Model saved to {output_dir}")
        return output_dir
    
    def prepare_labeled_data(self, csv_path: str) -> Tuple[List[str], List[List[int]], List[str]]:
        """Load and prepare labeled training data with automatic emotion detection"""
        df = pd.read_csv(csv_path, encoding='utf-8')
        
        # Automatically detect emotion columns
        self.emotion_names = self.detect_emotions_from_csv(csv_path)
        
        # Find the text column
        text_columns = ['content', 'poem', 'text']
        text_column = None
        for col in text_columns:
            if col in df.columns:
                text_column = col
                break
        
        if text_column is None:
            raise ValueError(f"Could not find text column. Available columns: {list(df.columns)}")
        
        texts = df[text_column].astype(str).tolist()
        labels = []
        
        # Build labels for each poem
        for _, row in df.iterrows():
            label = []
            for emotion in self.emotion_names:
                if emotion in row:
                    # Convert to binary (handle various formats)
                    val = row[emotion]
                    if pd.isna(val):
                        label.append(0)
                    elif isinstance(val, str):
                        label.append(1 if val.lower() in ['true', '1', 'yes', 'y'] else 0)
                    else:
                        label.append(int(bool(float(val))))
                else:
                    label.append(0)
            labels.append(label)
        
        print(f"Loaded {len(texts)} labeled poems with {len(self.emotion_names)} emotions")
        print(f"Emotion names: {self.emotion_names}")
        
        # Print distribution
        labels_array = np.array(labels)
        for i, emotion in enumerate(self.emotion_names):
            count = labels_array[:, i].sum()
            percentage = (count / len(labels)) * 100
            print(f"  {emotion}: {count} poems ({percentage:.1f}%)")
        
        return texts, labels, self.emotion_names
    
    def fine_tune_model(self, pretrained_model_path: str, texts: List[str], 
                       labels: List[List[int]], emotion_names: List[str],
                       output_dir: str = './final_model',
                       num_epochs: int = 10, batch_size: int = 16, learning_rate: float = 2e-5):
        """
        Fine-tune the pretrained model on labeled data
        """
        print("Starting fine-tuning on labeled data...")
        
        # Update emotion names
        self.emotion_names = emotion_names
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            texts, labels, test_size=0.3, random_state=42, stratify=None
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )
        
        # Create datasets
        train_dataset = LabeledPoetryDataset(X_train, y_train, self.tokenizer)
        val_dataset = LabeledPoetryDataset(X_val, y_val, self.tokenizer)
        test_dataset = LabeledPoetryDataset(X_test, y_test, self.tokenizer)
        
        # Initialize model with pretrained weights
        model = PoetryEmotionClassifier(
            model_name=pretrained_model_path,
            num_labels=len(self.emotion_names)
        ).to(self.device)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        num_training_steps = len(train_loader) * num_epochs
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.1, total_iters=num_training_steps
        )
        
        # Training loop
        best_val_f1 = 0
        best_model_state = None
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")
            
            for batch in train_pbar:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = model(input_ids, attention_mask, labels)
                loss = outputs['loss']
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                train_pbar.set_postfix({'loss': loss.item()})
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation phase
            model.eval()
            val_loss = 0
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validating"):
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    outputs = model(input_ids, attention_mask, labels)
                    val_loss += outputs['loss'].item()
                    
                    preds = torch.sigmoid(outputs['logits']) > 0.5
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            avg_val_loss = val_loss / len(val_loader)
            
            # Calculate F1 score
            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)
            val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
            
            print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, "
                  f"Val Loss={avg_val_loss:.4f}, Val F1={val_f1:.4f}")
            
            # Save best model
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_state = model.state_dict()
                print(f"New best model with F1={val_f1:.4f}")
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        # Test evaluation
        print("\nEvaluating on test set...")
        test_results = self.evaluate_model(model, test_loader)
        
        # Save final model
        os.makedirs(output_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(output_dir, 'model.pt'))
        self.tokenizer.save_pretrained(output_dir)
        
        # Save config
        config = {
            'model_name': pretrained_model_path,
            'emotion_names': self.emotion_names,
            'num_labels': len(self.emotion_names),
            'best_val_f1': best_val_f1,
            'test_results': test_results
        }
        
        with open(os.path.join(output_dir, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        print(f"Model saved to {output_dir}")
        return model, test_results
    
    def evaluate_model(self, model, test_loader):
        """Evaluate model performance"""
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels']
                
                outputs = model(input_ids, attention_mask)
                preds = torch.sigmoid(outputs['logits']) > 0.5
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Calculate metrics for each emotion
        results = {}
        for i, emotion in enumerate(self.emotion_names):
            y_true = all_labels[:, i]
            y_pred = all_preds[:, i]
            
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            
            # Handle case where there might be no positive predictions
            if '1' in report:
                results[emotion] = {
                    'precision': report['1']['precision'],
                    'recall': report['1']['recall'],
                    'f1-score': report['1']['f1-score'],
                    'support': report['1']['support']
                }
            else:
                # If no positive predictions, use weighted average or zeros
                results[emotion] = {
                    'precision': report.get('weighted avg', {}).get('precision', 0.0),
                    'recall': report.get('weighted avg', {}).get('recall', 0.0),
                    'f1-score': report.get('weighted avg', {}).get('f1-score', 0.0),
                    'support': int(np.sum(y_true))
                }
            
            print(f"\n{emotion}:")
            print(classification_report(y_true, y_pred, zero_division=0))
        
        # Overall metrics
        overall_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        results['overall_f1'] = overall_f1
        
        return results


class PoetryEmotionAnalyzer:
    """Analyzer for inference with attention visualization"""
    
    def __init__(self, model_dir: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load config
        with open(os.path.join(model_dir, 'config.json'), 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.emotion_names = self.config['emotion_names']
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        # Load model
        self.model = PoetryEmotionClassifier(
            model_name=self.config['model_name'],
            num_labels=self.config['num_labels']
        ).to(self.device)
        
        self.model.load_state_dict(
            torch.load(os.path.join(model_dir, 'model.pt'), map_location=self.device)
        )
        self.model.eval()
    
    def analyze_poem(self, poem_text: str) -> Dict:
        """Analyze a poem and return emotions with attention weights"""
        
        # Tokenize
        inputs = self.tokenizer(
            poem_text,
            return_tensors='pt',
            truncation=True,
            max_length=128,
            padding=True
        )
        
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask, return_attention=True)
            
            # Get emotion probabilities
            probs = torch.sigmoid(outputs['logits']).cpu().numpy()[0]
            
            # Get attention weights
            attention_weights = outputs['attention_weights'].cpu().numpy()[0]
            
            # Get tokens
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].cpu())
        
        # Map tokens back to original characters
        char_attention_map = self._map_tokens_to_chars(poem_text, tokens, attention_weights)
        
        # Create character-attention pairs
        char_attention_pairs = []
        for char, weight in char_attention_map.items():
            if char.strip():  # Skip whitespace
                char_attention_pairs.append((char, float(weight)))
        
        # Sort by attention weight
        char_attention_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Create results
        results = {
            'poem_text': poem_text,
            'emotions': {emotion: float(prob) for emotion, prob in zip(self.emotion_names, probs)},
            'predicted_emotions': [emotion for emotion, prob in zip(self.emotion_names, probs) if prob > 0.5],
            'token_importance': char_attention_pairs,  # Now character-level
            'attention_weights': attention_weights.tolist(),
            'tokens': tokens,
            'char_attention_map': char_attention_map
        }
        
        return results
    
    def _map_tokens_to_chars(self, original_text: str, tokens: List[str], attention_weights: np.ndarray) -> Dict[str, float]:
        """Map subword tokens back to original characters with their attention weights"""
        
        # Remove punctuation for character mapping
        text_chars = list(original_text.replace('，', '').replace('。', '').replace('！', '').replace('？', '').replace('、', ''))
        
        # Initialize character attention map
        char_attention = {}
        char_count = {}
        
        # Track position in original text
        char_idx = 0
        
        for token, weight in zip(tokens, attention_weights):
            if token in ['[CLS]', '[SEP]', '[PAD]', '[UNK]']:
                continue
                
            # Handle subword tokens (## prefix)
            if token.startswith('##'):
                token = token[2:]
            
            # For each character in the token
            for char in token:
                if char_idx < len(text_chars):
                    original_char = text_chars[char_idx]
                    
                    # Aggregate attention weights for the same character
                    if original_char not in char_attention:
                        char_attention[original_char] = 0
                        char_count[original_char] = 0
                    
                    char_attention[original_char] += weight
                    char_count[original_char] += 1
                    char_idx += 1
        
        # Average the attention weights for characters that appear multiple times
        for char in char_attention:
            if char_count[char] > 0:
                char_attention[char] /= char_count[char]
        
        return char_attention
    
    def visualize_attention(self, analysis_results: Dict, save_path: str = None):
        """Create attention visualization for characters"""
        
        # Get character-level attention
        char_importance = analysis_results['token_importance']
        poem_text = analysis_results['poem_text']
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Subplot 1: Character-level attention heatmap
        # Create a visual representation of the poem with attention
        chars = list(poem_text)
        char_attention_map = analysis_results.get('char_attention_map', {})
        
        # Create attention values for each character (including punctuation)
        attention_values = []
        for char in chars:
            if char in char_attention_map:
                attention_values.append(char_attention_map[char])
            elif char in '，。！？、':
                attention_values.append(0)  # No attention for punctuation
            else:
                attention_values.append(0.1)  # Default low attention
        
        # Create color map
        colors = plt.cm.Reds(np.array(attention_values))
        
        # Plot characters with background colors
        x_pos = 0
        y_pos = 0
        char_width = 1
        char_height = 1
        
        for i, (char, color) in enumerate(zip(chars, colors)):
            # Handle line breaks at punctuation
            if char in '。！？':
                ax1.text(x_pos + 0.5, y_pos + 0.5, char, ha='center', va='center', 
                        fontsize=20, fontweight='bold')
                ax1.add_patch(plt.Rectangle((x_pos, y_pos), char_width, char_height, 
                                          facecolor=color, edgecolor='black', linewidth=0.5))
                x_pos = 0
                y_pos -= char_height * 1.2
            else:
                ax1.text(x_pos + 0.5, y_pos + 0.5, char, ha='center', va='center', 
                        fontsize=20, fontweight='bold')
                ax1.add_patch(plt.Rectangle((x_pos, y_pos), char_width, char_height, 
                                          facecolor=color, edgecolor='black', linewidth=0.5))
                x_pos += char_width
        
        ax1.set_xlim(-0.5, 20)
        ax1.set_ylim(y_pos - 1, 2)
        ax1.axis('off')
        ax1.set_title('诗歌字符注意力热力图', fontsize=16, fontweight='bold', pad=20)
        
        # Add color bar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(0, max(attention_values) if attention_values else 1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax1, orientation='horizontal', pad=0.1, shrink=0.6)
        cbar.set_label('注意力权重', fontsize=12)
        
        # Subplot 2: Top important characters bar chart
        top_chars = char_importance[:15]  # Top 15 characters
        if top_chars:
            chars_list = [item[0] for item in top_chars]
            importance_list = [item[1] for item in top_chars]
            
            bars = ax2.bar(range(len(chars_list)), importance_list, color='skyblue')
            
            # Highlight top 5
            for i in range(min(5, len(bars))):
                bars[i].set_color('coral')
            
            ax2.set_xlabel('字符', fontsize=12)
            ax2.set_ylabel('重要性分数', fontsize=12)
            ax2.set_title('最重要的字符', fontsize=14, fontweight='bold')
            ax2.set_xticks(range(len(chars_list)))
            ax2.set_xticklabels(chars_list, fontsize=14)
            
            # Add value labels on bars
            for i, (char, score) in enumerate(zip(chars_list, importance_list)):
                ax2.text(i, score + 0.01, f'{score:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()


def train_complete_system(unlabeled_path: str, labeled_path: str, output_dir: str = './models'):
    """Complete training pipeline"""
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    pretrained_dir = os.path.join(output_dir, 'pretrained')
    final_model_dir = os.path.join(output_dir, 'final')
    
    # Initialize trainer
    trainer = OptimizedPoetryTrainer()
    
    # Load labeled data first to detect emotions
    texts, labels, emotion_names = trainer.prepare_labeled_data(labeled_path)
    
    # Load unlabeled data if available
    if os.path.exists(unlabeled_path):
        print("Loading unlabeled data...")
        with open(unlabeled_path, 'r', encoding='utf-8') as f:
            unlabeled_texts = [line.strip() for line in f if line.strip()]
        
        # Limit to manageable size for pretraining
        if len(unlabeled_texts) > 100000:
            unlabeled_texts = unlabeled_texts[:100000]
        
        print(f"Loaded {len(unlabeled_texts)} unlabeled poems")
        
        # Step 1: Pretrain on unlabeled data
        pretrained_model_path = trainer.pretrain_on_unlabeled_data(
            unlabeled_texts, 
            output_dir=pretrained_dir,
            num_epochs=3,
            batch_size=32
        )
    else:
        print("No unlabeled data found, skipping pretraining...")
        pretrained_model_path = trainer.base_model_name
    
    # Step 2: Fine-tune on labeled data
    model, test_results = trainer.fine_tune_model(
        pretrained_model_path,
        texts,
        labels,
        emotion_names,
        output_dir=final_model_dir,
        num_epochs=5,
        batch_size=16
    )
    
    print("\nTraining completed!")
    print(f"Overall test F1 score: {test_results['overall_f1']:.4f}")
    
    return final_model_dir


if __name__ == "__main__":
    # Example usage
    print("Chinese Poetry Emotion Classification System")
    print("=" * 50)
    
    # Paths
    unlabeled_path = './data/unlabeled_poems.txt'
    labeled_path = './data/poem_emotions_multilabel.csv'  # Updated for multilabel
    
    # Train the complete system
    final_model_dir = train_complete_system(unlabeled_path, labeled_path)
    
    # Test the analyzer
    print("\nTesting the analyzer...")
    analyzer = PoetryEmotionAnalyzer(final_model_dir)
    
    # Example poem
    test_poem = "独在异乡为异客，每逢佳节倍思亲"
    results = analyzer.analyze_poem(test_poem)
    
    print(f"\nAnalysis results for: {test_poem}")
    print(f"Emotions: {results['emotions']}")
    print(f"Predicted emotions: {results['predicted_emotions']}")
    print(f"Top 5 important tokens: {results['token_importance'][:5]}")