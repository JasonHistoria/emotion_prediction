# simple_trainer.py - Alternative training approach without Trainer class

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report
import os

class SimplePoetryTrainer:
    """Simplified trainer that doesn't use HuggingFace Trainer class"""
    
    def __init__(self, model_name: str = 'bert-base-chinese', emotion_names: list = None):
        self.model_name = model_name
        self.emotion_names = emotion_names or ['哀伤', '思念', '怨恨', '喜悦']
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
    def load_data(self, csv_path: str):
        """Load and prepare training data"""
        df = pd.read_csv(csv_path, encoding='utf-8')
        
        texts = df['content'].tolist()
        labels = []
        
        for _, row in df.iterrows():
            label = []
            for emotion in self.emotion_names:
                label.append(int(row[emotion]) if emotion in row else 0)
            labels.append(label)
        
        print(f"Loaded {len(texts)} poems with {len(self.emotion_names)} emotion categories")
        return texts, labels
    
    def prepare_datasets(self, texts, labels, test_size=0.2, val_size=0.1):
        """Split data into train/val/test sets"""
        from sklearn.model_selection import train_test_split
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=42
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42
        )
        
        # Create datasets
        train_dataset = PoetryDataset(X_train, y_train, self.tokenizer)
        val_dataset = PoetryDataset(X_val, y_val, self.tokenizer)
        test_dataset = PoetryDataset(X_test, y_test, self.tokenizer)
        
        print(f"Dataset splits - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        return train_dataset, val_dataset, test_dataset
    
    def train_model(self, train_dataset, val_dataset, output_dir='./trained_model',
                   epochs=5, batch_size=8, learning_rate=2e-5):
        """Train the model using simple PyTorch training loop"""
        
        # Initialize model
        self.model = PoetryEmotionClassifier(
            model_name=self.model_name, 
            num_labels=len(self.emotion_names)
        ).to(self.device)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Optimizer
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        
        # Loss function
        criterion = nn.BCEWithLogitsLoss()
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Training phase
            self.model.train()
            train_loss = 0
            train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
            
            for batch in train_pbar:
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs['logits'], labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_pbar.set_postfix({'loss': loss.item()})
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            val_pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}")
            
            with torch.no_grad():
                for batch in val_pbar:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(outputs['logits'], labels)
                    
                    val_loss += loss.item()
                    val_pbar.set_postfix({'loss': loss.item()})
            
            avg_val_loss = val_loss / len(val_loader)
            
            print(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.save_model(output_dir)
                print(f"✅ Best model saved with validation loss: {best_val_loss:.4f}")
        
        print(f"Training completed! Best model saved to {output_dir}")
        return self.model
    
    def save_model(self, output_dir):
        """Save model and tokenizer"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model state dict
        torch.save(self.model.state_dict(), os.path.join(output_dir, 'model.pt'))
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        # Save config
        config = {
            'model_name': self.model_name,
            'emotion_names': self.emotion_names,
            'num_labels': len(self.emotion_names)
        }
        
        import json
        with open(os.path.join(output_dir, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    
    def load_model(self, model_dir):
        """Load saved model"""
        import json
        
        # Load config
        with open(os.path.join(model_dir, 'config.json'), 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Initialize model
        self.model = PoetryEmotionClassifier(
            model_name=config['model_name'],
            num_labels=config['num_labels']
        ).to(self.device)
        
        # Load state dict
        self.model.load_state_dict(torch.load(
            os.path.join(model_dir, 'model.pt'), 
            map_location=self.device
        ))
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        print(f"Model loaded from {model_dir}")
        return self.model
    
    def evaluate_model(self, test_dataset):
        """Evaluate model performance"""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        all_predictions = []
        all_labels = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels']
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.sigmoid(outputs['logits'])
                predictions = (predictions > 0.5).int()
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        # Calculate metrics
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        # Per-emotion metrics
        for i, emotion in enumerate(self.emotion_names):
            y_true = all_labels[:, i]
            y_pred = all_predictions[:, i]
            
            print(f"\n{emotion} Classification Report:")
            print(classification_report(y_true, y_pred))
        
        return all_predictions, all_labels

# Add this to your poetry_emotion_classifier.py file, or create a new import
from simple_trainer import SimplePoetryTrainer