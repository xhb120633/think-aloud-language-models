"""
Neural network models for risky decision-making experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np

class ValueBasedModel(nn.Module):
    model_type = 'NN'
    def __init__(self, config):
        super().__init__()
        self.input_size = config['input_size']*2
        self.hidden_size = config['hidden_size']
        self.n_subjects = config['n_subjects']
        self.sub_embed_dim = config['sub_embed_dim']
        self.sub_embedding = nn.Embedding(self.n_subjects, self.sub_embed_dim)
        self.model_type = config['model_type']
        self.drop_p = config['dropout']
        self.utility_mlp = nn.Sequential(
            nn.Linear(self.input_size + self.sub_embed_dim, self.hidden_size),
            nn.Sigmoid(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Dropout(self.drop_p),
            nn.Sigmoid(),
            nn.Linear(self.hidden_size, 1)
        )

    def forward(self, inputs):
        piA, ViA, piB, ViB, individual_ids, _, _, = inputs
        sub_embed = self.sub_embedding(individual_ids)
        input_A = torch.cat([piA, ViA, sub_embed], dim=1)
        input_B = torch.cat([piB, ViB, sub_embed], dim=1)
        value_A = self.utility_mlp(input_A)
        value_B = self.utility_mlp(input_B)
        return value_B - value_A

    def predict(self, probabilities: np.ndarray, outcomes: np.ndarray, subject_ids: np.ndarray) -> np.ndarray:
        """
        Make predictions using the model.
        
        Args:
            probabilities: Array of probabilities [batch_size, 4]
            outcomes: Array of outcomes [batch_size, 4]
            subject_ids: Array of subject IDs [batch_size]
            
        Returns:
            Array of predictions (0 or 1)
        """
        self.eval()
        with torch.no_grad():
            # Convert inputs to tensors
            probs = torch.FloatTensor(probabilities)
            outs = torch.FloatTensor(outcomes)
            sub_ids = torch.LongTensor(subject_ids)
            
            # Move to same device as model
            device = next(self.parameters()).device
            probs = probs.to(device)
            outs = outs.to(device)
            sub_ids = sub_ids.to(device)
            
            # Get model predictions
            logits = self((probs[:, :2], outs[:, :2], probs[:, 2:], outs[:, 2:], sub_ids, None, None))
            probs = torch.sigmoid(logits)
            
            # Convert to numpy array
            predictions = (probs > 0.5).cpu().numpy().astype(int)
            
        return predictions

    def train_model(
        self,
        train_data: Dict[str, np.ndarray],
        val_data: Optional[Dict[str, np.ndarray]] = None,
        num_epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 1e-4
    ) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_data: Dictionary of training data
            val_data: Optional dictionary of validation data
            num_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            
        Returns:
            Dictionary of training metrics
        """
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.BCEWithLogitsLoss()
        
        # Training metrics
        metrics = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": []
        }
        
        # Training loop
        for epoch in range(num_epochs):
            # Training
            train_loss = 0.0
            num_batches = 0
            
            for i in range(0, len(train_data["probabilities"]), batch_size):
                # Get batch
                batch_probs = torch.FloatTensor(
                    train_data["probabilities"][i:i+batch_size]
                ).to(next(self.parameters()).device)
                
                batch_outs = torch.FloatTensor(
                    train_data["outcomes"][i:i+batch_size]
                ).to(next(self.parameters()).device)
                
                batch_sub_ids = torch.LongTensor(
                    train_data["subject_ids"][i:i+batch_size]
                ).to(next(self.parameters()).device)
                
                batch_labels = torch.FloatTensor(
                    train_data["choices"][i:i+batch_size]
                ).to(next(self.parameters()).device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self((
                    batch_probs[:, :2],
                    batch_outs[:, :2],
                    batch_probs[:, 2:],
                    batch_outs[:, 2:],
                    batch_sub_ids,
                    None,
                    None
                ))
                loss = criterion(outputs.squeeze(), batch_labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                num_batches += 1
            
            metrics["train_loss"].append(train_loss / num_batches)
            
            # Validation
            if val_data is not None:
                val_loss, val_acc = self.evaluate(val_data, batch_size)
                metrics["val_loss"].append(val_loss)
                metrics["val_accuracy"].append(val_acc)
            
            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {metrics['train_loss'][-1]:.4f}")
            if val_data is not None:
                print(f"Val Loss: {metrics['val_loss'][-1]:.4f}")
                print(f"Val Accuracy: {metrics['val_accuracy'][-1]:.4f}")
        
        return metrics
    
    def evaluate(
        self,
        data: Dict[str, np.ndarray],
        batch_size: int = 32
    ) -> Tuple[float, float]:
        """
        Evaluate the model.
        
        Args:
            data: Dictionary of evaluation data
            batch_size: Batch size
            
        Returns:
            Tuple of (loss, accuracy)
        """
        self.eval()
        criterion = nn.BCEWithLogitsLoss()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for i in range(0, len(data["probabilities"]), batch_size):
                # Get batch
                batch_probs = torch.FloatTensor(
                    data["probabilities"][i:i+batch_size]
                ).to(next(self.parameters()).device)
                
                batch_outs = torch.FloatTensor(
                    data["outcomes"][i:i+batch_size]
                ).to(next(self.parameters()).device)
                
                batch_sub_ids = torch.LongTensor(
                    data["subject_ids"][i:i+batch_size]
                ).to(next(self.parameters()).device)
                
                batch_labels = torch.FloatTensor(
                    data["choices"][i:i+batch_size]
                ).to(next(self.parameters()).device)
                
                # Forward pass
                outputs = self((
                    batch_probs[:, :2],
                    batch_outs[:, :2],
                    batch_probs[:, 2:],
                    batch_outs[:, 2:],
                    batch_sub_ids,
                    None,
                    None
                ))
                loss = criterion(outputs.squeeze(), batch_labels)
                
                # Calculate accuracy
                predictions = (torch.sigmoid(outputs) > 0.5).squeeze()
                correct += (predictions == batch_labels).sum().item()
                total += batch_labels.size(0)
                
                total_loss += loss.item()
        
        return total_loss / (len(data["probabilities"]) / batch_size), correct / total

class ContextDependentModel(nn.Module):
    model_type = 'NN'
    def __init__(self, config):
        super().__init__()
        self.input_size = config['input_size']*4  # default input_size is the maximum length of offers in an option
        self.hidden_size = config['hidden_size']
        self.n_subjects = config['n_subjects']
        self.sub_embed_dim = config['sub_embed_dim']
        self.model_type = config['model_type']
        self.drop_p = config['dropout']
        self.sub_embedding = nn.Embedding(self.n_subjects, self.sub_embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.input_size + self.sub_embed_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Dropout(self.drop_p),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 2)
        )
        
    def forward(self, inputs):
        piA, ViA, piB, ViB, individual_ids, _, _, = inputs
        sub_embed = self.sub_embedding(individual_ids)
        inputs = torch.cat([piA, ViA, piB, ViB, sub_embed], dim=1)
        pB = self.mlp(inputs)
        return pB

    def predict(self, probabilities: np.ndarray, outcomes: np.ndarray, subject_ids: np.ndarray) -> np.ndarray:
        """
        Make predictions using the model.
        
        Args:
            probabilities: Array of probabilities [batch_size, 4]
            outcomes: Array of outcomes [batch_size, 4]
            subject_ids: Array of subject IDs [batch_size]
            
        Returns:
            Array of predictions (0 or 1)
        """
        self.eval()
        with torch.no_grad():
            # Convert inputs to tensors
            probs = torch.FloatTensor(probabilities)
            outs = torch.FloatTensor(outcomes)
            sub_ids = torch.LongTensor(subject_ids)
            
            # Move to same device as model
            device = next(self.parameters()).device
            probs = probs.to(device)
            outs = outs.to(device)
            sub_ids = sub_ids.to(device)
            
            # Get model predictions
            logits = self((probs[:, :2], outs[:, :2], probs[:, 2:], outs[:, 2:], sub_ids, None, None))
            probs = torch.sigmoid(logits)
            
            # Convert to numpy array
            predictions = (probs > 0.5).cpu().numpy().astype(int)
            
        return predictions

    def train_model(
        self,
        train_data: Dict[str, np.ndarray],
        val_data: Optional[Dict[str, np.ndarray]] = None,
        num_epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 1e-4
    ) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_data: Dictionary of training data
            val_data: Optional dictionary of validation data
            num_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            
        Returns:
            Dictionary of training metrics
        """
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.BCEWithLogitsLoss()
        
        # Training metrics
        metrics = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": []
        }
        
        # Training loop
        for epoch in range(num_epochs):
            # Training
            train_loss = 0.0
            num_batches = 0
            
            for i in range(0, len(train_data["probabilities"]), batch_size):
                # Get batch
                batch_probs = torch.FloatTensor(
                    train_data["probabilities"][i:i+batch_size]
                ).to(next(self.parameters()).device)
                
                batch_outs = torch.FloatTensor(
                    train_data["outcomes"][i:i+batch_size]
                ).to(next(self.parameters()).device)
                
                batch_sub_ids = torch.LongTensor(
                    train_data["subject_ids"][i:i+batch_size]
                ).to(next(self.parameters()).device)
                
                batch_labels = torch.FloatTensor(
                    train_data["choices"][i:i+batch_size]
                ).to(next(self.parameters()).device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self((
                    batch_probs[:, :2],
                    batch_outs[:, :2],
                    batch_probs[:, 2:],
                    batch_outs[:, 2:],
                    batch_sub_ids,
                    None,
                    None
                ))
                loss = criterion(outputs.squeeze(), batch_labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                num_batches += 1
            
            metrics["train_loss"].append(train_loss / num_batches)
            
            # Validation
            if val_data is not None:
                val_loss, val_acc = self.evaluate(val_data, batch_size)
                metrics["val_loss"].append(val_loss)
                metrics["val_accuracy"].append(val_acc)
            
            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {metrics['train_loss'][-1]:.4f}")
            if val_data is not None:
                print(f"Val Loss: {metrics['val_loss'][-1]:.4f}")
                print(f"Val Accuracy: {metrics['val_accuracy'][-1]:.4f}")
        
        return metrics
    
    def evaluate(
        self,
        data: Dict[str, np.ndarray],
        batch_size: int = 32
    ) -> Tuple[float, float]:
        """
        Evaluate the model.
        
        Args:
            data: Dictionary of evaluation data
            batch_size: Batch size
            
        Returns:
            Tuple of (loss, accuracy)
        """
        self.eval()
        criterion = nn.BCEWithLogitsLoss()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for i in range(0, len(data["probabilities"]), batch_size):
                # Get batch
                batch_probs = torch.FloatTensor(
                    data["probabilities"][i:i+batch_size]
                ).to(next(self.parameters()).device)
                
                batch_outs = torch.FloatTensor(
                    data["outcomes"][i:i+batch_size]
                ).to(next(self.parameters()).device)
                
                batch_sub_ids = torch.LongTensor(
                    data["subject_ids"][i:i+batch_size]
                ).to(next(self.parameters()).device)
                
                batch_labels = torch.FloatTensor(
                    data["choices"][i:i+batch_size]
                ).to(next(self.parameters()).device)
                
                # Forward pass
                outputs = self((
                    batch_probs[:, :2],
                    batch_outs[:, :2],
                    batch_probs[:, 2:],
                    batch_outs[:, 2:],
                    batch_sub_ids,
                    None,
                    None
                ))
                loss = criterion(outputs.squeeze(), batch_labels)
                
                # Calculate accuracy
                predictions = (torch.sigmoid(outputs) > 0.5).squeeze()
                correct += (predictions == batch_labels).sum().item()
                total += batch_labels.size(0)
                
                total_loss += loss.item()
        
        return total_loss / (len(data["probabilities"]) / batch_size), correct / total 