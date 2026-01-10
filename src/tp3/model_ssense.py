#!/usr/bin/env python3
"""
Super-sense classification model based on contextual embeddings.

This module implements an MLP (Multi-Layer Perceptron) classifier that takes as input
contextual embeddings from pre-trained transformer models (BERT, CamemBERT, etc.)
and predicts super-sense labels for nouns.
"""

import torch
import torch.nn as nn


class SuperSenseClassifier(nn.Module):
    """
    Super-sense classifier based on a simple MLP.
    
    Architecture:
    1. Dense linear layer: embedding_dim -> hidden_dim
    2. ReLU activation
    3. Dropout for regularization
    4. Decision layer: hidden_dim -> num_labels
    
    The model takes as input pre-computed contextual embeddings from
    a transformer and predicts the super-sense label.
    
    Usage example:
        >>> # CamemBERT has embeddings of dimension 768
        >>> model = SuperSenseClassifier(embedding_dim=768, num_labels=25, 
        ...                              hidden_dim=256, dropout=0.3)
        >>> # A batch of 32 word embeddings
        >>> x = torch.randn(32, 768)
        >>> logits = model(x)
        >>> logits.shape
        torch.Size([32, 25])  # Scores for 25 super-sense labels
    """

    def __init__(self, embedding_dim, num_labels, hidden_dim=256, dropout=0.3):
        """
        Initializes the super-sense classifier.
        
        Args:
            embedding_dim (int): Dimension of the contextual embeddings as input.
                                For CamemBERT/BERT-base: 768
                                For CamemBERT-large: 1024
                                For DistilBERT: 768
            num_labels (int): Number of super-sense labels to predict
                             (typically 24 super-senses + 1 for '*')
            hidden_dim (int, optional): Dimension of the hidden layer. Default 256.
            dropout (float, optional): Dropout rate for regularization. Default 0.3.
        """
        super(SuperSenseClassifier, self).__init__()
        
        # Save hyperparameters
        self.embedding_dim = embedding_dim
        self.num_labels = num_labels
        self.hidden_dim = hidden_dim
        
        # First dense layer: transforms the contextual embedding
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        
        # Non-linear activation
        self.relu = nn.ReLU()
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Decision layer: produces scores for each super-sense
        self.decision = nn.Linear(hidden_dim, num_labels)
    
    def forward(self, x):
        """
        Forward pass: predicts super-senses from contextual embeddings.
        
        Args:
            x (torch.FloatTensor): Contextual embeddings of words.
                                   Shape: [batch_size, embedding_dim]
                                   or [num_words, embedding_dim]
        
        Returns:
            torch.FloatTensor: Scores (logits) for each super-sense label.
                              Shape: [batch_size, num_labels]
        
        Example:
            >>> x = torch.randn(32, 768)  # 32 word embeddings
            >>> logits = model.forward(x)
            >>> logits.shape
            torch.Size([32, 25])
            >>> predictions = torch.argmax(logits, dim=-1)
            >>> predictions.shape
            torch.Size([32])
        """
        # Step 1: First dense layer
        x = self.fc1(x)
        
        # Step 2: Non-linear activation
        x = self.relu(x)
        
        # Step 3: Dropout
        x = self.dropout(x)
        
        # Step 4: Decision layer
        logits = self.decision(x)
        
        return logits
