#!/usr/bin/env python3
"""
Super-sense classification model based on contextual embeddings.

This module implements an MLP (Multi-Layer Perceptron) classifier that takes as input
contextual embeddings from pre-trained transformer models (BERT, CamemBERT, etc.)
and predicts super-sense labels for nouns.
"""

import torch.nn as nn

class SuperSenseClassifier(nn.Module):
    """
    Super-sense classifier based on a simple MLP.
    """

    def __init__(self, embedding_dim, num_labels, dropout=0.3):
        """
        Initializes the super-sense classifier.
        """
        super(SuperSenseClassifier, self).__init__()
        
        self.fc1 = nn.Linear(embedding_dim, embedding_dim * 2)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.decision = nn.Linear(embedding_dim * 2, num_labels)
    
    def forward(self, x):
        """
        Forward pass: predicts super-senses from contextual embeddings.
        """
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        logits = self.decision(x)
        
        return logits
