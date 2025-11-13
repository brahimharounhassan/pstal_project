#!/usr/bin/env python3
import sys, torch, collections, tqdm, pdb
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

"""
Sentence-level text classification using PyTorch.

This script trains a sentence classifier (Bag-of-Words, GRU, or CNN)
on a labeled French corpus. It handles data reading, vocabulary creation,
padding, batching, training, and model saving.

Usage:
    ./train_classifier.py train.txt dev.txt bow|gru|cnn word|char
"""

################################################################################

class CNNClassifier(nn.Module):
  """
  Sentence classifier based on a 1D Convolutional Neural Network (CNN).
  Represents tokens with learned embeddings, applies convolution over the 
  sequence, and performs max pooling to extract features. The resulting vector 
  is passed through a linear layer to predict the class.
  """

  def __init__(self, d_embed, d_hidden, d_in, d_out):
    """
    Initialises the model.
    
    ## Parameters    
    d_embed : int --Size of the word embeddings (embedding dimension).
    d_hidden : int -- Number of convolutional filters (hidden dimension).
    d_in : int -- Vocabulary size (number of distinct token IDs).
    d_out : int -- Number of output classes.
    """
    super().__init__() 
    self.embed = nn.Embedding(d_in, d_embed, padding_idx=0)
    self.cnn = nn.Conv1d(d_embed, d_hidden, kernel_size=5)    
    self.dropout = nn.Dropout(0.1)
    self.nonlin = torch.nn.ReLU()
    self.decision = nn.Linear(d_hidden, d_out)      
    
  def forward(self, idx_words):
    """
    Computes the forward pass of the model.
    
    ## Parameters
    idx_words : torch.LongTensor of shape (batch_size, seq_len)
    Tensor containing token indices for each sentence in the batch.

    ## Returns : torch.Tensor of shape (batch_size, d_out)
    Unnormalized class scores (logits) for each sentence in the batch.
    """
    emb = self.embed(idx_words)  
    conv = self.nonlin(self.cnn(emb.transpose(2,1)))
    hidden = nn.functional.max_pool1d(conv, conv.shape[-1])
    return self.decision(self.dropout(hidden.squeeze()))   

################################################################################

class GRUClassifier(nn.Module):
  """
  Sentence classifier based on a Gated Recurrent Unit (GRU). Encodes sequence of 
  word embeddings using a GRU. Last hidden state summarizes the sentence and is 
  given to linear layer for classification.
  """
  
  def __init__(self, d_embed, d_hidden, d_in, d_out):
    """
    Initialises the model.
    
    ## Parameters    
    d_embed : int --Size of the word embeddings (embedding dimension).
    d_hidden : int -- Dimension of the GRU hidden state.
    d_in : int -- Vocabulary size (number of distinct token IDs).
    d_out : int -- Number of output classes.
    """
    super().__init__() 
    self.embed = nn.Embedding(d_in, d_embed, padding_idx=0)
    self.gru = nn.GRU(d_embed, d_hidden, batch_first=True, bias=False)    
    self.dropout = nn.Dropout(0.1)
    self.decision = nn.Linear(d_hidden, d_out)      
    
  def forward(self, idx_words):
    """
    Computes the forward pass of the model.
    
    ## Parameters
    idx_words : torch.LongTensor of shape (batch_size, seq_len)
    Tensor containing token indices for each sentence in the batch.

    ## Returns : torch.Tensor of shape (batch_size, d_out)
    Unnormalized class scores (logits) for each sentence in the batch.
    """
    embedded = self.embed(idx_words)    
    hidden = self.gru(embedded)[1].squeeze(dim=0)    
    return self.decision(self.dropout(hidden))    

################################################################################

class BOWClassifier(nn.Module):
  """
  Bag-of-words (BOW) sentence classifier. Tokens are represented by embeddings,
  sentence is represented by the average of all its token's embeddings (order 
  ignored), and a linear layer over the average vector predicts the class label.
  """
  
  def __init__(self, d_embed, d_in, d_out):
    """
    Initialises the model.
    
    ## Parameters    
    d_embed : int --Size of the word embeddings (embedding dimension).
    d_in : int -- Vocabulary size (number of distinct token IDs).
    d_out : int -- Number of output classes.
    """
    super().__init__() 
    self.embed = nn.Embedding(d_in, d_embed, padding_idx=0)
    self.dropout = nn.Dropout(0.3)
    self.decision = nn.Linear(d_embed, d_out)      
    
  def forward(self, idx_words):
    """
    Computes the forward pass of the model.
    
    ## Parameters
    idx_words : torch.LongTensor of shape (batch_size, seq_len)
    Tensor containing token indices for each sentence in the batch.

    ## Returns : torch.Tensor of shape (batch_size, d_out)
    Unnormalized class scores (logits) for each sentence in the batch.
    """
    embedded = self.embed(idx_words)
    averaged = torch.mean(embedded, dim=1) # dim 0 is batch    
    return self.decision(self.dropout(averaged))

################################################################################

def perf(model, dev_loader, criterion):
  """
  Calculate model loss and accuracy on the dev set. This function should be 
  called within `fit`, at the end of each epoch, to obtain the learning curve.

  ## Parameters
  model : nn.Module -- partly trained model (instance of one of 3 classes above)
  dev_loader, DataLoader -- provides batches of (X, y) pairs  
  criterion -- loss function (e.g. CrossEntropyLoss)

  ## Returns : (float, float) tuple of 2 elements  
  (loss, accuracy) Average loss and accuracy over the dev set
  """
  model.eval() # Turn off gradient calculation and dropout
  total_loss = correct = 0
  for (X, y) in dev_loader:
    with torch.no_grad():
      y_scores = model(X) 
      total_loss += criterion(y_scores, y)
      y_pred = torch.max(y_scores, dim=1)[1] # argmax
      correct += torch.sum(y_pred.data == y)
  total = len(dev_loader.dataset)
  return total_loss / total, correct / total

################################################################################
    
def fit(model, train_loader, dev_loader, epochs):
  """
  Train a text classification model for several epochs.
  
  ## Parameters
  model : nn.Module -- randomly initialized model to train  
  train_loader : DataLoader -- training batches
  dev_loader : DataLoader -- development batches for learning curve evaluation
  epochs : int -- number of training epochs  

  Prints training loss, dev loss, and dev accuracy after each epoch.
  Does not return anything: model passed as argument will be trained after call
  """
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters()) 
  for epoch in range(epochs):
    model.train() # Turn on gradient calculation and dropout
    total_loss = 0
    for (X, y) in tqdm.tqdm(train_loader) :      
      optimizer.zero_grad()
      y_scores = model(X)    
      loss = criterion(y_scores, y)
      loss.backward()
      optimizer.step()
      total_loss += loss.item()  
    print("train_loss = {:.4f}".format(total_loss / len(train_loader.dataset)))
    print("dev_loss = {:.4f} dev_acc = {:.4f}".format(*perf(model, dev_loader, criterion)))

################################################################################

def pad_tensor(X, max_len):
  """
  Pad variable-length sequences with zeros up to max_len. If the sentence is 
  longer than max_len, it will be cropped to contain exactly max_len tokens.

  ## Parameters
  X : list of lists of int -- token indices for each sentence  
  max_len : int -- maximum sequence length for padding

  ## Returns : torch.LongTensor of shape (len(X), max_len) 
  Sentences in the form of a tensor, padded with zeros.
    """
  res = torch.full((len(X), max_len), 0)
  for (i, row) in enumerate(X) :
    x_len = min(max_len, len(X[i]))
    res[i,:x_len] = torch.LongTensor(X[i][:x_len])
  return res

################################################################################

def create_dataloader(words, tags, max_len=40, batch_size=32, shuffle=True) :
  """
  Create a PyTorch DataLoader for batched training or evaluation.

  ## Parameters
  words : list[list[int]] -- encoded sentences (list of token ID lists)  
  tags : list[int] -- class labels for each sentence (as integer IDs)  
  max_len : int -- maximum sentence length after padding (default=40)  
  batch_size : int -- number of samples per batch (default=32)  
  shuffle : bool -- shuffle data order (True for training, False for evaluation)

  ## Returns : DataLoader
  DataLoader containing padded aligned word and tag tensors, ready for training
  """
  dataset = TensorDataset(pad_tensor(words, max_len), torch.LongTensor(tags))
  return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle) 
  
################################################################################

def read_corpus(filename, wordvocab, tagvocab, in_type, train_mode=True):
  """
  Read corpus file, build or reuse vocabularies and prepare tensors for training.

  ## Parameters
  filename : str -- path to input text file  
  wordvocab : defaultdict -- maps input tokens to IDs
  tagvocab : defaultdict -- maps target class labels to IDs  
  in_type : str -- 'char' for character-level or other for token-level input  
  train_mode : bool -- build new vocabs if True (default), otherwise use vocabs passed as parameters

  ## Returns : (list, list, defaultdict, defaultdict)
  (list of sentences, list of tag sequences, wordvocab, tagvocab).
  Sentences and tag sequences are lists of integer IDs (from vocabs)
  """  
  if train_mode : # Creates the vocabularies, initializes with special token IDs
    wordvocab = collections.defaultdict(lambda : len(wordvocab))
    wordvocab["<PAD>"]; wordvocab["<UNK>"] # Create special token IDs      
    tagvocab = collections.defaultdict(lambda : len(tagvocab))
  words, tags = [], []
  with open(filename, 'r', encoding="utf-8") as corpus:
    for line in corpus:
      fields = line.strip().split()
      tags.append(tagvocab[fields[0]])
      fields = " ".join(fields[1:]) if in_type == "char" else  fields[1:]
      if train_mode :
        # wordvocab[w] creates entry if w absent from wordvocab[w]
        words.append([wordvocab[w] for w in fields]) 
      else :
        # wordvocab[w] returns ID of "<UNK>" token if w absent from wordvocab[w]
        words.append([wordvocab.get(w, wordvocab["<UNK>"]) for w in fields])
  return words, tags, wordvocab, tagvocab
    
################################################################################

if __name__ == "__main__" :
  # Check command-line arguments are correct
  if len(sys.argv) != 5 or sys.argv[3] not in ['bow', 'gru', 'cnn'] or \
     sys.argv[4] not in ['word', 'char'] : # Prefer using argparse, more flexible
    print("Usage: {} trainfile.txt devfile.txt bow|gru|cnn word|char".format(sys.argv[0]), file=sys.stderr) 
    sys.exit(-1)   
    
  # Create a dictionary to store the model's hyperparameters (to be saved later)
  hp = {"model_type": sys.argv[3], "in_type": sys.argv[4], "d_embed": 250, "d_hidden": 200}
  
  # Load the corpus and create (batched) data loaders
  train_words, train_tags, wordvocab, tagvocab = read_corpus(sys.argv[1], None, None, hp["in_type"])
  train_loader = create_dataloader(train_words, train_tags)
  dev_words, dev_tags, _, _ = read_corpus(sys.argv[2], wordvocab, tagvocab, hp["in_type"], train_mode=False)
  dev_loader = create_dataloader(dev_words, dev_tags, shuffle=False)
  
  # Instantiate the model depending ont the command-line arguments
  if hp["model_type"] == "bow" :
    model = BOWClassifier(hp["d_embed"], len(wordvocab), len(tagvocab))
  elif hp["model_type"] == "gru" :
    model = GRUClassifier(hp["d_embed"], hp["d_hidden"], len(wordvocab), len(tagvocab))
  else: #if hp["model_type"] == "cnn" :
    model = CNNClassifier(hp["d_embed"], hp["d_hidden"], len(wordvocab), len(tagvocab))
  
  # Actual training happens here!!!
  fit(model, train_loader, dev_loader, epochs=15)
  
  # Save the model to be used in predict
  torch.save({"wordvocab": dict(wordvocab), "tagvocab": dict(tagvocab), 
              "model_params": model.state_dict(), "hyperparams": hp}, "model.pt")
