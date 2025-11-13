#!/usr/bin/env python3
"""
Sentence-level text classification: prediction script.

This script loads a trained model (BOW, GRU, or CNN) and its vocabularies,
reads a test corpus in the same format as the training data, and prints
the predicted class label for each sentence, one per line.

Usage:
    ./predict_textclass.py testfile.txt modelfile.pt
"""

import sys, torch, collections, tqdm, pdb
import torch.nn as nn
from train_textclass import read_corpus, BOWClassifier, GRUClassifier, CNNClassifier

################################################################################

def rev_vocab(vocab):
  """
  Create a reverse vocabulary mapping from ID to token.

  ## Parameters
  vocab : dict[str->int] -- vocabulary mapping label names to integer IDs

  ## Returns : list[str]
  List of labels ordered by their integer ID. Label name can be retrieved by 
  accessing position corresponding to label ID.
  """
  rev_dict = {y: x for x, y in vocab.items()}
  return [rev_dict[k] for k in range(len(rev_dict))]

################################################################################

if __name__ == "__main__" :    
  # Check command-line arguments
  if len(sys.argv) != 3 : # Prefer using argparse, more flexible
    print("Usage: {} testfile.txt modelfile.pt".format(sys.argv[0]), file=sys.stderr) 
    sys.exit(-1)
    
  # Load trained model and vocabularies
  load_dict = torch.load(sys.argv[2], weights_only=False)
  wordvocab = load_dict["wordvocab"]
  tagvocab = load_dict["tagvocab"]
  hp = load_dict["hyperparams"]  
  
  # Recreate the model with the same architecture and parameters
  if hp["model_type"] == 'bow' :
    model = BOWClassifier(hp["d_embed"], len(wordvocab), len(tagvocab))
  elif hp["model_type"] == 'gru' :
    model = GRUClassifier(hp["d_embed"], hp["d_hidden"], len(wordvocab), len(tagvocab))
  else : #if hp["model_type"] == 'cnn' :
    model = CNNClassifier(hp["d_embed"], hp["d_hidden"], len(wordvocab), len(tagvocab))
  model.load_state_dict(load_dict["model_params"])
  
  # Read test sentences (without rebuilding vocabularies)
  words, _, _, _ = read_corpus(sys.argv[1], wordvocab, tagvocab, hp["in_type"], 
                               train_mode=False)
                               
  # Predict and print one label per sentence
  revtagvocab = rev_vocab(tagvocab)
  for sentence in words :
    pred_scores = model(torch.LongTensor([sentence])) # No need to batch
    print(revtagvocab[pred_scores.argmax()]) # No need to softmax



