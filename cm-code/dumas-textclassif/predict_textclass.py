#!/usr/bin/env python3
import sys, torch, collections, tqdm, pdb
import torch.nn as nn
from train_textclass import read_corpus, BOWClassifier, GRUClassifier, CNNClassifier

################################################################################

def rev_vocab(vocab):
    rev_dict = {y: x for x, y in vocab.items()}
    return [rev_dict[k] for k in range(len(rev_dict))]

################################################################################

if __name__ == "__main__" :    
  if len(sys.argv) != 3 : # Prefer using argparse, more flexible
    print("Usage: {} testfile.txt modelfile.pt".format(sys.argv[0]), file=sys.stderr) 
    sys.exit(-1)
    
  load_dict = torch.load(sys.argv[2], weights_only=False)
  wordvocab = load_dict["wordvocab"]
  tagvocab = load_dict["tagvocab"]
  hp = load_dict["hyperparams"]  
  if hp["model_type"] == 'bow' :
    model = BOWClassifier(hp["d_embed"], len(wordvocab), len(tagvocab))
  elif hp["model_type"] == 'gru' :
    model = GRUClassifier(hp["d_embed"], hp["d_hidden"], len(wordvocab), len(tagvocab))
  else : #if hp["model_type"] == 'cnn' :
    model = CNNClassifier(hp["d_embed"], hp["d_hidden"], len(wordvocab), len(tagvocab))
  model.load_state_dict(load_dict["model_params"])
  
  words, _, _, _ = read_corpus(sys.argv[1], wordvocab, tagvocab, hp["in_type"], 
                               train_mode=False, batch_mode=False)
  revtagvocab = rev_vocab(tagvocab)
  for sentence in words :
    pred_scores = model(torch.LongTensor([sentence])) # No need to batch
    print(revtagvocab[pred_scores.argmax()]) # No need to softmax



