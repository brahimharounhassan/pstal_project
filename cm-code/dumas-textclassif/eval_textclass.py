#!/usr/bin/env python3
"""
Evaluate sentence-level text classification predictions.

This script computes the accuracy of predicted labels against a gold standard file.
Both files should contain one sentence per line, with the first field being the class label.

Usage:
    ./eval_textclass.py gold-testfile.txt pred-testfile.txt
"""

import sys

if __name__ == "__main__" :    
  # Check command-line arguments
  if len(sys.argv) != 3 : # Prefer using argparse, more flexible
    print("Usage: {} gold-testfile.txt pred-testfile.pt".format(sys.argv[0]), file=sys.stderr) 
    sys.exit(-1)
    
  # Open gold standard and prediction files
  with open(sys.argv[1], 'r', encoding='utf-8') as goldfile,\
       open(sys.argv[2], 'r', encoding='utf-8') as predfile:
    total = correct = 0
    
    # Compare first token (class label) on each line
    for (gline, pline) in zip(goldfile, predfile) :
      correct += int(gline.strip().split()[0] == pline.strip().split()[0])
      total += 1
  
  # Print overall accuracy score
  print(f"Accuracy = {correct * 100 / total:.2f}")

