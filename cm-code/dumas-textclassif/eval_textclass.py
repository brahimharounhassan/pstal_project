#!/usr/bin/env python3
import sys

if __name__ == "__main__" :    
  if len(sys.argv) != 3 : # Prefer using argparse, more flexible
    print("Usage: {} gold-testfile.txt pred-testfile.pt".format(sys.argv[0]), file=sys.stderr) 
    sys.exit(-1)
  with open(sys.argv[1], 'r', encoding='utf-8') as goldfile,\
       open(sys.argv[2], 'r', encoding='utf-8') as predfile:
    total = correct = 0
    for (gline, pline) in zip(goldfile, predfile) :
      correct += int(gline.strip().split()[0] == pline.strip().split()[0])
      total += 1
  print(f"Accuracy = {correct * 100 / total:.2f}")

# ./eval_textclass.py dumas_test.txt dumas_test_pred-bow.txt
# ./eval_textclass.py dumas_test.txt dumas_test_pred-gru.txt
