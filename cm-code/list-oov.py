#!/usr/bin/env python3

import sys
import argparse
from lib.conllulib import CoNLLUReader

################################################################################

parser = argparse.ArgumentParser(description="Lists words in inference corpus \
absent from train.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-i', "--inference", metavar="FILENAME.conllu", required=True,\
        dest="infer_filename", type=argparse.FileType('r', encoding='UTF-8'), \
        help="""Inference corpus in CoNLLU. (Required)""")
parser.add_argument('-t', "--train", metavar="FILENAME.conllu", required=True,\
        dest="train_filename", type=argparse.FileType('r', encoding='UTF-8'), \
        help="""Training corpus in CoNLL-U.""")        
parser.add_argument('-f', "--featcolumn", metavar="NAME", dest="name_feat",
        required=False, type=str, default="form", help="""Column name of input 
        feature, as defined in header. Use lowercase.""")
parser.add_argument('-c', "--characters", dest="chars", required=False, 
        action="store_true", help="""List characters instead of words""")
parser.add_argument('-C', "--count", dest="count", required=False, 
        action="store_true", help="""Count instead of listing""")
parser.add_argument('-u', "--upos-filter", metavar="NAME", dest="upos_filter",
        required=False, type=str, nargs='+', default=[], 
        help="""Only list OOV for words with UPOS in this list. \
        Empty list = no filter.""")        
                            
################################################################################

def process_args(parser):
  """
  Show (in debug mode) and process all command line options. Checks feat columns
  appear in corpora. Create training corpus vocabulary for OOV status check. 
  Input is an instance of `argparse.ArgumentParser`, returns list of `args`, 
  `infer_corpus` and `pred_corpus` as `CoNLLUReader`, `train_vocab` dictionary. 
  """
  args = parser.parse_args()
  args.name_feat = args.name_feat.lower()
  infer_corpus = CoNLLUReader(args.infer_filename)   
  train_corpus = CoNLLUReader(args.train_filename)
  _, vocab = train_corpus.to_int_and_vocab({args.name_feat:[]}, chars=args.chars)    
  if args.name_feat not in infer_corpus.header:
    Util.error("-f name must be valid conllu column among:\n{}", 
               infer_corpus.header)
  return args, infer_corpus, vocab
      
################################################################################

if __name__ == "__main__":
  args, infer_corpus, train_vocab = process_args(parser)
  oov_list = []
  total_count = 0
  for s_infer in infer_corpus.readConllu():    
    for tok_infer in s_infer:
      if not args.upos_filter or tok_infer['upos'] in args.upos_filter :        
        if args.chars:
          for ch in tok_infer[args.name_feat] :
            total_count += 1
            if ch not in train_vocab[args.name_feat]:
              oov_list.append(ch)
        else:
          total_count += 1
          if tok_infer[args.name_feat] not in train_vocab[args.name_feat]:
            oov_list.append(tok_infer[args.name_feat])            
  print("Inference file: {}".format(infer_corpus.name() ), file=sys.stderr)
  if args.upos_filter :
    print("Results on UPOS: {}".format(" ".join(args.upos_filter)), file=sys.stderr)
  if args.count :
    print(f"{len(oov_list)}/{total_count}={len(oov_list)/total_count*100:.2f}% OOVs") 
  else:
    print("\n".join(oov_list))
