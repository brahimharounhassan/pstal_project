#!/usr/bin/env python3

import sys
import argparse
from collections import defaultdict, Counter
import pdb
from conllulib import CoNLLUReader, Util
import re

################################################################################

parser = argparse.ArgumentParser(description="Calculates the accuracy of a \
prediction with respect to the gold file. By default, uses UPOS, but this can \
be configured with option --tagcolumn. For columns `feats` and `parseme:ne`, \
calculates also the precision, recall, F-score. For columns `head` and \
`deprel`, calculates LAS and UAS.",
formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-D', "--debug", action="store_true", dest="DEBUG_FLAG", 
        help="""Print debug information (grep it or pipe into `less -SR`)""")
parser.add_argument('-p', "--pred", metavar="FILENAME.conllu", required=True,\
        dest="pred_filename", type=argparse.FileType('r', encoding='UTF-8'), \
        help="""Test corpus in CoNLLU with *predicted* tags. (Required)""")
parser.add_argument('-g', "--gold", metavar="FILENAME.conllu", required=True,\
        dest="gold_filename", type=argparse.FileType('r', encoding='UTF-8'), \
        help="""Test corpus in CoNLLU with *gold* tags. (Required)""")
parser.add_argument('-t', "--train", metavar="FILENAME.conllu", required=False,\
        dest="train_filename", type=argparse.FileType('r', encoding='UTF-8'), \
        help="""Training corpus in CoNLL-U, from which tagger was learnt.""")        
parser.add_argument('-c', "--tagcolumn", metavar="NAME", dest="name_tag",
        required=False, type=str, default="upos", help="""Column name of tags, \
        as defined in header. Use lowercase.""")   
parser.add_argument('-f', "--featcolumn", metavar="NAME", dest="name_feat",
        required=False, type=str, default="form", help="""Column name of input 
        feature, as defined in header. Use lowercase.""")
parser.add_argument('-u', "--upos-filter", metavar="NAME", dest="upos_filter",
        required=False, type=str, nargs='+', default=[], 
        help="""Only calculate accuracy for words with UPOS in this list. \
        Empty list = no filter.""")        
                            
################################################################################

def process_args(parser):
  """
  Show (in debug mode) and process all command line options. Checks tag and feat
  columns appear in corpora. Create training corpus vocabulary if option present
  for OOV status check. Input is an instance of `argparse.ArgumentParser`, 
  returns list of `args`, `gold_corpus` and `pred_corpus` as `CoNLLUReader`, 
  `train_vocab` dictionary. 
  """
  args = parser.parse_args()
  Util.DEBUG_FLAG = args.DEBUG_FLAG
  args.name_tag = args.name_tag.lower()
  args.name_feat = args.name_feat.lower()
  Util.debug("Command-line arguments and defaults:")
  for (k,v) in vars(args).items():
    Util.debug("  * {}: {}",k,v)    
  gold_corpus = CoNLLUReader(args.gold_filename) 
  pred_corpus = CoNLLUReader(args.pred_filename) 
  train_vocab = None
  if args.train_filename:
    train_corpus = CoNLLUReader(args.train_filename)
    ignoreme, train_vocab = train_corpus.to_int_and_vocab({args.name_feat:[]})    
  if args.name_tag  not in gold_corpus.header or \
     args.name_feat not in gold_corpus.header:
    Util.error("-c and -f names must be valid conllu column among:\n{}", 
               gold_corpus.header)
  return args, gold_corpus, pred_corpus, train_vocab
  
################################################################################

def tp_count_feats(tok_pred, tok_gold, prf):
  """
  Increment number of true positives, trues and positives for morph feature eval
  Compares all features of `tok_pred` with thos of `tok_gold`
  Result is modification of `prf` dict, function does not return anything
  """
  pred_feats = tok_pred['feats'] if tok_pred['feats'] else {}
  gold_feats = tok_gold['feats'] if tok_gold['feats'] else {}          
  for key in pred_feats.keys():
    tp_inc = int(gold_feats.get(key,None) == pred_feats[key])
    prf[key]['tp'] = prf[key]['tp'] + tp_inc
    prf['micro-avg']['tp'] = prf['micro-avg']['tp'] + tp_inc
    p_inc = int(pred_feats.get(key,None) != None)
    prf[key]['p'] = prf[key]['p'] + p_inc
    prf['micro-avg']['p'] = prf['micro-avg']['p'] + p_inc
  for key in gold_feats.keys():
    t_inc = int(gold_feats.get(key,None) != None)
    prf[key]['t'] = prf[key]['t'] + t_inc
    prf['micro-avg']['t'] = prf['micro-avg']['t'] + t_inc

################################################################################

def parseme_cat_in(ent, ent_list):
  """
  Verify if `ent` is present in `ent_list` by comparing both span AND category.
  Default cuptlib implementation ignores category
  """
  for ent_cand in ent_list:
    if ent.span == ent_cand.span and ent.cat == ent_cand.cat :
      return True
  return False

################################################################################

def tp_count_parseme(s_pred, s_gold, name_tag, prf):
  """
  Count true positives, trues and positives for full entities in PARSEME format.
  Updates `prf` dict with counts from sentence `s_pred` and sentence `s_gold`
  `name_tag` is the name of the column among `parseme:ne` or `parseme:mwe`
  This code was not tested for `parseme:mwe`.
  """
  try :
    import parseme.cupt as cupt
  except ImportError:
    print("""Please install cuptlib before running this script\n\n  git clone \
https://gitlab.com/parseme/cuptlib.git\n  cd cuptlib\n  pip install .""")
    sys.exit(-1)
  ents_pred = cupt.retrieve_mwes(s_pred, column_name=name_tag)
  ents_gold = cupt.retrieve_mwes(s_gold, column_name=name_tag)
  prf['Exact-nocat']['p'] += len(ents_pred)
  prf['Exact-nocat']['t'] += len(ents_gold)
  for e_pred in ents_pred.values() :         
    if e_pred in ents_gold.values() :
      prf['Exact-nocat']['tp'] += 1
    if parseme_cat_in(e_pred, ents_gold.values()) :  
      prf['Exact-'+e_pred.cat]['tp'] += 1    
    prf['Exact-'+e_pred.cat]['p'] += 1
  for e_pred in ents_gold.values() :
    prf['Exact-'+e_pred.cat]['t'] += 1
  # Fuzzy (token-based) evaluation - categories always ignored here
  span_pred = sum([list(ep.int_span()) for ep in ents_pred.values()], start=[])
  span_gold = sum([list(eg.int_span()) for eg in ents_gold.values()], start=[])
  prf['Fuzzy-nocat']['p'] += len(span_pred)
  prf['Fuzzy-nocat']['t'] += len(span_gold)  
  for e_pred in span_pred :       
    if e_pred in span_gold :      
      prf['Fuzzy-nocat']['tp'] += 1
      
################################################################################

def print_results(pred_corpus_name, args, acc, prf, parsing=False):
  """
  Calculate and print accuracies, precision, recall, f-score, LAS, etc.
  """
  print("Predictions file: {}".format(pred_corpus_name))
  if args.upos_filter :
    print("Results concern only some UPOS: {}".format(" ".join(args.upos_filter)))
  accuracy = (acc['correct_tokens'] / acc['total_tokens']) * 100  
  if not parsing:
    acc_name = "Accuracy"
  else: 
    acc_name = "UAS"
  print("{} on all {}: {:0.2f} ({:5}/{:5})".format(acc_name, args.name_tag, 
        accuracy, acc['correct_tokens'], acc['total_tokens']))
  if parsing :
    accuracy_las = (acc['correct_tokens_las'] / acc['total_tokens']) * 100
    print("LAS on all {}: {:0.2f} ({:5}/{:5})".format(args.name_tag, 
          accuracy_las, acc['correct_tokens_las'], acc['total_tokens']))
  if args.train_filename :
    accuracy_oov = (acc['correct_oov'] / acc['total_oov']) * 100
    print("{} on OOV {}: {:0.2f} ({:5}/{:5})".format(acc_name, args.name_tag, 
          accuracy_oov, acc['correct_oov'], acc['total_oov']))
    if parsing :
      accuracy_oov_las = (acc['correct_oov_las'] / acc['total_oov']) * 100
      print("LAS on OOV {}: {:0.2f} ({:5}/{:5})".format(args.name_tag, 
          accuracy_oov_las, acc['correct_oov_las'], acc['total_oov']))
  if prf:
    print("\nPrecision, recall, and F-score for {}:".format(args.name_tag))
    macro = {"precis":0.0, "recall":0.0}
    for key in sorted(prf): # max prevents zero-division in P and R       
      precis = (prf[key]['tp'] / max(1, prf[key]['p'])) * 100
      recall = (prf[key]['tp'] / max(1, prf[key]['t'])) * 100
      fscore = ((2 * precis * recall) / max(1, precis + recall))
      if key != 'micro-avg':
        macro['precis'] = macro['precis'] + precis
        macro['recall'] = macro['recall'] + recall
      else:
        print()
      templ = "{:11}: P={:6.2f} ({:5}/{:5}) / R={:6.2f} ({:5}/{:5}) / F={:6.2f}"      
      print(templ.format(key, precis, prf[key]['tp'], prf[key]['p'], recall, 
                         prf[key]['tp'], prf[key]['t'], fscore))
    templ = "{:11}: P={:6.2f}" + " "*15 + "/ R={:6.2f}" + " "*15 + "/ F={:6.2f}"    
    if len(prf) > 1 : # Calculate macro-precision
      nb_scores = len(prf)-1 if "micro-avg" in prf else len(prf)
      ma_precis = (macro['precis'] / (nb_scores)) 
      ma_recall = (macro['recall'] / (nb_scores)) 
      ma_fscore = ((2*ma_precis*ma_recall)/max(1,ma_precis+ma_recall))
      print(templ.format("macro-avg", ma_precis, ma_recall, ma_fscore))

################################################################################

if __name__ == "__main__":
  args, gold_corpus, pred_corpus, train_vocab = process_args(parser)
  prf = defaultdict(lambda:{'tp':0,'t':0, 'p':0}) # used for feats, NEs and MWEs
  acc = Counter() # store correct and total for all and OOV
  parsing = False
  for (s_gold,s_pred) in zip(gold_corpus.readConllu(),pred_corpus.readConllu()):
    if args.name_tag.startswith("parseme"):
      tp_count_parseme(s_pred, s_gold, args.name_tag, prf)
    if args.name_tag in ["head", "deprel"]: # Any of both is considered LAS/UAS eval
      args.name_tag = "head"
      parsing = True
    for (tok_gold, tok_pred) in zip (s_gold, s_pred):
      if not args.upos_filter or tok_gold['upos'] in args.upos_filter :
        if train_vocab :
          train_vocab_feat = train_vocab[args.name_feat].keys()
          if tok_gold[args.name_feat] not in train_vocab_feat:
            acc['total_oov'] += 1
            oov = True
          else:
            oov = False
        if tok_gold[args.name_tag] == tok_pred[args.name_tag]:        
          acc['correct_tokens'] += 1       
          if train_vocab and oov :
            acc['correct_oov'] += 1
        # LAS ignores subrelations, as usual in CoNLL17/18 eval scripts
        gold_deprel = re.sub(':.*', '', tok_gold["deprel"])
        pred_deprel = re.sub(':.*', '', tok_pred["deprel"])
        if parsing and tok_gold["head"] == tok_pred["head"] and \
                       gold_deprel == pred_deprel: 
          acc['correct_tokens_las'] += 1
          if train_vocab and oov :
            acc['correct_oov_las'] += 1
        acc['total_tokens'] += 1
        if args.name_tag == 'feats':
          tp_count_feats(tok_pred, tok_gold, prf)
  print_results(pred_corpus.name(), args, acc, prf, args.name_tag == "head")
