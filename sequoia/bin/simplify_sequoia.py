#!/usr/bin/env python3
"""
Simplify Sequoia corpus for pedagogical purposes:
- Remove all range tokens (e.g. "2-3 du" = "2 de" + "3 le"), keep only full tokens
  => Range tokens usually contain no annotation: they mark the presence of a contraction
  => The text may become strange to read, e.g. "L'ambassadrice de le Portugal à les Pays-Bas"
- Column FRSEMCOR:NOUN (TP4)
  - Remove all supersense annotations for multiword units 
    => keeping multiwords would make data preparation unnecessarily complex
  - Keep all simple (non-MWE) supersenses for NOUN, PROPN and NUM, remove others 
    => this allows the classifier to focus on these POS tags and ignore very low-frequency ones
  - For composed supersenses ("/", "x" or "+" operators), keep last element 
    - e.g. Artifact/Cognition -> Cognition
    => This reduces the tagset for prediction, making the task a bit easier for the classifier
- Column PARSEME:MWE
  - Keep in PARSEME:MWE only multiword expression annotations
  - Add a column PARSEME:NE with named entity annotations separately  
    => This allows working on named entity recognition as a standalone task (TP2)
  - Remove from PARSEME:NE:
    - (a) discontinuous NEs (e.g. [Jeanine] et Willy [Schaer]), 
    - (b) overlapping NEs (keep the longest one, choose randomly if same length)
  => Overlaps and discontinuities are hard to represent in sequence labelling models
- Columns HEAD and DEPREL (TP5 and TP6)
  - Remove non-projective sentences
  => Non-projective parse trees are not straightforward to handle in the dependency parsing models we implement
  - [EXPERIMENTAL] Remove all deprel subrelations (after semicolon) to simplify the tagset
  
This script depends on the `cuptlib` library. You can install it with:

git clone https://gitlab.com/parseme/cuptlib.git
cd cuptlib
pip install .
"""

import sys
import conllu
import re
import pdb
import subprocess
try :
  import parseme.cupt as cupt
except ImportError:
  print("""Please install cuptlib before running this script\n\n  git clone \
  https://gitlab.com/parseme/cuptlib.git\n  cd cuptlib\n  pip install .""")
  sys.exit(-1)

#########################################

def remove_range_tokens(sent):
  range_counter = 0
  for (token_i, token) in enumerate(sent) :
    if type(token["id"]) != int : # Sentence ID is a complex object, remove it
      sent.pop(token_i)
      range_counter = range_counter + 1
  return range_counter

#########################################

def simplify_supersense(sent):
  del_ssense_counter = mod_ssense_counter = 0
  for token in sent:
    ssense_tags = token["frsemcor:noun"].split(";")
    for ssense_tag in ssense_tags :
      if ssense_tag[0].isdigit(): # Remove MWE supersense labels
        del_ssense_counter = del_ssense_counter + 1
        token["frsemcor:noun"] = "*"
      elif ssense_tag != "*" and token["upos"] not in ["NOUN", "PROPN", "NUM"]:      
        del_ssense_counter = del_ssense_counter + 1
        token["frsemcor:noun"] = "*"
      elif "/" in ssense_tag or "+" in ssense_tag or "x" in ssense_tag:
        token["frsemcor:noun"] = re.split("/|\+|x",ssense_tag)[-1]    
        mod_ssense_counter = mod_ssense_counter + 1
      else:
        token["frsemcor:noun"] = ssense_tag
    if token["frsemcor:noun"] == "Felling" : # Correct typos in one SSense tag
      token["frsemcor:noun"] = "Feeling" 
  return del_ssense_counter, mod_ssense_counter

def simplify_morphology(sent):
  for token in sent:
    if token["feats"] :
      if "Typo" in token["feats"]:
        del token['feats']['Typo']
      elif token["upos"] in ["ADV", "NUM"]:
        if "Number" in token["feats"]:
          del token['feats']['Number']
        elif "Gender" in token['feats']:
          del token['feats']['Gender']      

#########################################
  
def simplify_mwe_ne(sent):
  ne_ind = 1 # Start new named entities at index 1 in new column
  del_ne_counter = 0
  mwes = cupt.retrieve_mwes(sent) # get all MWE annotations  
  ne_list = []
  mwe_list = []
  for mwe in mwes.values():
    if mwe.cat.startswith("PROPN|NE"): # Named entity, add to NE list
      if mwe.n_gaps() == 0:
        ne_list.append(mwe)        
      else:
        del_ne_counter = del_ne_counter + 1
    else:
      mwe_list.append(mwe)    
  cupt.replace_mwes(sent, mwe_list) # Clean all annotations, add only "MWE" ones  
  def sorting_key(x):
    return (x.n_tokens(),len(sent)-sorted(list(x.span))[0],"final" in x.cat)
  ne_list_sort = sorted(ne_list,key=sorting_key,reverse=True)
  for token in sent:
    token["parseme:ne"] = "*"
  for ne in ne_list_sort:
    first_word = sorted(list(ne.span))[0]-1 # -1 accesses list position, not token ID
    if sent[first_word]["parseme:ne"] == "*": # No overlap, continue
      new_ne_cat = str(ne_ind) + ":" + ne.cat.split("-")[1].split(".")[0]
      sent[first_word]["parseme:ne"] = new_ne_cat
      for ne_i in sorted(list(ne.span))[1:]:
        sent[ne_i-1]["parseme:ne"] = str(ne_ind)
      ne_ind = ne_ind + 1
    else:
      del_ne_counter = del_ne_counter + 1
  return del_ne_counter    

#########################################

def is_projective(sent):
  for token in sent :
    start = dep_id = token["id"]
    end = head_id = token["head"]
    if dep_id > head_id :
      start = head_id
      end = dep_id
    for token_i in range(start,end-1): # sent is 0-indexed, ID is 1-indexed
      if sent[token_i]["head"] < start or sent[token_i]["head"] > end :               
        return False            
  return True     

#########################################

def remove_subrelations(sent):
  subrel_counter = sum([1 if ':' in t['deprel'] else 0 for t in sent])
  for token in sent :
    token['deprel'] = re.sub(':.*', '', token['deprel'])
  return subrel_counter

#########################################

if len(sys.argv) != 2:
  print('Usage: {} <input_corpus.conllu>'.format(sys.argv[0]), file=sys.stderr)  
  exit(-1)

with open(sys.argv[1], "r", encoding="UTF=8") as f:
  np_counter = range_counter = del_ne_counter = 0
  del_ssense_counter = mod_ssense_counter = 0 #subrel_counter = 0
  np_ids = []  
  for sent in conllu.parse_incr(f):    
    range_counter = range_counter + remove_range_tokens(sent)
    del_ssense_ci, mod_ssense_ci = simplify_supersense(sent)
    del_ssense_counter = del_ssense_counter + del_ssense_ci
    mod_ssense_counter = mod_ssense_counter + mod_ssense_ci
    del_ne_counter = del_ne_counter + simplify_mwe_ne(sent)
    simplify_morphology(sent)
#    subrel_counter = subrel_counter + remove_subrelations(sent)
    if is_projective(sent) : # Returns false to remove sentence
      if sent.metadata.get("global.columns", None): # Add header for new column
        sent.metadata["global.columns"] += " PARSEME:NE"
      print(sent.serialize(), end="")
    else:
      np_counter += 1
      np_ids.append(sent.metadata["sent_id"])
            
print( "{} range tokens removed.\n".format(range_counter), file=sys.stderr)

print( "{} discontinuous and overlapping NEs removed.\n".format(del_ne_counter), file=sys.stderr)

print( "{} supersense tags removed (on MWEs or strange POS).".format(del_ssense_counter), file=sys.stderr)
print( "{} supersense tags modified (complex operators).\n".format(mod_ssense_counter), file=sys.stderr)

#print( "{} subrelations removed from deprel.".format(subrel_counter), file=sys.stderr)
print( "{} non-projective sentences removed:".format(np_counter), file=sys.stderr)
print(", ".join(np_ids), file=sys.stderr)
