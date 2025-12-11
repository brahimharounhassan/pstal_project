#!/usr/bin/env python

from collections import Counter
import conllu
import sys

conllufile = open(sys.argv[1], 'r', encoding='UTF-8')

featureless_count = 0
features_names = {}

for sent in conllu.parse_incr(conllufile):
  for word in sent :
    if word["feats"] is None :
      featureless_count += 1
    else:
      for key in word["feats"] :
        features_names[key] = 1
      
print(featureless_count)
print(len(features_names))


