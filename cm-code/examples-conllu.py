#!/usr/bin/env python

################################################################################
### CM1 exercise to get familiar with conllu format and library

import conllu, sys
conllufile = open(sys.argv[1], 'r', encoding='UTF-8')
slens, wlens = [], []
for sent in conllu.parse_incr(conllufile):
  slens.append(len(sent))  
  wlens.extend([len(token['form']) for token in sent])
print("Avg sent len={:.2f}".format(sum(slens)/len(slens)))
print("Avg word len={:.2f}".format(sum(wlens)/len(wlens)))

import matplotlib.pyplot as plt
f,(a1,a2) = plt.subplots(1,2)
a1.hist(slens,bins=20)
a1.set_title("Sentence length")
a2.hist(wlens,bins=20)
a2.set_title("Word length")
plt.show()

################################################################################
### CM3 exercise to manipulate morphological features

from collections import Counter, defaultdict

conllufile = open(sys.argv[1], 'r', encoding='UTF-8')
slens, wlens = [], []
feats_pos_dict = defaultdict(lambda: Counter())
feats_dict = defaultdict(lambda: Counter())
no_feat = total_words = total_features = 0

for sent in conllu.parse_incr(conllufile):
  for w in sent :
    total_words += 1
    if w["feats"] :
      for (key,value) in w["feats"].items():
        feats_pos_dict[w["upos"]][key] += 1
        feats_dict[key][value] += 1
        total_features += 1
    else:
      no_feat += 1
      
print(f"Number of feature keys: {len(feats_dict)}")
for feat in feats_dict:
  print(f"  {feat}: {list(feats_dict[feat])}")
print(' ' * 9 + '|' + '|'.join([f"{pos:5}" for pos in feats_pos_dict])+'|')
for feat in feats_dict:
  print(f"{feat:9}|" + '|'.join([f"{feats_pos_dict[pos][feat] if feats_pos_dict[pos][feat] else ' ':5}" for pos in feats_pos_dict]) + '|')
print(f"Words with no feature: {no_feat}/{total_words} ({no_feat/total_words*100:.2f}%)")
print(f"Average features per word: {total_features/total_words:.2f}")



