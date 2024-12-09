#!/usr/bin/env python3

import sys
from lib.conllulib import TransBasedSent, CoNLLUReader

cur = CoNLLUReader(open(sys.argv[1],'r',encoding='utf-8'))
for sent in cur.readConllu() :
  tbs = TransBasedSent(sent)
  print(tbs)
