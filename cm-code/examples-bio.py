#!/usr/bin/env python3

from lib.conllulib import CoNLLUReader
test="""# global.columns = ID FORM parseme:ne
1	Le	1:PROD
2	Petit	1
3	Prince	1
4	de	*
5	Saint-Exupéry	2:PERS
6	est	*
7	entré	*
8	à	*
9	l'	*
10	École	3:ORG
11	Jules-Romains	3"""
for sent in CoNLLUReader.readConlluStr(test):
  print(CoNLLUReader.to_bio(sent))
#['B-PROD', 'I-PROD', 'I-PROD', 'O', 'B-PERS', 'O', 'O', 'O', 'O', 'B-ORG', 'I-ORG']
  
s1 = ["B-PERS", "I-PERS", "I-PERS", "O", "B-LOC", "I-LOC"]
s2 = ["I-PERS", "B-PERS", "I-PERS", "O", "I-LOC"]
print(CoNLLUReader.from_bio(s1, bio_style='bio'))
# ['1:PERS', '1', '1', '*', '2:LOC', '2']
print(CoNLLUReader.from_bio(s1, bio_style='io'))
# WARNING: Got B tag in spite of 'io' bio_style: interpreted as I
# WARNING: Got B tag in spite of 'io' bio_style: interpreted as I
# ['1:PERS', '1', '1', '*', '2:LOC', '2']
print(CoNLLUReader.from_bio(s2, bio_style='bio'))
# WARNING: Invalid I-initial tag I-PERS converted to B
# WARNING: Invalid I-initial tag I-LOC converted to B
# ['1:PERS', '2:PERS', '2', '*', '3:LOC']
print(CoNLLUReader.from_bio(s2, bio_style='io'))
# WARNING: Got B tag in spite of 'io' bio_style: interpreted as I
# ['1:PERS', '1', '1', '*', '2:LOC']
