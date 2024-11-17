#!/usr/bin/bash
#Count number of different morphological supertags in Sequoia
cat ../../tp/share/sequoia/sequoia-ud.parseme.frsemcor.simple.full | 
grep "^[0-9]" | 
cut -d "	" -f 6 | 
sort | 
uniq | 
wc -l
