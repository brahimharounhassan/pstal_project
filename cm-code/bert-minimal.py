#!/usr/bin/env python3

import torch
from transformers import AutoModel, AutoTokenizer

name  = 'almanach/camembert-base'
#sent  = "Des poids lourds et engins en feu \
#         dans une entreprise en Vendée ."
sent = "La gare routière attend toujours ses illuminations ."
#sent = "Quelle surprise ! Arturo a la covid"
tok   = AutoTokenizer.from_pretrained(name)
model = AutoModel.from_pretrained(name)

tok_sent = tok(sent.split(), is_split_into_words=True, 
               return_tensors='pt')
tok_ids  = tok_sent['input_ids'][0]
decoded = tok.convert_ids_to_tokens(tok_ids) 
print(decoded)
print(tok_sent.word_ids())
with torch.no_grad(): # no training
  embeds = model(**tok_sent)['last_hidden_state'][0]
print(embeds.shape)
