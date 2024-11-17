#!/usr/bin/env python

################################################################################
# defaultdict examples

from collections import defaultdict
word_count = defaultdict(lambda: 0) # return 0 if absent key
sentence = "the man and the dog and the tree"
for word in sentence.split():
  word_count[word] += 1
print(dict(word_count)) 
# {'the': 3, 'man': 1, 'and': 2, 'dog': 1, 'tree': 1}
print(word_count["hi"]) 
# 0  
print(dict(word_count)) 
# {'the': 3, 'man': 1, 'and': 2, 'dog': 1, 'tree': 1, 'hi': 0}

################################################################################
# Embedding layer examples

from torch import LongTensor
import torch.nn as nn

embed = nn.Embedding(20, 4, padding_idx=0)
print(embed.weight[:2])
# tensor([[ 0.0000,  0.0000,  0.0000,  0.0000],
#        [-0.9336, -0.0982,  0.2726,  1.9872], grad_fn=<...>)
print(embed(LongTensor([3,4])).shape)
# torch.Size([2, 4])
print(embed(LongTensor([3, 4, 3, 0])).shape)
# torch.Size([4, 4])
print(embed(LongTensor([[3, 4, 3], [0, 1, 1]])).shape) # batched
# torch.Size([2, 3, 4])

################################################################################
# RNN unit examples

gru = nn.GRU(4, 10, batch_first=True)
emb = nn.Embedding(20, 4, padding_idx=0)
x   = emb(LongTensor([[3, 4, 3, 2, 5],   # B=2 (batch)
                      [1, 12, 1, 0, 0]])) # L=5 (timesteps)
print(x.shape)
# torch.Size([2, 5, 4])
y = gru(x)
print(y[0].shape)
# torch.Size([2, 5, 10])
print(y[1].shape)
# torch.Size([1, 2, 10]) # notice batch is dim1

################################################################################
# TensorDataset/DataLoader examples

from torch.utils.data import TensorDataset, DataLoader
import torch
x = torch.rand(7,3)
y = LongTensor([[i] for i in range(7)])
tds = TensorDataset(x,y)
print(len(tds))
# 7
print([(e.shape, e.dtype) for e in tds[5]])
# [(torch.Size([3]), torch.float32), (torch.Size([1]), torch.int64)]
dl = DataLoader(tds, batch_size=2)
print(next(iter(dl))[0].shape)
#torch.Size([2, 3])
print(next(iter(dl))[1].shape)
#torch.Size([2, 1])
for (bx, by) in dl :
  print(bx.shape, by.shape)
  
################################################################################


