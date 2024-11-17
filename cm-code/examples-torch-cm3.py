#!/usr/bin/env python3

from torch import tensor, nn

################################################################################
# Multi-task learning minimal example

class DummyMTL (nn.Module):
  
  def __init__(self):
    super().__init__() 
    self.a = nn.Parameter(tensor([2.],requires_grad=True))
    self.b = nn.Parameter(tensor([3.],requires_grad=True))
    
  def forward(self):    
    c = self.a + self.b # output of task 1 (sum)
    d = self.a * self.b # output of task 2 (product)
    return (c, d)       # Two "task" outputs
    
  def print_gradients(self):    
    print([f"{n}.grad={p.grad and p.grad.item()}" \
           for (n, p) in self.named_parameters()])

################################################################################
    
model = DummyMTL()
(c,d) = model()  # calls forward   
model.print_gradients()
# ['a.grad=None', 'b.grad=None']
c.backward() 
model.print_gradients()
# ['a.grad=1.0', 'b.grad=1.0']
d.backward()
model.print_gradients()
# ['a.grad=4.0', 'b.grad=3.0']

################################################################################

# Now imagine c and d are 2 loss functions
model = DummyMTL()
(c,d) = model()  # calls forward   
model.print_gradients()
# ['a.grad=None', 'b.grad=None']
loss = c + d # combined during training
loss.backward()
model.print_gradients()
# ['a.grad=4.0', 'b.grad=3.0']

################################################################################
# Conv1D examples

import torch

mat = torch.rand(3,5)
conv = nn.Conv1d(in_channels=3, out_channels=2, kernel_size=3)
print(conv(mat).shape)

#conv_res = self.char_conv[str(k_s)](char_embs[:,word_i].transpose(1,2))
#pool_res = nn.functional.max_pool1d(conv_res, conv_res.shape[-1])

