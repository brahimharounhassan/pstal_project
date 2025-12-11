#!/usr/bin/env python3

from torch import tensor, nn

################################################################################
# Multi-task learning minimal example

class DummyMTL (nn.Module):
  
  def __init__(self):
    super().__init__() 
    self.a = tensor([2.], requires_grad=True)
    self.b = tensor([3.], requires_grad=True)
    
  def forward(self):    
    c = self.a + self.b # output of task 1 (sum)
    d = self.a * self.b # output of task 2 (product)
    return (c, d)       # Two "task" outputs
    
  def print_gradients(self): # print [a.grad, b.grad]
    print([p.grad and p.grad.item() for p in [self.a, self.b]])
      
################################################################################
    
model = DummyMTL()
(c,d) = model()  # calls forward   
model.print_gradients()
# [None, None]
c.backward() 
model.print_gradients()
# [1.0, 1.0]
d.backward()
model.print_gradients()
# [4.0, 3.0]

################################################################################

# Now imagine c and d are 2 loss functions
model = DummyMTL()
(c,d) = model()  # calls forward   
model.print_gradients()
# [None, None]
loss = c + d # combined during training
loss.backward()
model.print_gradients()
# [4.0, 3.0]

################################################################################
# torch.gather examples

import torch

a = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
mask = torch.tensor([[1], [1]])
print(a.gather(dim=1, index=mask))
# tensor([[2.], [5.]])
print(a.gather(0, mask))
# tensor([[4.], [4.]])
mask = torch.tensor([[1,0,1], [0,1,0]])
print(a.gather(0, mask))
# tensor([[4., 2., 6.], [1., 5., 3.]])
mask = torch.tensor([[2,1], [0,2]])
print(a.gather(1, mask))
#tensor([[3., 2.], [4., 6.]])

################################################################################
# Conv1D examples

mat = torch.rand(3, 8)
cnn = nn.Conv1d(in_channels=3, out_channels=1, kernel_size=2)
print(cnn(mat).shape)
# torch.Size([1, 7])
cnn = nn.Conv1d(in_channels=3, out_channels=5, kernel_size=3)
print(cnn(mat).shape)
# torch.Size([5, 6])
cnn = nn.Conv1d(in_channels=2, out_channels=5, kernel_size=3)
print(cnn(mat).shape)
# Error: expected input to have 2 channels, but got 3 channels instead

################################################################################
