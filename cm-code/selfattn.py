#!/usr/bin/env python3

import numpy as np
from scipy.special import softmax

################################################################################

def selfattention(X, WK, WQ, WV, bidir=False):
  K = X @ WK
  Q = X @ WQ
  V = X @ WV
  print(f"K=\n{K}\n\nQ=\n{Q}\n\nV=\n{V}\n")

  scores = Q @ K.T
  print(f"scores=\n{scores}\n")
  if not bidir :
    mask = np.zeros(scores.shape)
    mask[np.triu_indices(scores.shape[0], 1)] = -np.inf
    scores = scores + mask
    print(f"mask=\n{mask}\n\nscores-masked=\n{scores}\n")

  alpha = softmax(scores, axis=1)
  print(f"alpha=\n{np.round(alpha,2)}\n")

  A = alpha @ V
  print(f"A=\n{np.round(A, 2)}\n") # 
  return A
  
  #Y2 = (V.T @ alpha.T).T
  #print("Y2={}\n".format(np.round(Y2,2)))
  
################################################################################

WK= np.array([[0, 0, 1], [1, 1, 0], [0, 1, 0], [1, 1, 0]]) # input 4 dim
WQ = np.array([[1, 0, 1], [1, 0, 0], [0, 0, 1], [0, 1, 1]]) # input 4 dim
WV = np.array([[0, 2, 0], [0, 3, 0], [1, 0, 3], [1, 1, 0]]) # input 4 dim

X = np.array([[1, 0, 1, 0],[0, 2, 0, 2],[1, 1, 1, 1]]) # input 4 dim
print(f"X=\n{X}\n")

selfattention(X, WK, WQ, WV, bidir=False)

#X = np.array([[1., 0., 1.],[0., 2., 0.],[1., 1., 1.]])

#WK= np.array([[0, 0, 1], [1, 1, 0], [0, 1, 0]])#, [1, 1, 0]])
#WQ = np.array([[1, 0, 1], [1, 0, 0], [0, 0, 1]])#, [0, 1, 1]])
#WV = np.array([[0, 2, 0], [0, 3, 0], [1, 0, 3]])#, [1, 1, 0]])











