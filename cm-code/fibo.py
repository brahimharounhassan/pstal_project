#!/usr/bin/env python3

n_appels_rec = n_appels_dynprog = 0

def fiboRec(n):
  global n_appels_rec
  n_appels_rec += 1
  if n <= 1:
    return n
  else:
    return fiboRec(n-1) + fiboRec(n-2)

################################################################################

def fiboDynProg(n):
  global n_appels_dynprog
  fib = [0] * (n+1)
  fib[1] = 1
  n_appels_dynprog += 2
  for i in range(2,n+1):
    n_appels_dynprog += 1
    fib[i] = fib[i-1] + fib[i-2]
  return fib[n]

for n in range(1,20):
  n_appels_rec = n_appels_dynprog = 0
  resultRec = fiboRec(n)
  resultDynProg = fiboDynProg(n)
  print(f"n={n:2} fib_rec(n)={resultRec:4}, n_appels_rec={n_appels_rec:5} fib_dp(n)={resultDynProg:4}, n_appels_dp={n_appels_dynprog:5}")
