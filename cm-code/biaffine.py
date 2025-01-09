#!/usr/bin/env python3

import numpy as np

n = 4
d = 10

Hhead = np.random.rand(n,d) # head vectors as rows
Hdep  = np.random.rand(n,d) # dep  vectors as rows
Uarc  = np.random.rand(d,d) # biaffine intermediate matrix
uarc  = np.random.rand(d,1) # biaffine bias

Sarc_original = Hhead @ Uarc @ Hdep.T + Hhead @ uarc
#print(Sarc_original)

ubias = np.vstack((Uarc.T, uarc.T)) # Uarc transposition not in paper, but not important since learned
Hdepext = np.hstack((Hdep,np.ones((n, 1))))
Sarc_simple = ((Hdepext @ ubias) @ Hhead.T).T # final transpose: dependents as columns
#print(Sarc_simple)

print(f"Original and simplified versions identical: {np.allclose(Sarc_original, Sarc_simple)}") # quality up to tolerance
