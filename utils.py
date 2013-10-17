import numpy as np
from collections import defaultdict

# space should be the first symbol
alph = " abcdefghijklmnopqrstuvwxyz!?-.:()='"

ALPH = dict((c,i) for i,c in enumerate(alph))
ALPH = defaultdict( lambda: 0, ALPH)

ALPH_rev = dict((v,k) for k,v in ALPH.iteritems())

def translate(string):
    return [ALPH[t] for t in string]

def one_hot_enc(index, n):
    v = np.zeros(n)
    v[index] = 1
    return v

def one_hot_dec(vector):
    return np.argmax(vector)
