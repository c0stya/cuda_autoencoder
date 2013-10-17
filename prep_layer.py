"""
Preprocess dataset for sent140 corpus.

"""
import numpy as np
import sys

from utils import alph, translate, ALPH_rev

def encode(data, alph):
    m, n = data.shape # batch_size x n_steps

    enc = np.zeros((m,n*alph), dtype=np.uint8)

    for i in xrange(m):
        for j in xrange(n):
            enc[i, j*alph + data[i,j]] = 1

    return enc

if __name__ == '__main__':

    if len(sys.argv) < 3:
        print "Usage: %s <input_corpus> <output>" % (sys.argv[0])
        sys.exit(1)

    inp = sys.argv[1]
    out = sys.argv[2]

    with open(inp) as h:
        alph = np.load(h)[0]
        X = np.load(h)
        Y = np.load(h)

    # this could be memory-critical part
    enc_X = encode(X, alph)

    with open(out, 'wb') as h:
        np.save(h, enc_X) # indicates the layer

