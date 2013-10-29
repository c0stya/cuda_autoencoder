import numpy as np
import cudamat as cm

from tune import load_activations
from tune import load_targets

import sys

# where get 
# should we use CUDA for prediction. I guess no.

def load_model(filename):
    with open(filename) as handl:
        H, bh = handl.load(h)

def predict(X, Y, model, act):

    H, bh = model

    # forward pass
    n_layers = len(H)

    for i in range(n_layers-1):
        # a = sigmoid( ap*H + bh )
        a = sigmoid( a.dot(H[i]) + bh[i] )

    # last layer
    a = a.dot(H[i+1]) + bh[i+1]

    if act == 'logistic':
        a = sigmoid(a)
    elif act == 'softmax':
        a = softmax(a)
    else:
        pass

    return a

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print "Usage: %s <x> <y> <model>"

    X = load_activations(sys.argv[1])
    Y = load_targets(sys.argv[2])
    model = load_model(sys.argv[3])
