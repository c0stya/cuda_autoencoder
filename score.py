import numpy as np
import cudamat as cm
import cPickle

from tune import load_activations
from tune import load_targets

import sys

# where get 
# should we use CUDA for prediction. I guess no.

def load_model(filename):
    with open(filename) as handler:
        H, bh = cPickle.load(handler)
    return (H, bh)

def softmax(x):
    m = np.amax(x, axis=1)
    m.shape = (m.shape[0], 1)
    e = np.exp(x - m)

    d = np.sum(e, axis=1)
    d.shape = (d.shape[0], 1)

    return e / d

def sigmoid(x): 
    return 1.0 / (1.0 + np.exp(-x))

def predict(X, model, act):

    H, bh = model

    # forward pass
    n_layers = len(H)

    a = X
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
        print "Usage: %s <x> <y> <model>" % (sys.argv[0])
        sys.exit(1)

    X = load_activations(sys.argv[1])[:1000]
    Y = load_targets(sys.argv[2])[:1000]
    model = load_model(sys.argv[3])

    act = predict(X, model, act='softmax')
    y_pred = np.argmax(act, axis=1) 
    y_true = np.argmax(Y, axis=1)
    print y_true

    rate = ( y_true == y_pred ).astype(np.int)
    print rate

    print np.mean(rate)

