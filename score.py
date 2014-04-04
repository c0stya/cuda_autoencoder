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

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-a', '--activations', required=True, help='data file')
    parser.add_argument('-t', '--targets', required=True, help='data file')
    parser.add_argument('-p', '--params', default='', help='parameters of the model')
    parser.add_argument('-ah', '--act_hidden', default='sigmoid', choices=['sigmoid', 'tanh'], help='hidden layers activation function')
    parser.add_argument('-ao', '--act_out', default='logistic', choices=['linear', 'logistic', 'softmax'], help='output layer activation function')

    args = parser.parse_args()
    model = load_model(args.params)

    if len(sys.argv) < 3:
        print "Usage: %s <x> <y> <model>" % (sys.argv[0])
        sys.exit(1)

    X = load_activations(args.activations)
    Y = load_targets(args.targets)

    activ = args.act_out

    a = predict(X, model, act=activ)

    if activ == 'logistic':
        Y = Y.ravel()
        y_pred = (a.ravel() > 0.5).astype(int)
        print a.ravel()
        print Y
        print y_pred
        rate = (y_pred == Y).astype(int)
        print 'Accuracy:', np.mean(rate)
    elif activ == 'softmax':
        y_pred = np.amax(a, axis=1)
        rate = (y_pred == Y).astype(int)
        print 'Accuracy:', np.mean(rate)
    else:
        y_pred = a.ravel()
        print 'L2 loss:', np.mean ( np.power(Y - y_pred, 2), axis=0 )

