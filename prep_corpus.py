"""
Preprocess dataset for sent140 corpus.

"""
import csv
import numpy as np
import cPickle as pickle
import re
import sys

from utils import alph, translate, ALPH_rev

FILENAME = './sent140.csv'
LENGTH = 140

def normalize(string):
    '''Remove out-of-alphabet symbols and align to LENGTH '''
    s = string.ljust(LENGTH)
    return s

def get_dataset(filename, limit=0):
    X,Y = [], []
    with open(filename) as h:
        dataset = csv.reader(h)
        i = 0
        for row in dataset:
            y = 0 if int(row[0]) == 0 else 1
            Y.append(y)
            x = translate(normalize(row[5][:140]))
            X.append(x)
            if limit !=0 and i > limit:
                break
            i+=1
    return X,Y

if __name__ == '__main__':

    if len(sys.argv) < 3:
        print "Usage: %s <input_corpus> <output>" % (sys.argv[0])
        sys.exit(1)

    inp = sys.argv[1]
    out = sys.argv[2]

    X,Y = get_dataset(filename=inp, limit=1000000)

    with open(out, 'wb') as h:
        np.save(h, [len(alph)])
        np.save(h, np.array(X, dtype=np.int8))
        np.save(h, np.array(Y, dtype=np.int8))

