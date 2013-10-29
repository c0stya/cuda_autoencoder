import cPickle
import numpy as np

cdir = './cifar-10-batches-py/'

def one_hot_enc(x):
    x_shape = len(x)
    y = np.zeros((x_shape, np.max(x)+1)).astype(np.int8)
    for i in range(x_shape):
        y[i,x[i]] = 1
    return y

if __name__ == '__main__':
    with open(cdir + 'data_batch_1','rb') as h:
        b1 = cPickle.load(h)
        one_hot_enc(b1['labels'])
    with open(cdir + 'data_batch_2','rb') as h:
        b2 = cPickle.load(h)
    with open(cdir + 'data_batch_3','rb') as h:
        b3 = cPickle.load(h)
    with open(cdir + 'data_batch_4','rb') as h:
        b4 = cPickle.load(h)
    with open(cdir + 'data_batch_5','rb') as h:
        b5 = cPickle.load(h)

    data = np.vstack([
        b1['data'],
        b2['data'],
        b3['data'],
        b4['data'],
        b5['data']
    ])

    labels = np.concatenate([
        b1['labels'],
        b2['labels'],
        b3['labels'],
        b4['labels'],
        b5['labels']
    ])

    l0 = data/255.0

    with open('l0.bin', 'wb') as h:
        np.save(h, l0)

    with open('labels.bin', 'wb') as h:
        np.save(h, one_hot_enc(labels))

