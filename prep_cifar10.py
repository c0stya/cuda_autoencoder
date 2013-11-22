import cPickle
import numpy as np

cdir = './cifar-10-batches-py/'

def pool(X, m, n, k, l, C):
    '''
    Perfrom edge-aware pooling with stride equal 1.
    Return 'm-n+1' by 'n-m+1' matrix.

    Note: It's not super fast. But since it's run-once code
    let it be that way for now.

    '''
    for i in xrange(k):
        for j in xrange(l):
            C[i,j] = np.sum(X[i:i+m, j:j+n])
    return C

def conv(x, m, n):

    k,l  = 32-m+1, 32-n+1

    n_img = x.shape[0]
    x.shape = (n_img, 3, 32, 32)

    xconv = np.zeros((n_img, 3, k, l), dtype=np.uint)
    C = np.zeros((k, l), dtype=np.uint)

    for i in range(n_img):
        for c in range(3):  # colors
            xconv[i,c] = pool(x[i,c], m, n, k, l, C)

    xconv.shape = (n_img, 3*k*l)

    return xconv/(m*n)

def one_hot_enc(x):
    x_shape = len(x)
    y = np.zeros((x_shape, np.max(x)+1)).astype(np.int8)
    for i in range(x_shape):
        y[i,x[i]] = 1
    return y

if __name__ == '__main__':
    with open(cdir + 'data_batch_1','rb') as h:
        b1 = cPickle.load(h)
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

    l0 = conv(data, 5, 5)/255.0
    print l0.shape

    with open('l0.bin', 'wb') as h:
        np.save(h, l0)

    with open('labels.bin', 'wb') as h:
        np.save(h, one_hot_enc(labels))

