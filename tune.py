import numpy as np
import cudamat as cm
import time

from cudamat import CUDAMatrix as M

from utils import one_hot_dec, ALPH_rev

n_epoch = 10
init_scale = 0.1
batch_size = 1000
noise_rate = 0.05

DEBUG = False

learning_rate = 0.01
momentum = 0.9

def load_layer(filename):
    with open(filename) as h:
        X = np.load(h)
    return X

def load_labels(filename):
    with open(filename, 'rb') as h:
        alph = np.load(h)[0]
        X = np.load(h)
        Y = np.load(h)

    Y.shape = (Y.shape[0], 1)
    return Y

def save_model(model, filename):
    with open(filename, 'wb') as h:
        for p in model:
            # load from GPU memory
            np.save(h, p.asarray())

def predict(X, params):

    err = 0.0
    H, bo = params

    # a = f( x*H + bh )

    a = X.dot(H)
    a.add_row_vec(bo)
    cm.sigmoid(a)

    y_pred = ((a.asarray() > 0.5).astype(int))

    del a

    return y_pred

def accuracy(y_true, y_pred):
    return (y_true == y_pred).sum()*1.0/(y_true.shape[0])

def grad(X, Y, params, grads, aux):

    err = 0.0

    H, bo = params
    _H, _bo = grads

    a, eo, loss = aux

    _H.assign(0.0)
    _bo.assign(0.0)

    # watch out for the redundand accumulations

    ### FORWARD PASS ###

    # a = f( x*H + bh )

    X.dot(H, target=a)
    a.add_row_vec(bo)
    cm.sigmoid(a)

    ### BACKWARD PASS ###

    # eo = a - y

    a.subtract(Y, target=eo)

    ### COMPUTE GRADIENTS ###

    _H.add_dot(X.T, eo)
    _bo.add_sums(eo, axis=0)

    ### COMPUTE ERROR ###
    cm.cross_entropy_bernoulli(Y, a, target=loss)

    cs = loss.sum(axis=0)
    rs = cs.sum(axis=1)
    err = np.sum(rs.asarray())

    return err

def train(hX, hY, filename=None):

    n_items = hX.shape[0]
    n_in = hX.shape[1]
    n_out = hY.shape[1]
    n_batches = (n_items-part)/batch_size

    # initialize a new model
    H = np.random.normal( scale=init_scale, size=(n_in, n_out))
    bo = np.zeros((1,n_out))

    H = M(H)
    bo = M(bo)

    X = cm.empty((batch_size, n_in))
    Y = cm.empty((batch_size, n_out))

    params = [H, bo]

    _H = M(np.zeros(H.shape))
    _bo = M(np.zeros(bo.shape))

    grads = [_H, _bo]

    a = M(np.zeros((batch_size, n_out)))
    eo = M(np.zeros((batch_size, n_out)))

    loss = M(np.zeros(Y.shape))

    aux = [a, eo, loss]

    x_train, y_train = hX[:-part], hY[:-part]
    x_test, y_test   = M(hX[-part:]), hY[-part:]

    ### TRAINING ###

    for epoch in range(n_epoch):
        err = []
        upd = [0] * len(params)

        t0 = time.clock()

        for i in range(n_batches):
            s = slice(i*batch_size, (i+1)*batch_size)
            X.overwrite(x_train[s])
            Y.overwrite(y_train[s])

            # apply momentum
            for g in grads:
                g.mult(momentum)

            cost = grad(X, Y, params, grads, aux)

            # update parameters 
            for p,g in zip(params, grads):
                p.subtract_mult(g, mult=learning_rate/(batch_size))

            err.append(cost/(batch_size))

        # trying the test set
        y_pred = predict(x_test, params)

        print "Epoch: %d, Loss: %.8f, Acc: %.4f Time: %.4fs" % (
                    epoch, np.mean( err ),
                    accuracy(y_test, y_pred),
                    time.clock()-t0)
        if filename:
            save_model(params, filename)    # it's quite fast

    return params

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--filename', required=True, help='activation vectors file')
    parser.add_argument('-d', '--datafile', required=True, help='data file')
    parser.add_argument('-o', '--out', default='params.bin', help='file to store parameters')
    parser.add_argument('-m', '--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('-l', '--learning_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('-b', '--batch_size', type=int, default=100, help='batch size')
    parser.add_argument('-e', '--epoch', type=int, default=10, help='number of epochs')
    parser.add_argument('-c', '--continue_using_model', default='', help='continue with the model')
    parser.add_argument('-n', '--noise_rate', type=float, default=0.01, help='specify curruption rate')

    args = parser.parse_args()

    momentum = args.momentum
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    n_epoch = args.epoch
    noise_rate = args.noise_rate

    X = load_layer(args.filename)
    Y = load_labels(args.datafile)

    part = 1000 # size of the test part

    cm.cublas_init()

    if DEBUG:
        _check_grad()
    else:
        prev_model = None
        if args.continue_using_model:
            print "Ignoring -x parameter"
            prev_model = load_model(args.continue_using_model)

        model = train(X, Y, args.out)

        print "Saving model to:", args.out
        save_model(model, args.out)

