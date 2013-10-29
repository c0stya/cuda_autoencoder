import numpy as np
import cudamat as cm
import time

from cudamat import CUDAMatrix as M
import cPickle

n_epoch = 10
n_hidden = 100
init_scale = 0.01
batch_size = 1000

learning_rate = 0.01
momentum = 0.9

DEBUG=False

def load_layer(filename):
    with open(filename) as h:
        X = np.load(h)
    return X

def load_labels(filename):
    with open(filename, 'rb') as h:
        Y = np.load(h)

    Y.shape = (Y.shape[0], 1)
    return Y

def save_model(model, filename):
    with open(filename, 'wb') as handl:
        H, bh = model
        _H, _bh = [], []

        for p0,p1 in zip(H, bh):
            _H.append(p0.asarray)
            _bh.append(p1.asarray)

        cPickle.dump(handl, model)

def grad(X, Y, act, out_act, params, grads, aux):

    H, bh = params
    _H, _bh = grads
    a, eh = aux

    # forward pass
    a[0].assign(X)
    n_layers = len(eh)

    for i in range(n_layers):
        # a = sigmoid( ap*H + bh )
        a[i].dot(H[i], target = a[i+1])
        a[i+1].add_row_vec(bh[i])

        if i < n_layers-1:
            cm.sigmoid(a[i+1])
        else:
            # last layer
            if act_out == 'logistic':
                cm.sigmoid(a[i+1])
            elif act_out == 'softmax':
                a_t = a[i+1].transpose()
                cm.softmax(a_t)
                a_t.transpose(target=a[i+1])
                a_t.free_device_memory()
            else:
                pass

    # backward pass

    # compute error term of the last layer
    a[-1].subtract(Y, target=eh[-1])

    # check the following
    for i in range(n_layers-1, -1, -1):

        # compute derivatives
        _H[i].assign(0.0)
        _H[i].add_dot(a[i].T, eh[i])
        eh[i].sum(axis=0, target=_bh[i])

        # compute error term for the previous layer
        if i > 0:
            # eh = sigmoid'(a) x ( ehp*H' )
            eh[i].dot(H[i].T, target=eh[i-1])
            eh[i-1].apply_logistic_deriv(a[i])

    if out_act == 'logistic':
        cm.cross_entropy_bernoulli(Y, a[n_layers], target=loss)
    elif out_act == 'softmax':
        loss = cm.cross_entropy(Y, a[n_layers], target=loss)
    elif out_act == 'linear':
        a[-1].mult(a[-1], target=loss)

    return loss.sum()

def train(x, y, model, args):
    n_items = x.shape[0]

    n_batches = n_items/batch_size

    H, bh, a, eh = [], [], [], []
    _H, _bh = [], []

    n_in = x.shape[1]
    n_out = y.shape[1]

    lh, _ = model[-1]
    l_H = np.random.normal( scale=0.1, size=(lh.shape[1], n_out))
    l_bh = np.zeros((1,n_out))

    model.append( (l_H, l_bh) )

    for p_H,p_bh in model:
        H.append(M(p_H))
        bh.append(M(p_bh))
        _H.append(cm.empty(p_H.shape))
        _bh.append(cm.empty(p_bh.shape))

        # allocate space for the activation vectors
        a.append(cm.empty((batch_size, p_H.shape[0])))

        # allocate space for the error vectors
        eh.append(cm.empty((batch_size, p_H.shape[1])))

    # last layer
    a.append(cm.empty((batch_size, n_out)))

    # each parameter and gradient is a list
    params = [H, bh]
    grads = [_H, _bh]
    aux = [a, eh]

    X = cm.empty((batch_size, n_in))
    Y = cm.empty((batch_size, n_out))

    x_val = M(x[0:batch_size])
    y_val = M(y[0:batch_size])

    for epoch in range(n_epoch):
        err = 0.0
        t0 = time.clock()

        v_err = grad(x_val, y_val,
            args.act_middle, args.act_out,
            params, grads, aux)

        for i in range(1,n_batches):
            s = slice(i*batch_size, (i+1)*batch_size)

            X.overwrite(x[s])
            Y.overwrite(y[s])

            # apply momentum
            '''
            for layer in grads:
                for g in layer:
                    g.mult(momentum)
            '''

            lss = grad(X, Y,
                args.act_middle, args.act_out,
                params, grads, aux)

            # update parameters 
            for _p,_g in zip(params, grads):
                for p,g in zip(_p,_g):
                    p.subtract_mult(g, mult=learning_rate/(batch_size))

            err.append(lss/batch_size)


        print "Epoch: %d, Loss: %.8f, VLoss: %.8f, Time: %.4fs" % (
                    epoch,
                    np.mean(err),
                    v_err/batch_size,
                    time.clock()-t0 )

        #if filename:
        #   save_model(params, filename)

    return params

def load_params(filename):
    params = []
    with open(filename) as h:
        while True:
            try:
                H = np.load(h)
                bh = np.load(h)
                bo = np.load(h)
                params.append((H, bh))
            except IOError:
                break
    return params

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-a', '--activations', required=True, help='data file')
    parser.add_argument('-t', '--targets', required=True, help='data file')
    parser.add_argument('-m', '--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('-l', '--learning_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('-b', '--batch_size', type=int, default=100, help='batch size')
    parser.add_argument('-x', '--hidden', type=int, default=100, help='number of hidden units')
    parser.add_argument('-e', '--epoch', type=int, default=10, help='number of epochs')
    parser.add_argument('-p', '--params', default='', help='continue with the model')
    parser.add_argument('-am', '--act_middle', default='linear', choices=['linear', 'logistic'], help='')
    parser.add_argument('-ao', '--act_out', default='linear', choices=['linear', 'logistic', 'softmax'], help='')

    args = parser.parse_args()

    model = load_params(args.params)

    momentum = args.momentum
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    n_epoch = args.epoch

    X = load_layer(args.activations)
    Y = load_labels(args.targets)

    part = 1000 # size of the test part

    cm.cublas_init()

    prev_model = None

    model = train(X, Y, model, args)

    # print "Saving model to:", args.out
    # save_model(model, args.out)

