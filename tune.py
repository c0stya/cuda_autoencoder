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

def load_activations(filename):
    with open(filename) as h:
        X = np.load(h)
    return X

def load_targets(filename):
    with open(filename, 'rb') as h:
        Y = np.load(h)
    return Y

def load_model(filename):
    with open(filename) as handler:
        H, bh = cPickle.load(handler)
    return (H, bh)

def save_model(model, filename):
    with open(filename, 'wb') as handle:
        H, bh = model
        _H, _bh = [], []

        for p0,p1 in zip(H, bh):
            _H.append(p0.asarray())
            _bh.append(p1.asarray())

        cPickle.dump((_H,_bh), handle)

def grad(X, Y, act, params, grads, aux):

    H, bh = params
    _H, _bh = grads
    a, eh, loss = aux

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
            if act == 'logistic':
                cm.sigmoid(a[i+1])
            elif act == 'softmax':
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

    if act == 'logistic':
        cm.cross_entropy_bernoulli(Y, a[n_layers], target=loss)
    elif act == 'softmax':
        loss = cm.cross_entropy(Y, a[n_layers], target=loss)
    elif act == 'linear':
        a[-1].mult(a[-1], target=loss)

    return loss.sum()

def train(x, y, model, prev, args):
    n_items = x.shape[0]

    n_batches = n_items/batch_size

    if not prev:
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

    # loss
    loss = M(np.zeros((batch_size, n_out)))

    # each parameter and gradient is a list
    params = [H, bh]
    grads = [_H, _bh]
    aux = [a, eh, loss]

    X = cm.empty((batch_size, n_in))
    Y = cm.empty((batch_size, n_out))

    x_val = M(x[0:batch_size])
    y_val = M(y[0:batch_size])

    for epoch in range(n_epoch):
        err = []
        t0 = time.clock()

        v_err = grad(x_val, y_val, args.act_out,
            params, grads, aux)

        for i in range(1,n_batches):
            s = slice(i*batch_size, (i+1)*batch_size)

            X.overwrite(x[s])
            Y.overwrite(y[s])

            # apply momentum
            for layer in grads:
                for g in layer:
                    g.mult(momentum)

            cost = grad(X, Y, args.act_out,
                params, grads, aux)

            # update parameters 
            for _p,_g in zip(params, grads):
                for p,g in zip(_p,_g):
                    p.subtract_mult(g, mult=learning_rate/(batch_size))

            err.append(cost/batch_size)


        print "Epoch: %d, Loss: %.8f, VLoss: %.8f, Time: %.4fs" % (
                    epoch,
                    np.mean(err),
                    v_err/batch_size,
                    time.clock()-t0 )

        if args.out_params:
            save_model(params, args.out_params)

    return params

def load_params(filename):
    params = []
    with open(filename) as h:
        while True:
            try:
                H = np.load(h)
                O = np.load(h)
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
    parser.add_argument('-o', '--out_params', default='', help='continue with the model')
    parser.add_argument('-m', '--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('-l', '--learning_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('-b', '--batch_size', type=int, default=100, help='batch size')
    parser.add_argument('-x', '--hidden', type=int, default=100, help='number of hidden units')
    parser.add_argument('-e', '--epoch', type=int, default=10, help='number of epochs')
    parser.add_argument('-p', '--params', default='', help='continue with the model')
    parser.add_argument('-ao', '--act_out', default='linear', choices=['linear', 'logistic', 'softmax'], help='')
    parser.add_argument('-c', '--continue', dest='cont',
                action='store_true', default=False, help='continue with the model')

    args = parser.parse_args()

    prev = False        # marks that we need to create an additional layer
    if args.cont:
        print "Ignoring -x parameter"
        model = load_model(args.out)
        prev = True
    else:
        model = load_params(args.params)

    momentum = args.momentum
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    n_epoch = args.epoch

    X = load_activations(args.activations)
    Y = load_targets(args.targets)

    cm.cublas_init()

    model = train(X, Y, model, prev, args)

    # print "Saving model to:", args.out
    save_model(model, args.out_params)

