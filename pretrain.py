import numpy as np
import cudamat as cm
import time

from cudamat import CUDAMatrix as M

from utils import one_hot_dec, ALPH_rev

n_epoch = 10
n_hidden = 100
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

def load_model(filename='params.bin'):
    with open(filename, 'rb') as h:
        H = np.load(h)
        bh = np.load(h)
        bo = np.load(h)

    return [H,bh,bo]

def add_noise(data, alph):
    prob = [1-noise_rate] + [noise_rate/(alph-1)] * (alph-1)

    noise = np.random.choice( np.arange(alph), data.shape, p=prob )
    data = (data + noise) % alph

    return data

def save_model(model, filename):
    with open(filename, 'wb') as h:
        for p in model:
            # load from GPU memory
            np.save(h, p.asarray())

def activate(X, params, a):

    batch_size = X.shape[0]
    H, bh, bo = params

    # a = f( x*H + bh )

    X.dot(H, target=a)
    a.add_row_vec(bh)
    cm.sigmoid(a)

    return a

def grad(X, Y, params, grads, aux):

    H, bh, bo = params
    _H, _bh, _bo = grads

    a, z, eh, eo, loss, ones = aux

    _H.assign(0.0)
    _bh.assign(0.0)
    _bo.assign(0.0)

    # watch out for the redundand accumulations

    ### FORWARD PASS ###

    # a = tanh( x*H + bh )

    X.dot(H, target=a)
    a.add_row_vec(bh)
    cm.sigmoid(a)

    # b = sigm( a*H_prime + bo )

    a.dot(H.T, target=z)
    z.add_row_vec(bo)
    cm.sigmoid(z)

    ### BACKWARD PASS ###

    # eo = z - y

    z.subtract(Y, target=eo)

    # eh = tanh'(a) x ( eo * H_prime )
    eo.dot(H, target = eh)
    eh.apply_logistic_deriv(a)

    ### COMPUTE GRADIENTS ###

    _H.add_dot(eo.T, a)
    _H.add_dot(X.T, eh)

    _bo.add_sums(eo, axis=0)
    _bh.add_sums(eh, axis=0)

    ### COMPUTE ERROR ###
    cm.cross_entropy_bernoulli(Y, z, target=loss)
    err = loss.sum()

    return err

def _check_grad():
    batch_size = 5
    n_hidden = 100
    n_in = n_out = 20
    EPS = 0.0001

    H = np.random.uniform( -0.1, 0.1, size=(n_in, n_hidden))
    bh = np.zeros((1,n_hidden))
    bo = np.zeros((1,n_out))

    eps = np.zeros_like(H)
    eps[0,0] = EPS
    H0 = H + eps
    H1 = H - eps

    H = M(H)
    H0 = M(H0)
    H1 = M(H1)
    bh = M(bh)
    bo = M(bo)
    X = cm.empty((batch_size, n_in))

    _H = M(np.zeros(H.shape))
    _bh = M(np.zeros(bh.shape))
    _bo = M(np.zeros(bo.shape))
    grads = [_H, _bh, _bo]

    X = M(np.random.binomial(1, 0.5, size=(batch_size,n_in)))

    a = M(np.zeros((batch_size, n_hidden)))
    z = cm.empty((batch_size, n_out))

    eh = M(np.zeros((batch_size, n_hidden)))
    eo = M(np.zeros((batch_size, n_out)))

    aux = [a, z, eh, eo]

    loss0 = grad(X, X, (H0, bh, bo), grads, aux)
    loss1 = grad(X, X, (H1, bh, bo), grads, aux)

    grad(X, X, (H, bh, bo), grads, aux)
    print 'Grad:\t\t%.6f' % (_H.asarray()[0,0])
    print 'Grad approx:\t%.6f, (%f, %f)' % ( (loss0 - loss1)/(2*EPS),  loss0, loss1)


def pretrain(data, n_hidden, model=None, filename=None, act_file=None):
    valid_size = 1000
    n_items = data.shape[0]
    n_in = n_out = data.shape[1]
    n_batches = n_items/batch_size # leave one for validation

    print "Pretraining, scale:", data.shape

    if model:

        # check model consistency with the current dataset

        H,bh,bo = model
        assert (H.shape[0] == n_in), 'Input matrix shape mismatch'
        assert (H.shape[0] == n_out), 'Output matrix shape mismatch'
        assert (bo.shape[1] == n_out), 'Bias vector shape mismatch'

        n_hidden = H.shape[1]

    else:

        # initialize a new model

        H = np.random.normal( scale=init_scale, size=(n_in, n_hidden))
        bh = np.zeros((1,n_hidden))
        bo = np.zeros((1,n_out))

    H = M(H)
    bh = M(bh)
    bo = M(bo)

    X = cm.empty((batch_size, n_in))
    Y = cm.empty((batch_size, n_out))

    params = [H, bh, bo]

    _H = M(np.zeros(H.shape))
    _bh = M(np.zeros(bh.shape))
    _bo = M(np.zeros(bo.shape))

    grads = [_H, _bh, _bo]

    a = M(np.zeros((batch_size, n_hidden)))
    z = cm.empty((batch_size, n_out))

    eh = M(np.zeros((batch_size, n_hidden)))
    eo = M(np.zeros((batch_size, n_out)))

    loss = M(np.zeros(Y.shape))
    ones = np.ones((1,batch_size*n_out), dtype=np.float32)

    aux = [a, z, eh, eo, loss, ones]

    X_val = M(data[(n_batches-1)*batch_size: n_batches*batch_size])

    ### TRAINING ###

    for epoch in range(n_epoch):
        err = []
        upd = [0] * len(params)

        t0 = time.clock()

        for i in range(n_batches-1):
            s = slice(i*batch_size, (i+1)*batch_size)
            X.overwrite(data[s])

            # apply momentum
            for g in grads:
                g.mult(momentum)

            cost = grad(X, X, params, grads, aux)

            # update parameters 
            for p,g in zip(params, grads):
                p.subtract_mult(g, mult=learning_rate/(batch_size))

            err.append(cost/(batch_size))

        # measure the reconstruction error
        v_err = grad(X_val, X_val, params, grads, aux)

        print "Epoch: %d, Loss: %.8f, VLoss: %.8f, Time: %.4fs" % (
                    epoch, np.mean( err ),
                    v_err/batch_size,
                    time.clock()-t0 )
        if filename:
            save_model(params, filename)    # it's quite fast

    ### STORE ACTICATION VECTORS ### 

    if act_file:
        A = np.zeros((data.shape[0], n_hidden))
        for i in range(n_batches):
            s = slice(i*batch_size, (i+1)*batch_size)
            X.overwrite(data[s])
            activate(X, params, a)
            A[s] = a.asarray()

        with open(act_file, 'wb') as h:
            print "Saving the activation vectors to: %s, shape: %s, %s " % \
                    (act_file, A.shape[0], A.shape[1])
            np.save(h, A)

    return params

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--filename', default='corpus.bin', help='data file')
    parser.add_argument('-o', '--out', default='params.bin', help='file to store parameters')
    parser.add_argument('-a', '--activations', default=None, help='file to store activation vectors')
    parser.add_argument('-m', '--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('-l', '--learning_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('-b', '--batch_size', type=int, default=100, help='batch size')
    parser.add_argument('-x', '--hidden', type=int, default=100, help='number of hidden units')
    parser.add_argument('-e', '--epoch', type=int, default=10, help='number of epochs')
    parser.add_argument('-c', '--continue', dest='cont',
                action='store_true', default=False, help='continue with the model')
    parser.add_argument('-n', '--noise_rate', type=float, default=0.01, help='specify curruption rate')

    args = parser.parse_args()

    momentum = args.momentum
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    n_hidden = args.hidden
    n_epoch = args.epoch
    noise_rate = args.noise_rate
    act_file = args.activations

    X = load_layer(args.filename)

    # cudamat fix for large matrix summations
    if cm.MAX_ONES < batch_size*X.shape[1]:
        print "Warning: extending cudamat 'ones' size"
        cm.MAX_ONES = batch_size*X.shape[1]

    cm.cublas_init()

    if DEBUG:
        _check_grad()
    else:
        prev_model = None
        if args.cont:
            print "Ignoring -x parameter"
            prev_model = load_model(args.out)

        model = pretrain(X, n_hidden, prev_model, args.out, act_file)

        print "Saving model to:", args.out
        save_model(model, args.out)
