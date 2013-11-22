import numpy as np
import cudamat as cm
import time
from cudamat import CUDAMatrix as M

n_epoch = 10
n_hidden = 100
init_scale = 1
batch_size = 1000
noise_rate = 0.05

learning_rate = 0.01
momentum = 0.9

def load_layer(filename):
    with open(filename) as h:
        X = np.load(h)
    return X

def load_model(filename='params.bin'):
    with open(filename, 'rb') as h:
        H = np.load(h)
        O = np.load(h)      # use untying weights
        bh = np.load(h)
        bo = np.load(h)

    return [H,O,bh,bo]

def save_model(model, filename):
    with open(filename, 'wb') as h:
        for p in model:
            # load from GPU memory
            np.save(h, p.asarray())

def activate(X, params, a):

    batch_size = X.shape[0]
    H, O, bh, bo = params

    # a = f( x*H + bh )

    X.dot(H, target=a)
    a.add_row_vec(bh)
    cm.sigmoid(a)

    return a

def grad(X, Y, act_type, rho, params, grads, aux):

    H, O, bh, bo = params
    _H, _O, _bh, _bo = grads

    a, z, eh, eo, loss, s, s_m = aux

    _H.assign(0.0)
    _O.assign(0.0)
    _bh.assign(0.0)
    _bo.assign(0.0)

    # watch out for the redundand accumulations

    ### FORWARD PASS ###

    # a = tanh( x*H + bh )

    X.dot(H, target=a)
    a.add_row_vec(bh)
    cm.sigmoid(a)

    # b = sigm( a*O + bo )
    #a.dot(H.T, target=z)   # use tyied weights

    a.dot(O, target=z)
    z.add_row_vec(bo)

    if act_type == 'logistic':
        cm.sigmoid(z)          # DEBUG

    ### BACKWARD PASS ###

    # eo = z - y

    z.subtract(Y, target=eo)

    # eh = sigmoid'(a) x ( eo * O + (rho-1)/(s-1) - rho/s )

    eo.dot(O.T, target = eh)

    # the following needs to be verified
    if rho > 0:
        a.sum(axis=0, target=s)
        s.mult(1.0/a.shape[0])        # normalize by batch_size
        s.reciprocal()
        s.mult(rho)

        a.sum(axis=0, target=s_m)   # TODO: remove this redundancy
        s_m.mult(1.0/a.shape[0])        # normalize by batch_size
        s_m.subtract(1.0)
        s_m.reciprocal()
        s_m.mult(rho-1)
        s.subtract(s_m)

        eh.add_row_mult(s, -1.0)
    
    eh.apply_logistic_deriv(a)

    ### COMPUTE GRADIENTS ###

    _O.add_dot(a.T, eo)
    _H.add_dot(X.T, eh)

    _bo.add_sums(eo, axis=0)
    _bh.add_sums(eh, axis=0)

    ### COMPUTE ERROR ###
    if act_type == 'logistic':
        cm.cross_entropy_bernoulli(Y, z, target=loss)
    elif act_type == 'linear':
        eo.mult(eo, target=loss) #loss.add_mult(eo, eo)   # DEBUG
    else:
        raise ValueError("Activation function '%s' is unknown" % args.act_type)

    err = loss.sum()

    return err

def pretrain(data, n_hidden, args, model=None):
    n_items = data.shape[0]
    n_in = n_out = data.shape[1]
    n_batches = n_items/batch_size # leave one for validation

    if model:

        # check model consistency with the current dataset

        H,O,bh,bo = model
        assert (H.shape[0] == n_in), 'Input matrix shape mismatch'
        assert (H.shape[0] == n_out), 'Output matrix shape mismatch'
        assert (bo.shape[1] == n_out), 'Bias vector shape mismatch'

        n_hidden = H.shape[1]

    else:

        # initialize a new model

        #H = np.random.normal( scale=init_scale, size=(n_in, n_hidden))
        #O = np.random.normal( scale=init_scale, size=(n_hidden, n_out))
        interv = np.sqrt(6.0/(n_in+n_out+1))
        H = np.random.normal( -interv, interv, size=(n_in, n_hidden))
        O = np.random.normal( -interv, interv, size=(n_hidden, n_out))
        bh = np.zeros((1,n_hidden))
        bo = np.zeros((1,n_out))

    H = M(H)
    O = M(O)
    bh = M(bh)
    bo = M(bo)

    X = cm.empty((batch_size, n_in))
    Y = cm.empty((batch_size, n_out))

    params = [H, O, bh, bo]

    _H = M(np.zeros(H.shape))
    _O = M(np.zeros(O.shape))
    _bh = M(np.zeros(bh.shape))
    _bo = M(np.zeros(bo.shape))

    grads = [_H, _O, _bh, _bo]

    a = M(np.zeros((batch_size, n_hidden)))
    z = cm.empty((batch_size, n_out))

    eh = M(np.zeros((batch_size, n_hidden)))
    eo = M(np.zeros((batch_size, n_out)))

    loss = M(np.zeros(Y.shape))

    # terms for calculting sparse penalty
    s = cm.empty((1, n_hidden))
    s_m = cm.empty((1, n_hidden))

    aux = [a, z, eh, eo, loss, s, s_m]

    # whiten, DEBUG!
    # from scipy.cluster.vq import whiten 
    # data = whiten(data)

    X_val = M(data[(n_batches-1)*batch_size: n_batches*batch_size])

    ### TRAINING ###

    for epoch in range(n_epoch):
        err = []

        t0 = time.clock()

        for i in range(n_batches-1):
            s = slice(i*batch_size, (i+1)*batch_size)
            X.overwrite(data[s])

            if args.noise_rate > 0:
                Y.overwrite(data[s])
                Y.dropout(args.noise_rate)
            else:
                Y = X

            # apply momentum
            for g in grads:
                g.mult(momentum)

            cost = grad(X, Y, args.act_type, args.sparse, params, grads, aux)

            # update parameters 
            for p,g in zip(params, grads):
                p.subtract_mult(g, mult=learning_rate) #/(batch_size))

            err.append(cost/(batch_size))

        # measure the reconstruction error
        v_err = grad(X_val, X_val, args.act_type, args.sparse, params, grads, aux)

        print "Epoch: %d, Loss: %.8f, VLoss: %.8f, Time: %.4fs" % (
                    epoch, np.mean( err ),
                    v_err/batch_size,
                    time.clock()-t0 )
        if args.out:
            save_model(params, args.out)

    ### STORE ACTICATION VECTORS ### 

    if args.activations:
        A = np.zeros((data.shape[0], n_hidden))
        for i in range(n_batches):
            s = slice(i*batch_size, (i+1)*batch_size)
            X.overwrite(data[s])
            activate(X, params, a)
            A[s] = a.asarray()

        with open(args.activations, 'wb') as h:
            print "Saving the activation vectors to: %s, shape: %s, %s " % \
                    (args.activations, A.shape[0], A.shape[1])
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
    parser.add_argument('-n', '--noise_rate', type=float, default=0.0, help='specify the curruption rate')
    parser.add_argument('-t', '--act_type', default='linear', choices=['linear', 'logistic'], help='')
    parser.add_argument('-s', '--sparse', type=float, default=0.0, help='add sparse penalty')

    args = parser.parse_args()

    momentum = args.momentum
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    n_hidden = args.hidden
    n_epoch = args.epoch
    noise_rate = args.noise_rate

    X = load_layer(args.filename)

    # cudamat fix for large matrix summations
    if cm.MAX_ONES < batch_size*X.shape[1]:
        print "Warning: extending cudamat 'ones' size"
        cm.MAX_ONES = batch_size*X.shape[1]

    cm.cublas_init()
    M.init_random()

    prev_model = None
    if args.cont:
        print "Ignoring -x parameter"
        prev_model = load_model(args.out)

    M.init_random()

    model = pretrain(X, n_hidden, args, prev_model)

    print "Saving model to:", args.out
    save_model(model, args.out)

