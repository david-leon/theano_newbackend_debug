# coding:utf-8
# <Descriptions>
# Created   :  mm, dd, yyyy
# Revised   :  mm, dd, yyyy
# All rights reserved
#------------------------------------------------------------------------------------------------
__author__ = 'dawei.leng'
import os
os.environ['THEANO_FLAGS'] = "floatX=float32, optimizer=fast_run, warn_float64='raise'"

import theano
from lasagne.layers import DenseLayer, MaxPool2DLayer, InputLayer, \
                           ExpressionLayer, Pool1DLayer, \
                           get_output, get_all_params
from lasagne.nonlinearities import *
from lasagne.updates import *

from lasagne_ext.objectives import ctc_cost_logscale, ctc_best_path_decode, ctc_CER
from lasagne_ext.nonlinearities import *
from lasagne_ext.utils import get_layer_by_name

def filter_merge(x):
    x1 = x.dimshuffle((0, 3, 1, 2))
    x2 = tensor.flatten(x1, ndim=3)
    return x2

def filter_merge_output_shape(input_shape):
    new_shape = (input_shape[0], input_shape[3], input_shape[1] * input_shape[2])
    return new_shape

def compile_model(model, fn_tags=('train', 'predict')):
    """
    Compile the model train & predict functions
    :param model:
    :param fn_tags: list of function tags
    :return:
    """
    fns = []
    for fn_tag in fn_tags:
        if fn_tag == 'train':
            X_var = get_layer_by_name(model, 'input0').input_var
            Y_var = tensor.fmatrix('Y')
            X_mask_var = tensor.fmatrix('X_mask')
            Y_mask_var = tensor.fmatrix('Y_mask')
            scorematrix = get_output(model)
            loss = ctc_cost_logscale(Y_var, scorematrix, Y_mask_var, X_mask_var)
            resultseq, resultseq_mask = ctc_best_path_decode(scorematrix, X_mask_var)
            CER, TE, TG = ctc_CER(resultseq, Y_var.T, resultseq_mask, Y_mask_var.T)
            params = get_all_params(model, trainable=True)
            updates = adadelta(loss, params)
            train_fn = theano.function([X_var, Y_var, X_mask_var, Y_mask_var], [loss, CER, TE, TG], updates=updates)
            fns.append(train_fn)
        elif fn_tag == 'predict':
            X_var = get_layer_by_name(model, 'input0').input_var
            scorematrix_test = get_output(model, deterministic=True)
            predict_fn = theano.function([X_var], scorematrix_test, no_default_updates=True)
            fns.append(predict_fn)
        else:
            raise ValueError('fn_tag = %s not recognized' % fn_tag)
    if len(fns) == 1:
        return fns[0]
    else:
        return fns


#--- best ---#
def build_model(feadim, Nclass, kernel_size=3, border_mode='same', input_length=None, noise=(0.1, 0.2, 0.1)):
    """
    Input shape: X.shape=(B, 1, rows, cols), GT.shape=(B, L)
    :param feadim:
    :param Nclass:
    :param loss:
    :param optimizer:
    :return:
    """
    input0    = InputLayer(shape=(None, 1, feadim, input_length), name='input0')
    pool0     = MaxPool2DLayer(input0, pool_size=(2, 2), name='pool0')
    pool1     = MaxPool2DLayer(pool0, pool_size=(2, 2), name='pool1')
    pool2     = MaxPool2DLayer(pool1, pool_size=(2, 1), name='pool2')
    pool3     = MaxPool2DLayer(pool2, pool_size=(2, 1), name='pool3')
    permute0  = ExpressionLayer(pool3, filter_merge, output_shape=filter_merge_output_shape, name='permute0')
    pool4     = Pool1DLayer(permute0, pool_size=2, mode='average_exc_pad', axis=1, name='pool4')
    dense0    = DenseLayer(pool4, num_units=Nclass+1, nonlinearity=softmax, num_leading_axes=2, name='dense0')
    return dense0

if __name__ == '__main__':
    import numpy as np
    gpu_id = 'cuda0'

    if gpu_id == 'cpu':
        print('WARNING: Now CPU is used for training')
    elif gpu_id.startswith('gpu'):
        import theano.sandbox.cuda
        print('Using GPU (oldbackend) %s' % gpu_id)
        theano.sandbox.cuda.use(gpu_id, force=True)
    else:  # new backend of Theano
        import theano.gpuarray
        print('Using GPU (newbackend) %s' % gpu_id)
        theano.gpuarray.use(gpu_id, force=True)

    model = build_model(feadim=36, Nclass=6335, border_mode='same', kernel_size=3, input_length=None)
    print("model compiling...")
    train_fn, predict_fn = compile_model(model)

    print("train starts...")
    for i in range(1000):
        X = np.random.rand(4, 1, 36, 249).astype(np.float32)
        X_mask = np.random.rand(4, 31).astype(np.float32)
        Y = np.random.randint(0, 6334, size=(4,7)).astype(np.float32)
        Y_mask = np.random.rand(4,7).astype(np.float32)
        ctcloss, CER_batch, ed_batch, seqlen_batch = train_fn(X, Y, X_mask, Y_mask)
        print('ctcloss = ', ctcloss)
