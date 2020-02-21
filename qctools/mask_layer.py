# coding:utf-8
import sys
import re
import csv
import six
import math
import copy
import numpy as np
import tensorflow as tf
import keras.backend as K
from collections import Iterable, OrderedDict
from keras.layers import Dense, Conv2D, InputSpec
from keras.callbacks import Callback, CSVLogger
from qctools.matrix_tools import QCMatrix, QCMatrixWithSpatialCoupling
from qctools.mask_kernel import ConvMaskKernel, ConvMaskConstraint, ConvMaskRegularizer
from qctools.mask_kernel import DenseMaskKernel, DenseMaskConstraint, DenseMaskRegularizer


class MaskDense(Dense):    
    '''
    Subclass of `keras.layers.Dense` class for network compression

    Arguments:
    ==========
    units : float
        Inherited attribute from `keras.layers.Dense`
    init_alpha : float
        initialize value for the argument `alpha` in `ConvMaskKernel`
    mask_matrix : instance object
        matrix object
    use_bias : bool
        flag of whether or not use bias vector
    **kwargs : dict
        the other Inherited attributes from keras.layers.Conv2D
    '''

    def __init__(self,
                 units,
                 init_alpha,
                 mask_matrix,
                 **kwargs):

        kwargs_ = dict(**kwargs)
        super(MaskDense, self).__init__(units, **kwargs_)
        self.use_bias = False # must be False
        self.init_alpha = init_alpha
        self.mask_matrix = mask_matrix # ex.) mask_matrix = QCMatrix(gamma=4) when set QCMatrix as mask_matrix_ins
    
    def get_mask_kernel(self):
        if self.mask_config:
            return DenseMaskKernel(**self.mask_config).get()
    
    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        
        # Overwrite Start ---------------------------------------------------------------------
        self.input_dim = input_dim
        self.mask_config = dict(
            shape=(self.input_dim, self.units),
            mask_matrix_ins=self.mask_matrix)
        self.kernel_regularizer = DenseMaskRegularizer(alpha=self.init_alpha, **self.mask_config) # use kernel constraint
        # self.kernel_constraint = DenseMaskConstraint(**self.mask_config)
        # Overwrite End -----------------------------------------------------------------------

        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True


class MaskConv2D(Conv2D):
    '''
    Subclass of `keras.layers.Conv2D`

    Arguments:
    ==========
    filters : np.array
        Inherited attribute from `keras.layers.Conv2D`
    kernel_size : inte
        Inherited attribute from `keras.layers.Conv2D`
    init_alpha : float
        initialize value for the argument `alpha` in `ConvMaskKernel`
    mask_matrix : instance object
        matrix object
    use_bias : bool
        flag of whether or not use bias vector
    **kwargs : dict
        the other Inherited attributes from keras.layers.Conv2D
    '''

    def __init__(self,
                 filters,
                 kernel_size,
                 init_alpha,
                 mask_matrix,
                 **kwargs):

        args_ = (filters, kernel_size)
        kwargs_ = dict(**kwargs)
        super(MaskConv2D, self).__init__(*args_, **kwargs_)
        self.use_bias = False # must be False
        self.init_alpha = init_alpha
        self.mask_matrix = mask_matrix # ex.) mask_matrix_ins = QCMatrix(gamma=4) when set QCMatrix as mask_matrix_ins
    
    def get_mask_kernel(self):
        if self.mask_config:
            return ConvMaskKernel(**self.mask_config).get()

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs  should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]       
        kernel_shape = self.kernel_size + (input_dim, self.filters)
        
        # Overwrite Start ---------------------------------------------------------------------
        self.input_dim = input_dim
        self.mask_config = dict(
            shape=(self.kernel_size[0], self.kernel_size[1], self.input_dim, self.filters),
            mask_matrix_ins=self.mask_matrix,
            data_format = self.data_format)
        self.kernel_regularizer = ConvMaskRegularizer(alpha=self.init_alpha, **self.mask_config)
        # self.kernel_constraint = ConvMaskConstraint(**self.mask_config) # use kernel_constraint
        # Overwrite End -----------------------------------------------------------------------
        
        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        self.built = True

           

class MaskExponentialAlphaScheduler(Callback):
    '''
    Custom `keras.callbacks.Callback` class for regularizing QcConv2D layer 
    that control the value of alpha for each epoch.

    Attributes
    ==========
    base_alpha : float
        value of alpha in first epoch
    epoch : int
        current epoch
    growth_steps : int
        `base_alpha` is multiplied by `growth_rate` in `growth_steps` steps
    growth_rate : int
        `base_alpha` is multiplied by `growth_rate` in `growth_steps` steps
    '''
    
    def __init__(self,
                init_alpha,
                growth_steps,
                growth_rate,
                clip_min=0,
                clip_max=200,
                staircase=None,
                name=None):
        
        self.init_alpha = init_alpha
        self.growth_steps = growth_steps
        self.growth_rate = growth_rate
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.staircase = staircase
        self.name = name
    
    def on_epoch_end(self, epoch, logs={}):
        new_alpha = self.alpha_exponential_growth(epoch)
        logs["alpha"] = new_alpha
        for lay in self.model.layers:
            if hasattr(lay, 'kernel_regularizer') and hasattr(lay.kernel_regularizer, 'alpha'):
                K.set_value(lay.kernel_regularizer.alpha, new_alpha)
                # debug code
                # print("epoch :", epoch, " layer name :", lay.name, " alpha :", K.eval(lay.kernel_regularizer.alpha), flush=True) # for debug

    def alpha_exponential_growth(self, epoch):
        p = epoch / self.growth_steps
        if self.staircase:
            p = math_ops.floor(p)
        new_alpha = self.init_alpha * math.pow(self.growth_rate, p)

        if new_alpha > self.clip_max:
            new_alpha = self.clip_max
        elif new_alpha < self.clip_min:
            new_alpha = self.clip_min
            
        return new_alpha



class MaskCustomCSVLogger(CSVLogger):
    '''
    Custom class 

    Attributes
    ==========
    filename : file path (str)
    sess : tf.Session() object
    model : keras.model object
    dataset : tuple
    separator : 
    apppend : 
    '''

    def __init__(self, filename, model, dataset, separator=',', append=False):
        CSVLogger.__init__(self, filename, separator=separator, append=append)
        self.model = model
        self.X_train, self.y_train, self.X_test, self.y_test = dataset
    
    def _mask_weights(self):
        layer_weights_before_masked={}
        for lay in self.model.layers:
            if hasattr(lay, 'kernel_regularizer') and hasattr(lay.kernel_regularizer, 'alpha'):
                # set origin weight matrix as w_ori
                w_ori = lay.get_weights()[0] # if use_bias is True, len(lay.get_weights()) == 2
                layer_weights_before_masked[lay.name] = copy.deepcopy(w_ori)
                # set mask weight as W_mask
                b_mask = lay.get_mask_kernel()
                w_mask = w_ori * b_mask
                """
                if np.sum(w_mask) == np.sum(w_ori):
                    print("w_ori :", w_ori[0][0])
                    print("b_mask :", b_mask[0][0])
                    print("w_mask :", w_mask[0][0])
                    print("hogehoge")
                    sys.exit()
                """
                lay.set_weights([w_mask])

        return layer_weights_before_masked

    def _set_weights(self, layinfo):
        for lay_name, weights in layinfo.items():
            for lay in self.model.layers:
                if lay.name == lay_name:
                    lay.set_weights([weights])

    def _validate(self, X_train, y_train, X_test, y_test):
        train_loss_notmask, train_acc_notmask = self.model.evaluate(X_train, y_train, verbose=0)
        test_loss_notmask,  test_acc_notmask  = self.model.evaluate(X_test,  y_test, verbose=0)

        # mask layer weights
        layer_weights_before_masked = self._mask_weights()

        train_loss_mask, train_acc_mask = self.model.evaluate(X_train, y_train, verbose=0)
        test_loss_mask,  test_acc_mask  = self.model.evaluate(X_test,  y_test, verbose=0)

        # reset original weights
        self._set_weights(layer_weights_before_masked)

        metrics_notmask = (train_loss_notmask, train_acc_notmask, test_loss_notmask, test_acc_notmask)
        metrics_mask    = (train_loss_mask,    train_acc_mask,    test_loss_mask,    test_acc_mask)
        return metrics_notmask, metrics_mask
    
    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        metrics_notmask, metrics_mask = self._validate(self.X_train, self.y_train, self.X_test, self.y_test)
        
        # Additional log variables in the csv file.
        logs = logs or {}
        logs["train_loss"] = metrics_notmask[0] # train_loss_notmask
        logs["train_acc"]  = metrics_notmask[1] # train_acc_notmask
        logs["test_loss"]  = metrics_notmask[2] # test_loss_notmask
        logs["test_acc"]   = metrics_notmask[3] # test_acc_notmask
        logs["train_loss_mask"] = metrics_mask[0] # train_loss_mask
        logs["train_acc_mask"]  = metrics_mask[1] # train_acc_mask
        logs["test_loss_mask"]  = metrics_mask[2] # test_loss_mask
        logs["test_acc_mask"]   = metrics_mask[3] # test_acc_mask

        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, six.string_types):
                return k
            elif isinstance(k, Iterable) and not is_zero_dim_ndarray:
                return '"[%s]"' % (', '.join(map(str, k)))
            else:
                return k

        if self.keys is None:
            self.keys = sorted(logs.keys())
            self.keys

        if self.model.stop_training:
            # We set NA so that csv parsers do not fail for this last epoch.
            logs = dict([(k, logs[k] if k in logs else 'NA') for k in self.keys])

        if not self.writer:
            class CustomDialect(csv.excel):
                delimiter = self.sep
            fieldnames = ['epoch'] + self.keys
            if six.PY2:
                fieldnames = [unicode(x) for x in fieldnames]
            self.writer = csv.DictWriter(self.csv_file,
                                         fieldnames=fieldnames,
                                         dialect=CustomDialect)
            if self.append_header:
                self.writer.writeheader()

        row_dict = OrderedDict({'epoch': epoch})
        row_dict.update((key, handle_value(logs[key])) for key in self.keys)
        self.writer.writerow(row_dict)
        self.csv_file.flush()
