# coding:utf-8
import re
import math
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.constraints import Constraint
from keras.regularizers import Regularizer
from keras.callbacks import Callback
from qctools.matrix_tools import QCMatrix, QCMatrixWithSpatialCoupling

class DenseMaskKernel(object):
    '''
    Arguments:
    ==========
    shape : (in_ch, out_ch)
        shape of the weight matrix. Tuple object.
    mask_matrix_ins : instance object
        matrix instance objects defined in 'matrix_tools.py'
    '''

    def __init__(self, shape, mask_matrix_ins):
        assert len(shape)==2, "The argument `shape` should be the 2 dimensional tuple object."
        self.shape = shape
        self.mask_matrix_ins = mask_matrix_ins

    def get_config(self):
        sefl.config = dict(
            shape=self.shape, 
            mask_matrix_ins=self.mask_matrix_ins)
        return self.config

    def get(self):
        nrow, ncol = self.shape
        # make mask matrix
        self.mask_mat = self.mask_matrix_ins.build(int(nrow), int(ncol))
        return self.mask_mat


class ConvMaskKernel(object):
    '''
    Arguments:
    ==========
    shape: (kh, kw, in_ch, out_ch) 
        shape of the weight matrix (i.e. conv kernels). Tuple for keras-tf-order.
        - kh : kernel height size
        - kw : kernel width size
        - out_ch : kernel output channel size
        - in_ch : kernel input channel size
    mask_matrix_ins : instance object
        matrix instance objects defined in `matrix_tools.py`.
    data_format : string, 'channels_first' or 'channels_last'
        the Inherited attribute from keras.layers.Conv2D
    '''
        
    def __init__(self,
                 shape,
                 mask_matrix_ins,
                 data_format='channels_last'):
        assert len(shape)==4, "The argument `shape` should be the 4 dimensional tuple object."
        self.shape = shape
        self.mask_matrix_ins = mask_matrix_ins
        self.data_format = data_format

    def get_config(self):
        self.config = dict(
            shape=self.shape,
            mask_matrix_ins=self.mask_matrix_ins,
            data_format=self.data_format)
        return self.config

    def get(self):
        '''
        Return
        ======
        mask_mat : np.array
            binary matrix masked by the shape given by W_shape.
        '''
        kh, kw, in_ch, out_ch = self.shape
        ncol = kh * kw * in_ch
        nrow = out_ch
        
        # make masked matrix
        mask_mat = self.mask_matrix_ins.build(int(nrow), int(ncol))
        
        # check format (channel first or channel last)
        if self.data_format == 'channels_last':
            return mask_mat.reshape((out_ch, kh, kw, in_ch)).transpose(1, 2, 3, 0)
        elif self.data_format == 'channels_first':
            return mask_mat.reshape((out_ch, in_ch, kh, kw)).transpose(2, 3, 1, 0)


class DenseMaskConstraint(Constraint):
    '''
    Subclass of `keras.constraints.Constraint`

    Arguments:
    ==========
    **mask_config : {shape:?, mask_matrix_ins:?}
        dict object to define `DenseMaskKernel` class instance which includes the information about weight matrix.
    '''
    def __init__(self, **mask_config):
        self.base_config = super().get_config()
        self.mask_config = mask_config
        mask_kernel = DenseMaskKernel(**mask_config).get() # binary matrix called by mask_matrix_ins
        self._mask = K.constant(mask_kernel)
        
    def get_config(self):
        self.config = dict(list(self.base_config.items()) + list(self.mask_config.items()))
        return self.config

    def __call__(self, x):
        return self._mask * x


class ConvMaskConstraint(Constraint):
    '''
    Subclass of `keras.constraints.Constraint`
    
    Arguments:
    ==========
    **mask_config : {shape:?, mask_matrix_ins:?, data_format:?}
        dict object to define `ConvMaskKernel` class instance 
        which includes the information about weight matrix (i.e. conv kernels).
        `keep_kernel_channel_last` is the Inherited attribute from `keras.layers.Conv2D` class.
    '''   

    def __init__(self, **mask_config):
        self.base_config = super().get_config()
        self.mask_config = mask_config
        mask_kernel = ConvMaskKernel(**mask_config).get() # binary matrix called by mask_matrix_ins
        self._mask = K.constant(mask_kernel)

    def get_config(self):
        self.config = dict(list(self.base_config.items()) + list(self.mask_config.items()))
        return self.config

    def __call__(self, x):
        return self._mask * x


class DenseMaskRegularizer(Regularizer):
    '''
    Subclass of `keras.regularizers.Regularizer`

    Arguments:
    ==========
    name : string
        instance name
    lp_norm : string. 'l1' or 'l2'
        flag of norm type the used in the regularization.
    alpha : float
        hyperparams for the regularization
        which is used in loss function as the coef of the regularization term.
    mask_config : {shape:?, mask_matrix_ins:?}
        dict object to define `DenseMaskKernel` class instance 
        which includes the information about weight matrix.
    '''

    def __init__(self, name=None, lp_norm='l2', alpha=None, **mask_config):
        self.name = name
        self.lp_norm = lp_norm
        self.alpha = K.variable(alpha)
        self.mask_config = mask_config
        self.mask_kernel = DenseMaskKernel(**mask_config).get()
        self._coef = K.constant(1.0 - self.mask_kernel)

    def get_config(self):
        return self.mask_config

    def __call__(self, x):
        if self.lp_norm == 'l1':
            return K.sum(self.alpha * self._coef * K.abs(x) / 1)
        elif self.lp_norm == 'l2':
            return K.sum(self.alpha * self._coef * K.square(x) / 2)
    
class ConvMaskRegularizer(Regularizer):
    '''
    Subclass of `keras.regularizers.Regularizer`

    Attributes
    ==========
    name : string
        instance name
    lp_norm : string, 'l1' or 'l2'
        flag of norm type the used in the regularization.
    alpha : float
        hyperparams for the regularization
        which is used in loss function as the coef of the regularization term.
    **mask_config : {shape:?, mask_matrix_ins:?, data_format:?}
        dict object to define `ConvMaskKernel` class instance 
        which includes the information about weight matrix (i.e. conv kernels).
        `keep_kernel_channel_last` is the Inherited attribute from keras.layers.Conv2D
    '''   
    def __init__(self, name=None, lp_norm='l2', alpha=None, **mask_config):
        self.name = name
        self.lp_norm = lp_norm
        self.alpha = K.variable(alpha)
        self.mask_config = mask_config
        self.mask_kernel = ConvMaskKernel(**mask_config).get()
        self._coef = K.constant(1.0 - self.mask_kernel)
                        
    def get_config(self):
        return self.mask_config

    def __call__(self, x):
        if self.lp_norm == 'l1':
            return K.sum(self.alpha * self._coef * K.abs(x) / 1)
        elif self.lp_norm == 'l2':
            return K.sum(self.alpha * self._coef * K.square(x) / 2)
            # return K.sum(K.eval(self.alpha) * self._coef * K.square(x) / 2)
