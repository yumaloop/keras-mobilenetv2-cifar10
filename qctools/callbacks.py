# coding:utf-8
import sys
import re
import csv
import six
import math
import copy
import numpy as np
import keras.backend as K
from collections import Iterable, OrderedDict
from keras.callbacks import Callback, CSVLogger


class MaskLagrangianAlphaScheduler(Callback):
    """
    Custom `keras.callbacks.Callback` class for regularizing QcConv2D layer 
    that control the value of alpha for each epoch.

    Attributes
    ==========
    init_alpha : float
        value of alpha in first epoch
    eta : float
        hyperparam for gradient decent method. (as lr in SGD)
    loss_threshold_rate : float
        threshold for difference between step-by-step values of loss function.
    """
    def __init__(self,
                init_alpha,
                eta=100,
                loss_threshold_rate=0.01):
        self.alpha = init_alpha
        self.eta = eta
        self.loss_threshold_rate = loss_threshold_rate
        self.batch_losses =  []
        self.losses = [] # list of loss value in each epoch

    def on_batch_end(self, batch, logs=[]):
        self.batch_losses.append(logs.get('loss'))
        
    def on_epoch_end(self, epoch, logs=[]):
        # loss
        batch_losses_mean = sum(self.batch_losses) / len(self.batch_losses)
        self.losses.append(batch_losses_mean)
        
        # l2norm
        l2norm_mask, num_params_mask = self.calc_l2norm_mask()
        logs["l2norms_mask"] = l2norm_mask
        
        # alpha
        if epoch > 0:
            loss_diff = (self.losses[-1] - self.losses[-2]) / self.losses[-2]
            if self.loss_threshold_rate > loss_diff :
                new_alpha = self.alpha + self.eta * l2norm_mask / num_params_mask
                self.update_alpha(new_alpha)
                self.alpha = new_alpha
        logs["alpha"] = self.alpha
                    
    def update_alpha(self, new_alpha):
        for lay in self.model.layers:
            if hasattr(lay, 'kernel_regularizer') and hasattr(lay.kernel_regularizer, 'alpha'):
                K.set_value(lay.kernel_regularizer.alpha, new_alpha) 
                # print("epoch :", epoch, " layer name :", lay.name, " alpha :", K.eval(lay.kernel_regularizer.alpha), flush=True) # for debug

    def calc_l2norm_mask(self):
        l2norm_mask = 0.
        num_params_mask = 0
        for lay in self.model.layers:
            if hasattr(lay, 'kernel_regularizer') and hasattr(lay.kernel_regularizer, 'alpha'):
                kernel_binmask = lay.get_mask_kernel()
                kernel_weights = lay.get_weights()[0]
                kernel_mask = np.where(kernel_binmask == 0, 1, 0) * kernel_weights
                l2norm_mask += np.sum(kernel_mask ** 2)
                num_params_mask += kernel_mask.size
        return l2norm_mask, num_params_mask


    
class MaskExponentialAlphaScheduler(Callback):
    '''
    Custom `keras.callbacks.Callback` class for regularizing QcConv2D layer 
    that control the value of alpha for each epoch.

    Attributes
    ==========
    init_alpha : float
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
        # l2norm
        l2norm_mask = self.calc_l2norm_mask()
        logs["l2norms_mask"] = l2norm_mask
        
        # alpha
        new_alpha = self.alpha_exponential_growth(epoch)
        self.update_alpha(new_alpha)
        logs["alpha"] = new_alpha
                
    def update_alpha(self, new_alpha):
        for lay in self.model.layers:
            if hasattr(lay, 'kernel_regularizer') and hasattr(lay.kernel_regularizer, 'alpha'):
                K.set_value(lay.kernel_regularizer.alpha, new_alpha) 
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
    
    def calc_l2norm_mask(self):
        l2norm_mask = 0.
        for lay in self.model.layers:
            if hasattr(lay, 'kernel_regularizer') and hasattr(lay.kernel_regularizer, 'alpha'):
                kernel_binmask = lay.get_mask_kernel()
                kernel_weights = lay.get_weights()[0]
                kernel_mask = np.where(kernel_binmask == 0, 1, 0) * kernel_weights
                l2norm_mask += np.sum(kernel_mask ** 2)
        return l2norm_mask



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
