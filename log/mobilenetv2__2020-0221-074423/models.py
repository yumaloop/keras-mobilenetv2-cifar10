import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, DepthwiseConv2D, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D, Activation, ReLU, Add
from keras.layers.normalization import BatchNormalization

class MobileNetV2():
    
    def __init__(self, 
                 input_shape=(32, 32, 3),
                 nb_class=10, 
                 alpha=1.0,
                 include_top=True,
                 l2_coef=1e-2,
                 pooling=None):
        '''
        input_shape: 
            shape of input tensor
        nb_class: 
            number of classes
        alpha: 
            param for relu6 function
        l2_coef: 
            coefficient of L2 norm term
        include_top: 
            flag to determine whether or not the model includes the FC(Dense) layer 
        pooling: 
            flag to determine whther or not the model includes the Pooling layer 
        '''
        self.input_shape = input_shape
        self.nb_class = nb_class
        self.alpha = alpha
        self.include_top = include_top
        self.l2_coef = l2_coef
        self.pooling = pooling
        
        # first_conv_filters: Num of filters in Bottom Conv2D
        self.first_conv_filters = self._make_divisible(self.input_shape[0] * self.alpha, 8)
        
        # last_conv_filters: Num of filters in Top Conv2D
        if self.alpha > 1.0:
            self.last_conv_filters = self._make_divisible(1280 * self.alpha, 8)
        else:
            self.last_conv_filters = 1280

    def _make_divisible(self, v, divisor, min_value=None):
        ''' define each filter size. '''
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v
        
    def _inverted_res_block(self, inputs, filters, stride, 
                            expansion, block_id, dropout=0.):
        '''
        Return the calcurated output tensor of inverted residual block in model.
        
        inputs: input tensor
        expansion: 
        stride: 
        filters: 
        block_id: ブロックのID
        dropout: ドロップアウトを行う比率
        '''
        x = inputs
        in_channels = inputs._keras_shape[-1]
        pointwise_conv_filters = int(filters * self.alpha)
        pointwise_filters = self._make_divisible(pointwise_conv_filters, 8)

        # Expansion Convolution
        if block_id:
            x = Conv2D(
                expansion * in_channels,
                kernel_size=1,
                strides=1,
                padding='same',
                use_bias=False,
                activation=None,
                kernel_initializer="he_normal",
                name='block_' + str(block_id) + '_expand_Conv2D')(x)
            x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='block_' + str(block_id) + '_expand_BN')(x)
            x = ReLU(6., name='block_' + str(block_id) + '_expand_relu6')(x)
            
        # Depthwise Convolution
        x = DepthwiseConv2D(
                kernel_size=3, 
                strides=stride, 
                activation=None, 
                use_bias=False, 
                padding='same', 
                kernel_initializer="he_normal", 
                name='block_' + str(block_id) + '_depthwise_Conv2D')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='block_' + str(block_id) + '_depthwise_BN')(x)
        x = ReLU(6., name='block_' + str(block_id) + '_depthwise_relu6')(x)
            
        # Projection Convolution
        x = Conv2D(
                pointwise_filters, 
                kernel_size=1, 
                strides=1, 
                padding='same', 
                use_bias=False, 
                activation=None, 
                kernel_initializer="he_normal", 
                name='block_' + str(block_id) + '_project_Conv2D')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='block_' + str(block_id) + '_project_BN')(x)

        # Add inputs
        if in_channels == pointwise_filters and stride == 1:
            return Add(name='block_' + str(block_id) + '_add')([inputs, x])
        
        # Dropout
        if dropout:
            x = Dropout(rate=dropout)(x)
        
        return x
    
    def _bottom_block(self, inputs):
        x = Conv2D(self.first_conv_filters,
            kernel_size=3,
            strides=1,
            padding='same',
            use_bias=False,
            kernel_initializer="he_normal",
            name='bottom_Conv2D')(inputs)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='bottom_BN')(x)
        x = ReLU(6., name='bottom_relu6')(x)
        return x
    
    def _top_block(self, inputs):
        x = Conv2D(self.last_conv_filters,
                   kernel_size=1,
                   use_bias=False,
                   kernel_initializer="he_normal",
                   name='top_Conv2D')(inputs)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='top_BN')(x)
        x = ReLU(6., name='top_relu6')(x)

        # Top layer (whether or not model includes Full Connected layer in top)
        if self.include_top:
            x = GlobalAveragePooling2D(name='top_GlobalAveragePool2D')(x)
            x = Dense(self.nb_class, activation='softmax', use_bias=True, name='top_softmax')(x)
        else:
            if self.pooling == 'avg':
                x = GlobalAveragePooling2D(name='top_GlobalAveragePool2D')(x)
            elif self.pooling == 'max':
                x = GlobalMaxPooling2D(name='top_GlobalMaxPool2D')(x)
        return x
        
    def build(self):          
        with tf.variable_scope('MobileNetV2'):
            
            with tf.variable_scope('Bottom_Group'):
                inputs = Input(shape=self.input_shape)
                x = self._bottom_block(inputs)

            with tf.variable_scope('Res_Group_0', reuse=tf.AUTO_REUSE):
                x = self._inverted_res_block(x, filters=16, stride=1, expansion=1, block_id=0 )
                x = self._inverted_res_block(x, filters=24, stride=1, expansion=6, block_id=1 )
                x = self._inverted_res_block(x, filters=24, stride=1, expansion=6, block_id=2 )

            with tf.variable_scope('Res_Group_1', reuse=tf.AUTO_REUSE):
                x = self._inverted_res_block(x, filters=32, stride=2, expansion=6, block_id=3 )
                x = self._inverted_res_block(x, filters=32, stride=1, expansion=6, block_id=4 )
                x = self._inverted_res_block(x, filters=32, stride=1, expansion=6, block_id=5, dropout=0.25)

            with tf.variable_scope('Res_Group_2', reuse=tf.AUTO_REUSE):
                x = self._inverted_res_block(x, filters=64, stride=2, expansion=6, block_id=6 )
                x = self._inverted_res_block(x, filters=64, stride=1, expansion=6, block_id=7 )
                x = self._inverted_res_block(x, filters=64, stride=1, expansion=6, block_id=8 )
                x = self._inverted_res_block(x, filters=64, stride=1, expansion=6, block_id=9, dropout=0.25 )

            with tf.variable_scope('Res_Group_3', reuse=tf.AUTO_REUSE):
                x = self._inverted_res_block(x, filters=96, stride=1, expansion=6, block_id=10 )
                x = self._inverted_res_block(x, filters=96, stride=1, expansion=6, block_id=11 )
                x = self._inverted_res_block(x, filters=96, stride=1, expansion=6, block_id=12, dropout=0.25 )

            with tf.variable_scope('Res_Group_4', reuse=tf.AUTO_REUSE):
                x = self._inverted_res_block(x, filters=160, stride=2, expansion=6, block_id=13 )
                x = self._inverted_res_block(x, filters=160, stride=1, expansion=6, block_id=14 )
                x = self._inverted_res_block(x, filters=160, stride=1, expansion=6, block_id=15, dropout=0.25 )
                x = self._inverted_res_block(x, filters=320, stride=1, expansion=6, block_id=16, dropout=0.25 )

            with tf.variable_scope('Top_Group'):
                x = self._top_block(x)
                        
        # create model of MobileNetV2 (for CIFAR-10)
        model = keras.models.Model(inputs=inputs, outputs=x, name='mobilenetv2_cifar10')
        return model
    

