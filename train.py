#!/usr/bin/env python
# coding: utf-8

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import shutil
import json
import argparse
import numpy as np
import tensorflow as tf
from keras.utils import np_utils
from keras.optimizers import SGD, Adam
from keras.callbacks import Callback, ModelCheckpoint, CSVLogger
from datetime import datetime
from pprint import pprint
from qctools.mask_layer import MaskExponentialAlphaScheduler, MaskCustomCSVLogger  #  CustomCallback class
from models import MobileNetV2
from load_data import load_cifar10


# def main(args):
def main():
    input_shape=(32, 32, 3)
    num_classes=10
    
    # Load cifar10 data
    (X_train, y_train),(X_test, y_test) = load_cifar10()
    
    # Define model
    model = MobileNetV2(input_shape=input_shape, nb_class=num_classes, include_top=True).build()
    MODEL_NAME = "mobilenetv2" + datetime.now().strftime("%y-%m%d-%H%M%S")
    
    # Path & Env. settings -------------------------------------------------------------
    LOG_DIR = os.path.join("./log", MODEL_NAME)
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    shutil.copyfile(os.path.join(os.getcwd(), 'train.py'), os.path.join(LOG_DIR, 'train.py'))
    shutil.copyfile(os.path.join(os.getcwd(), 'run.sh'), os.path.join(LOG_DIR, 'run.sh'))
    shutil.copyfile(os.path.join(os.getcwd(), 'models.py'), os.path.join(LOG_DIR, 'models.py'))

    MODEL_WEIGHT_CKP_PATH=os.path.join(LOG_DIR, "best_weights.h5")
    MODEL_TRAIN_LOG_CSV_PATH=os.path.join(LOG_DIR, "train_log.csv")
    # MODEL_INIT_WEIGHTS_PATH=str(args.weights_path)
    # ----------------------------------------------------------------------------------

    # Compile model 
    model.summary()
    model.compile(optimizer=SGD(lr=2e-2, momentum=0.9, decay=0.0, nesterov=False), # SGD(lr=2e-2, momentum=0.9, decay=0.0, nesterov=False)
                  loss='categorical_crossentropy',
                  loss_weights=[1.0], # The loss weight for model output without regularization loss. Set 0.0 due to validate only regularization factor.
                  metrics=['accuracy'])

    # load init weights from pre-trained model
    """
    if args.trans_learn:
        model.load_weights(MODEL_INIT_WEIGHTS_PATH, by_name=False)
        print("Load model init weights from", MODEL_INIT_WEIGHTS_PATH)
    print("Produce training results in", LOG_DIR)
    """

    '''
    # debug code
    print("model.total_loss :")
    pprint(model.total_loss)
    print("mdoel.losses :")
    pprint(model.losses)
    '''

    # Set model callbacks
    callbacks = []
    callbacks.append(ModelCheckpoint(MODEL_WEIGHT_CKP_PATH, monitor='val_loss', save_best_only=True, save_weights_only=True))
    callbacks.append(CSVLogger(MODEL_TRAIN_LOG_CSV_PATH))
    callbacks.append(MaskCustomCSVLogger(LOG_DIR+'/metrics.csv', model, dataset=(X_train, y_train, X_test, y_test)))
    callbacks.append(MaskExponentialAlphaScheduler(init_alpha=1.0, growth_steps=10, growth_rate=10, clip_min=0, clip_max=100))

    # Train model
    history = model.fit(X_train, 
                  y_train, 
                  batch_size=128, 
                  epochs=30,
                  verbose=1,
                  callbacks=callbacks,
                  validation_data=(X_test, y_test))

    # Validation
    val_loss, val_acc = model.evaluate(X_test, y_test, verbose=1)
    print("--------------------------------------")
    print("model name : ", MODEL_NAME)
    print("validation loss     : {:.5f}".format(val_loss)) 
    print("validation accuracy : {:.5f}".format(val_acc)) 

    # Save model as "instance"
    ins_name = 'model_instance'
    ins_path = os.path.join(LOG_DIR, ins_name) + '.h5'
    model.save(ins_path)

    # Save model as "architechture"
    arch_name = 'model_fin_architechture'
    arch_path = os.path.join(LOG_DIR, arch_name) + '.json'
    json_string = model.to_json()
    with open(arch_path, 'w') as f:
        f.write(json_string)    

if __name__ == '__main__':
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='model architecture.', choices=['cnn7', 'lenet5'], default='lenet5')
    parser.add_argument('--type',  help='matrix type for custom layers.', choices=['base', 'qc', 'qcsc'], default='base', required=True)
    parser.add_argument('--trans_learn', help='flag to design whether or not apply transfer learning.', action='store_true')
    parser.add_argument('--weights_path', help='file path to the initial model weights (.h5)', default='./log/base/best_weights.h5')
    args = parser.parse_args()
    main(args)
    """
    main()
