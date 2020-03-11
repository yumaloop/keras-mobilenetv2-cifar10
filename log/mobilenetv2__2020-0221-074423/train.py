#!/usr/bin/env python
# coding: utf-8

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import shutil
import argparse
import numpy as np
import tensorflow as tf
import keras
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler
from datetime import datetime
from models import MobileNetV2
from load_data import load_cifar10


# def main(args):
def main(args):
    input_shape=(32, 32, 3)
    num_classes=10
    batch_size=int(args.batch_size)
    epochs=int(args.epochs)
    
    # Load cifar10 data
    (X_train, y_train),(X_test, y_test) = load_cifar10()
    
    # Define model
    model = MobileNetV2(input_shape=input_shape, nb_class=num_classes, include_top=True).build()
    MODEL_NAME = "mobilenetv2__" + datetime.now().strftime("%Y-%m%d-%H%M%S")
    
    # Path & Env. settings -------------------------------------------------------------
    LOG_DIR = os.path.join("./log", MODEL_NAME)
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    shutil.copyfile(os.path.join(os.getcwd(), 'train.sh'), os.path.join(LOG_DIR, 'train.sh'))
    shutil.copyfile(os.path.join(os.getcwd(), 'train.py'), os.path.join(LOG_DIR, 'train.py'))
    shutil.copyfile(os.path.join(os.getcwd(), 'models.py'), os.path.join(LOG_DIR, 'models.py'))

    MODEL_WEIGHT_CKP_PATH=os.path.join(LOG_DIR, "best_weights.h5")
    MODEL_TRAIN_LOG_CSV_PATH=os.path.join(LOG_DIR, "train_log.csv")
    # ----------------------------------------------------------------------------------

    # Compile model 
    model.summary()
    model.compile(optimizer=keras.optimizers.SGD(lr=2e-2, momentum=0.9, decay=0.0, nesterov=False),
                  loss='categorical_crossentropy',
                  loss_weights=[1.0], # The loss weight for model output without regularization loss. Set 0.0 due to validate only regularization factor.
                  metrics=['accuracy'])

    # Load initial weights from pre-trained model
    if args.trans_learn:
        model.load_weights(str(args.weights_path), by_name=False)
        print("Load model init weights from", MODEL_INIT_WEIGHTS_PATH)

    print("Produce training results in", LOG_DIR)

    # Set learning rate
    learning_rates=[]
    for i in range(5):
        learning_rates.append(2e-2)
    for i in range(50-5):
        learning_rates.append(1e-2)
    for i in range(100-50):
        learning_rates.append(8e-3)
    for i in range(150-100):
        learning_rates.append(4e-3)
    for i in range(200-150):
        learning_rates.append(2e-3)
    for i in range(300-200):
        learning_rates.append(1e-3)

    # Set model callbacks
    callbacks = []
    callbacks.append(ModelCheckpoint(MODEL_WEIGHT_CKP_PATH, monitor='val_loss', save_best_only=True, save_weights_only=True))
    callbacks.append(CSVLogger(MODEL_TRAIN_LOG_CSV_PATH))
    callbacks.append(LearningRateScheduler(lambda epoch: float(learning_rates[epoch])))

    # data generator with data augumatation
    datagen = keras.preprocessing.image.ImageDataGenerator(
            featurewise_center=False, 
            featurewise_std_normalization=False, 
            rotation_range=0.0,
            width_shift_range=0.2, 
            height_shift_range=0.2, 
            vertical_flip=False,
            horizontal_flip=True)
    datagen.fit(X_train)

    # Train model
    history = model.fit_generator(
              datagen.flow(X_train, y_train, batch_size=batch_size),
              steps_per_epoch=len(X_train) / batch_size,
              epochs=epochs,
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=128)
    parser.add_argument('--epochs', default=100)
    parser.add_argument('--trans_learn', help='flag to design whether or not apply transfer learning.', action='store_true')
    parser.add_argument('--weights_path', help='file path to the initial model weights (.h5)', default='./log/base/best_weights.h5')
    args = parser.parse_args()
    main(args)
