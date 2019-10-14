from __future__ import absolute_import, division, print_function, unicode_literals

from keras import backend as K
import tensorflow as tf
from tensorflow.keras import layers

import os
import time
import matplotlib.pyplot as plt

from nc_loader import ERA5Dataset

def Unet():
    concat_axis = 3
    inputs = layers.Input(shape = (720, 1440, 3))

    feats = 16
    bn0 = layers.BatchNormalization(axis=3)(inputs)
    conv1 = layers.Conv2D(feats, (5, 5), activation='relu', padding='same', name='conv1_1')(bn0)
    bn1 = layers.BatchNormalization(axis=3)(conv1)
    conv1 = layers.Conv2D(feats, (5, 5), activation='relu', padding='same')(bn1)
    bn2 = layers.BatchNormalization(axis=3)(conv1)

    pool1 = layers.MaxPooling2D(pool_size=(3, 3))(bn2)
    conv2 = layers.Conv2D(2*feats, (3, 3), activation='relu', padding='same')(pool1)
    bn3 = layers.BatchNormalization(axis=3)(conv2)
    conv2 = layers.Conv2D(2*feats, (3, 3), activation='relu', padding='same')(bn3)
    bn4 = layers.BatchNormalization(axis=3)(conv2)

    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(bn4)

    conv3 = layers.Conv2D(4*feats, (3, 3), activation='relu', padding='same')(pool2)
    bn5 = layers.BatchNormalization(axis=3)(conv3)
    conv3 = layers.Conv2D(4*feats, (3, 3), activation='relu', padding='same')(bn5)
    bn6 = layers.BatchNormalization(axis=3)(conv3)

    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(bn6)

    conv4 = layers.Conv2D(8*feats, (3, 3), activation='relu', padding='same')(pool3)
    bn7 = layers.BatchNormalization(axis=3)(conv4)
    conv4 = layers.Conv2D(8*feats, (3, 3), activation='relu', padding='same')(bn7)
    bn8 = layers.BatchNormalization(axis=3)(conv4)

    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(bn8)

    conv5 = layers.Conv2D(16*feats, (3, 3), activation='relu', padding='same')(pool4)
    bn9 = layers.BatchNormalization(axis=3)(conv5)
    conv5 = layers.Conv2D(16*feats, (3, 3), activation='relu', padding='same')(bn9)
    bn10 = layers.BatchNormalization(axis=3)(conv5)

    pool5 = layers.MaxPooling2D(pool_size=(2, 2))(bn10)

    conv6 = layers.Conv2D(32*feats, (3, 3), activation='relu', padding='same')(pool5)
    bn11 = layers.BatchNormalization(axis=3)(conv6)
    conv6 = layers.Conv2D(32*feats, (3, 3), activation='relu', padding='same')(bn11)
    bn12 = layers.BatchNormalization(axis=3)(conv6)

    up_conv6 = layers.UpSampling2D(size=(2, 2))(bn12)
    up7 = layers.concatenate([up_conv6, conv5], axis=concat_axis)

    conv7 = layers.Conv2D(16*feats, (3, 3), activation='relu', padding='same')(up7)
    bn13 = layers.BatchNormalization(axis=3)(conv6)
    conv7 = layers.Conv2D(16*feats, (3, 3), activation='relu', padding='same')(bn12)
    bn14 = layers.BatchNormalization(axis=3)(conv6)
    
    up_conv5 = layers.UpSampling2D(size=(2, 2))(bn10)
    up6 = layers.concatenate([up_conv5, conv4], axis=concat_axis)

    conv6 = layers.Conv2D(8*feats, (3, 3), activation='relu', padding='same')(up6)
    bn15 = layers.BatchNormalization(axis=3)(conv6)
    conv6 = layers.Conv2D(8*feats, (3, 3), activation='relu', padding='same')(bn15)
    bn16 = layers.BatchNormalization(axis=3)(conv6)

    up_conv6 = layers.UpSampling2D(size=(2, 2))(bn16)
    up7 = layers.concatenate([up_conv6, conv3], axis=concat_axis)
    conv7 = layers.Conv2D(4*feats, (3, 3), activation='relu', padding='same')(up7)
    bn13 = layers.BatchNormalization(axis=3)(conv7)
    conv7 = layers.Conv2D(4*feats, (3, 3), activation='relu', padding='same')(bn13)
    bn14 = layers.BatchNormalization(axis=3)(conv7)

    up_conv7 = layers.UpSampling2D(size=(2, 2))(bn14)
    up8 = layers.concatenate([up_conv7, conv2], axis=concat_axis)
    conv8 = layers.Conv2D(2*feats, (3, 3), activation='relu', padding='same')(up8)
    bn15 = layers.BatchNormalization(axis=3)(conv8)
    conv8 = layers.Conv2D(2*feats, (3, 3), activation='relu', padding='same')(bn15)
    bn16 = layers.BatchNormalization(axis=3)(conv8)

    up_conv8 = layers.UpSampling2D(size=(3, 3))(bn16)
    up9 = layers.concatenate([up_conv8, conv1], axis=concat_axis)
    conv9 = layers.Conv2D(feats, (3, 3), activation='relu', padding='same')(up9)
    bn17 = layers.BatchNormalization(axis=3)(conv9)
    conv9 = layers.Conv2D(feats, (3, 3), activation='relu', padding='same')(bn17)
    bn18 = layers.BatchNormalization(axis=3)(conv9)

    conv10 = layers.Conv2D(1, (1, 1))(bn18)

    model = tf.keras.models.Model(inputs=inputs, outputs=conv10)

    return model


train_fnames = ["/home/lar116/project/ERA5_ECMWF/era5s_geop_201801.nc", 
                "/home/lar116/project/ERA5_ECMWF/era5s_geop_201802.nc", 
                "/home/lar116/project/ERA5_ECMWF/era5s_geop_201804.nc", 
                "/home/lar116/project/ERA5_ECMWF/era5s_geop_201805.nc", 
                "/home/lar116/project/ERA5_ECMWF/era5s_geop_201806.nc"] 

test_fnames = ["/home/lar116/project/ERA5_ECMWF/era5s_geop_201803.nc"]


strategy = tf.distribute.MirroredStrategy(devices=["/device:GPU:0", "/device:GPU:1"])
print ('Number of devices: {}'.format(strategy.num_replicas_in_sync))
EPOCHS = 10
BATCH_SIZE_PER_REPLICA = 4
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

train_dataset = ERA5Dataset(train_fnames, batch_size=BATCH_SIZE_PER_REPLICA)
test_dataset = ERA5Dataset(test_fnames, batch_size=BATCH_SIZE_PER_REPLICA)

train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)


with strategy.scope():
  model = Unet()
  optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, decay=1e-6, momentum=0.9, nesterov=True)
  
  train_loss = tf.keras.metrics.Mean()
  test_loss = tf.keras.metrics.Mean()

  def train_step(inputs, outputs):
    with tf.GradientTape() as t:
      loss = tf.reduce_mean(tf.square(outputs - model(inputs, training=True)))
    
    grads = t.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    train_loss(tf.reduce_mean(tf.square(outputs - model(inputs))))
    return loss

  def test_step(inputs, outputs):
    with tf.GradientTape() as t:
      loss = tf.reduce_mean(tf.square(outputs - model(inputs, training=False)))
    
    test_loss(tf.reduce_mean(tf.square(outputs - model(inputs))))
    return loss

  @tf.function
  def distributed_train_step(inputs, outputs):
    per_replica_losses = strategy.experimental_run_v2(train_step, args=(inputs, outputs,))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
 
  @tf.function
  def distributed_test_step(inputs, outputs):
    return strategy.experimental_run_v2(test_step, args=(inputs, outputs,))


  for epoch in range(EPOCHS):
    total_loss = 0.0
    num_batches = 0
    for (batch, (inputs, outputs)) in enumerate(train_dataset):
      total_loss += distributed_train_step(inputs, outputs)
      num_batches += 1
    train_loss = total_loss / num_batches

