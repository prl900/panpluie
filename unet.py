from __future__ import absolute_import, division, print_function, unicode_literals

from keras import backend as K
import tensorflow as tf

import os
import time
import matplotlib.pyplot as plt

from nc_loader import ERA5Dataset

train_fnames = ["/data/ERA5/era5s_geop_201801.nc", 
                "/data/ERA5/era5s_geop_201802.nc", 
                "/data/ERA5/era5s_geop_201804.nc", 
                "/data/ERA5/era5s_geop_201805.nc", 
                "/data/ERA5/era5s_geop_201806.nc"] 
train_dataset = ERA5Dataset(train_fnames, batch_size=4)

test_fnames = ["/data/ERA5/era5s_geop_201803.nc"]
test_dataset = ERA5Dataset(test_fnames, batch_size=4)


def downsample(filters, size, apply_batchnorm=True):
    #initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    #result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
    #                                  #kernel_initializer=initializer, use_bias=False))
    result.add(tf.keras.layers.Conv2D(filters, size, strides=1, activation='relu', padding='same'))#, 
    result.add(tf.keras.layers.Conv2D(filters, size, strides=2, activation='relu', padding='same'))#, 
                                      #kernel_regularizer=tf.keras.regularizers.l1(0.01)))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization(axis=3))

    #result.add(tf.keras.layers.LeakyReLU())

    return result


def upsample(filters, size, apply_dropout=False):
    #initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    #result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same',
    #                                           kernel_initializer=initializer, use_bias=False))
    result.add(tf.keras.layers.Conv2D(filters, size, strides=1, activation='relu', padding='same'))#, 
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2, activation='relu', padding='same'))#, 
                                               #kernel_regularizer=tf.keras.regularizers.l1(0.01)))

    result.add(tf.keras.layers.BatchNormalization(axis=3))

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    #result.add(tf.keras.layers.ReLU())

    return result


def Unet():
    down_stack = [downsample(64, 4),#, apply_batchnorm=False), # (bs, 360, 720, 64)
                  downsample(128, 4), # (bs, 180, 360, 128)
                  downsample(256, 4), # (bs, 90, 180, 256)
                  downsample(512, 4), # (bs, 45, 90, 512)
                 ]

    up_stack = [upsample(256, 4, apply_dropout=True),
                upsample(128, 4, apply_dropout=True),
                upsample(64, 4, apply_dropout=True),
               ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(1, 4, strides=2, padding='same', activation='relu')#,
                                           #kernel_initializer=initializer, activation='relu')

    concat = tf.keras.layers.Concatenate()

    inputs = tf.keras.layers.Input(shape=[None,None,3])
    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = concat([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)



model = Unet()
model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True), loss='mse')
model.fit_generator(train_dataset, epochs=50, verbose=2, validation_data=test_dataset)
