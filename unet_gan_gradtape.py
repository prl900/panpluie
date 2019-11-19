from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K

import os
import time
import matplotlib.pyplot as plt

from nc_loader import ERA5Dataset

def Generator():
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


def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result


def Discriminator():
  initializer = tf.random_normal_initializer(0., 0.02)

  tar = tf.keras.layers.Input(shape=[None, None, 1], name='target_image')
  inp = tf.keras.layers.Input(shape=[None, None, 3], name='input_image')

  x = tf.keras.layers.concatenate([inp, tar])

  down1 = downsample(64, 4, False)(x)
  down2 = downsample(128, 4)(down1)
  down3 = downsample(256, 4)(down2)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)
  conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1)

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)
  last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad2)

  return tf.keras.Model(inputs=[inp, tar], outputs=last)



loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(disc_real_output, disc_generated_output):
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
  
  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss


def generator_loss(disc_generated_output, gen_output, target):
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  # mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

  LAMBDA = 100
  total_gen_loss = gan_loss + (LAMBDA * l1_loss)

  return total_gen_loss


generator = Generator()
discriminator = Discriminator()
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


@tf.function
def train_step(input_image, target):

  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_output = generator(input_image, training=True)
    
    disc_real_output = discriminator([input_image, target], training=True)
    disc_generated_output = discriminator([input_image, gen_output], training=True)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    gen_loss = generator_loss(disc_generated_output, gen_output, target)

  generator_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))


"""
@tf.function
def train_step(model, inputs, outputs, optimizer):

  with tf.GradientTape() as t:
    loss = tf.reduce_mean(tf.square(outputs - model(inputs, training=True)))

  grads = t.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))
"""


@tf.function
def calc_loss(model, inputs, outputs):
    return tf.reduce_mean(tf.square(outputs - model(inputs)))


"""
def train(train_dataset, test_dataset, model):
  optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, decay=1e-6, momentum=0.9, nesterov=True)

  train_loss = tf.keras.metrics.Mean()
  test_loss = tf.keras.metrics.Mean()
  
  f = open("train_record.out","w+")

  for epoch in range(100):
    for (batch, (inputs, outputs)) in enumerate(train_dataset):
      print(batch)
      train_step(model, inputs, outputs, optimizer)
      train_loss(calc_loss(model, inputs, outputs))
      
    for (inputs, outputs) in test_dataset:
      test_loss(calc_loss(model, inputs, outputs))

    template = 'Epoch {}, Loss: {:.4f}, Test Loss: {:.4f}'
    print(template.format(epoch+1, train_loss.result(), test_loss.result()))
    f.write(template.format(epoch+1, train_loss.result(), test_loss.result()))
    f.flush()

    train_loss.reset_states()
    test_loss.reset_states()

  f.close()
"""

def fit(train_ds, test_ds, epochs):
  train_loss = tf.keras.metrics.Mean()
  test_loss = tf.keras.metrics.Mean()
  template = 'Epoch {}, Loss: {:.4f}, Test Loss: {:.4f}\n'

  f = open("train_gan_record.out","w+")

  for epoch in range(epochs):
    start = time.time()

    # Train
    for batch, (inputs, target) in enumerate(train_ds):
      train_step(inputs, target)
      train_loss(calc_loss(generator, inputs, target))
    
    for batch, (inputs, target) in enumerate(test_ds):
      test_loss(calc_loss(generator, inputs, target))

    print(template.format(epoch+1, train_loss.result(), test_loss.result()))
    f.write(template.format(epoch+1, train_loss.result(), test_loss.result()))
    f.flush() 
   
    train_loss.reset_states()
    test_loss.reset_states()

    plot_output(epoch, generator, inputs, target)

    print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time()-start))

  f.close()
  generator.save('gan_generator.h5')
  discriminator.save('gan_discriminator.h5')



train_fnames = ["/home/lar116/project/ERA5_ECMWF/era5s_geop_201801.nc", 
                "/home/lar116/project/ERA5_ECMWF/era5s_geop_201802.nc", 
                "/home/lar116/project/ERA5_ECMWF/era5s_geop_201804.nc", 
                "/home/lar116/project/ERA5_ECMWF/era5s_geop_201805.nc", 
                "/home/lar116/project/ERA5_ECMWF/era5s_geop_201806.nc"] 

test_fnames = ["/home/lar116/project/ERA5_ECMWF/era5s_geop_201803.nc"]

"""
train_fnames = ["/data/ERA5/era5s_geop_201801.nc", 
                "/data/ERA5/era5s_geop_201802.nc", 
                "/data/ERA5/era5s_geop_201804.nc", 
                "/data/ERA5/era5s_geop_201805.nc", 
                "/data/ERA5/era5s_geop_201806.nc"] 

test_fnames = ["/data/ERA5/era5s_geop_201803.nc"]
"""

train_dataset = ERA5Datast(train_fnames, batch_size=4)
test_dataset = ERA5Dataset(test_fnames, batch_size=4)
EPOCHS = 50
fit(train_dataset, test_dataset, EPOCHS)
