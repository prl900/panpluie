from __future__ import absolute_import, division, print_function, unicode_literals

from keras import backend as K
import tensorflow as tf

import os
import time
import matplotlib.pyplot as plt

BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256

import numpy as np

def load(image_file):
  image = tf.io.read_file(image_file)
  image = tf.image.decode_jpeg(image)

  w = tf.shape(image)[1]

  w = w // 2
  real_image = image[:, :w, :]
  input_image = image[:, w:, :]

  input_image = tf.cast(input_image, tf.float32)
  patch = np.zeros((256, 256, 3), dtype=np.float32)
  patch[100:150, 150:200,:] = np.nan
  real_image = tf.add(tf.cast(real_image, tf.float32), tf.constant(patch, dtype=np.float32))
  #real_image = tf.cast(real_image, tf.float32)

  return input_image, real_image

# normalizing the images to [-1, 1]
def normalize(input_image, real_image):
  input_image = (input_image / 127.5) - 1
  real_image = (real_image / 127.5) - 1

  return input_image, real_image

#@tf.function()
def load_image_train(image_file):
  input_image, real_image = load(image_file)
  #input_image, real_image = random_jitter(input_image, real_image)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image


def load_image_test(image_file):
  input_image, real_image = load(image_file)
  #input_image, real_image = resize(input_image, real_image, IMG_HEIGHT, IMG_WIDTH)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image

train_dataset = tf.data.Dataset.list_files(PATH+'train/*.jpg')
train_dataset = train_dataset.map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.cache().shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(1)
#print(tf.shape(train_dataset))
#exit()

test_dataset = tf.data.Dataset.list_files(PATH+'test/*.jpg')
test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.batch(1)
print(test_dataset)


OUTPUT_CHANNELS = 3


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

def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result

def Generator():
  down_stack = [
    downsample(64, 4, apply_batchnorm=False), # (bs, 128, 128, 64)
    downsample(128, 4), # (bs, 64, 64, 128)
    downsample(256, 4), # (bs, 32, 32, 256)
    downsample(512, 4), # (bs, 16, 16, 512)
    downsample(512, 4), # (bs, 8, 8, 512)
    #downsample(512, 4), # (bs, 4, 4, 512)
    #downsample(512, 4), # (bs, 2, 2, 512)
    #downsample(512, 4), # (bs, 1, 1, 512)
  ]

  up_stack = [
    #upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
    #upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
    #upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
    upsample(512, 4, apply_dropout=True), # (bs, 16, 16, 1024)
    upsample(256, 4, apply_dropout=True), # (bs, 32, 32, 512)
    upsample(128, 4), # (bs, 64, 64, 256)
    upsample(64, 4), # (bs, 128, 128, 128)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh') # (bs, 256, 256, 3)

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

def generator_loss(gen_output, target):
  # mean square error
  return mse_holes(target, gen_output)


generator = Generator()
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

EPOCHS = 150

def generate_images(model, test_input, tar, i=0):
  # the training=True is intentional here since
  # we want the batch statistics while running the model
  # on the test dataset. If we use training=False, we will get
  # the accumulated statistics learned from the training dataset
  # (which we don't want)
  prediction = model(test_input, training=True)
  plt.imsave("pred_{:02d}.png".format(i), np.array(prediction[0] * 0.5 + 0.5)) 
  plt.imsave("targ_{:02d}.png".format(i), np.array(tar[0] * 0.5 + 0.5)) 
  return  
  plt.imsave("test_{:02d}".format(i), test_input[0]) 
  
  plt.figure(figsize=(15,15))

  display_list = [test_input[0], tar[0], prediction[0]]
  title = ['Input Image', 'Ground Truth', 'Predicted Image']

  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    # getting the pixel values between [0, 1] to plot it.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.show()

@tf.function
def train_step(input_image, target):

  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_output = generator(input_image, training=True)

    gen_loss = generator_loss(gen_output, target)

  generator_gradients = gen_tape.gradient(gen_loss,
                                          generator.trainable_variables)

  generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))


def fit(train_ds, epochs, test_ds):
  for epoch in range(epochs):
    print(epoch)
    start = time.time()

    # Train
    for input_image, target in train_ds:
      train_step(input_image, target)

    #clear_output(wait=True)
    # Test on the same image so that the progress of the model can be 
    # easily seen.
    for example_input, example_target in test_ds.take(1):
      print("generate image called")
      generate_images(generator, example_input, example_target, epoch)

    print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time()-start))


fit(train_dataset, EPOCHS, test_dataset)
