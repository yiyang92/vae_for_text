from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
from tensorflow import layers
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

# dilated convolution mnist

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# placeholders
x = tf.placeholder(shape=[None, None], dtype=tf.float32, name='raw_input')
y = tf.placeholder(shape=[None], dtype=tf.int64, name='raw_labels')
# dilated convolutions [1, 2, 4]
x_image = tf.reshape(x, [-1, 28, 28, 1])
conv1 = layers.conv2d(x_image, 15, [3, 3], padding="same", dilation_rate=(1, 1), activation=tf.nn.relu)
conv2 = layers.conv2d(conv1, 20, [1, 1], padding="same", dilation_rate=(2, 2), activation=tf.nn.relu)
conv3 = layers.conv2d(conv2, 10, [1, 1], padding="same", dilation_rate=(4, 4), activation=tf.nn.relu)
# prnt = tf.Print(conv3, [tf.shape(conv3)])
shape = [tf.shape(conv1), tf.shape(conv2), tf.shape(conv3)]
# last layer
ff = tf.contrib.layers.flatten(conv3)
ff1 = layers.dense(ff, 200, tf.nn.relu)
ff2 = layers.dense(ff1, 100, activation=tf.nn.relu)
ff3 = layers.dense(ff2, 10, activation=None)
# loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=ff3))
# optimizer
optimize = tf.train.AdamOptimizer().minimize(loss)
# launch graph
epochs = 4
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for e in range(epochs):
        for _ in range(1000):
            batch = mnist.next_batch(100)
            loss_, _, shape_ = sess.run([loss, optimize, shape],feed_dict={x:batch[0], y:batch[0]})
        acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(ff3, 1), tf.argmax(y, 1)), tf.float32))
        if e == 0: print(shape_)
        print("For epoch number: {} accuracy is: {} loss is: {} ".format(e, acc.eval(feed_dict={x: mnist.train.images, y: mnist.train.labels}), loss_))



