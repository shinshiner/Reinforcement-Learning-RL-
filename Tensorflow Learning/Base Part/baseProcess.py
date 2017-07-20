# naive training, train a list of parameter (W & b)
# with these settings, the result will converge by 2500 steps

import tensorflow as tf
import numpy as np

# construct data set
target = [0.123, 0.234, 0.345, 0.456, 0.567, 0.678, 0.789, 0.898, 0.987, 10.654]
_x = np.float32(np.random.rand(10, 1000))
_y = np.dot(target, _x) + 0.233

b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.random_uniform(shape = [1, 10], minval = -100.0, maxval = 100.0))
y = tf.matmul(W, _x) + b

# set loss
loss = tf.reduce_mean(tf.square(y - _y))

# set optimizer, namely gradient descent mode
# when the learning rate is too high, the descent will "boom" and get "nan"
optimizer = tf.train.GradientDescentOptimizer(0.15)

# set the target of training
train = optimizer.minimize(loss)

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# training
for step in xrange(0, 10001):
    sess.run(train)
    if step % 100 == 0:
        print step, sess.run(W), sess.run(b), "loss = ", sess.run(loss)
