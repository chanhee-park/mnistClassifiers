# http://solarisailab.com/archives/303
# 텐서플로우(TensorFlow)를 이용한 MNIST 문자 인식 프로그램 만들기

# TensorFlow 라이브러리를 추가한다.
import tensorflow as tf
import json
import numpy as np

# MNIST 데이터를 다운로드 한다.
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./mnist_data", one_hot=True)

# 변수들을 설정한다.
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

# cross-entropy 모델을 설정한다.
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# 경사하강법으로 모델을 학습한다.
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# 학습된 모델이 얼마나 정확한지를 출력한다.
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

'''
0.9181
'''


# 내 데이터로 테스트
print("내 데이터로 테스트")

with open('./data/test2000_images.json') as data_file:
    data = json.load(data_file)
images = np.zeros((len(data), 784))
for i in range(len(data)):
    images[i] = data[i]

with open('./data/test2000_correctValues.json') as data_file:
    data = json.load(data_file)
correct_vals = np.zeros((len(data), 10))
for i in range(len(data)):
    correct_vals[i] = data[i]

prediction = tf.argmax(y, 1)
print(sess.run(prediction, feed_dict={x: [images[0]], y_: [correct_vals[0]]}),
      correct_vals[0][sess.run(prediction, feed_dict={x: [images[0]], y_: [correct_vals[0]]})] == 1.)

print(sess.run(prediction, feed_dict={x: [images[1]], y_: [correct_vals[1]]}),
      correct_vals[1][sess.run(prediction, feed_dict={x: [images[1]], y_: [correct_vals[1]]})] == 1.)

print(sess.run(prediction, feed_dict={x: [images[2]], y_: [correct_vals[2]]}),
      correct_vals[2][sess.run(prediction, feed_dict={x: [images[2]], y_: [correct_vals[2]]})] == 1.)

print(sess.run(prediction, feed_dict={x: images, y_: correct_vals}))

print(sess.run(accuracy, feed_dict={x: images, y_: correct_vals}))
'''
0.91059226
'''

