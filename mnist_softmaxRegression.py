# http://solarisailab.com/archives/303
# 텐서플로우(TensorFlow)를 이용한 MNIST 문자 인식 프로그램 만들기

# TensorFlow 라이브러리를 추가한다.
import tensorflow as tf
import json
import numpy as np

# MNIST 데이터를 다운로드 한다.
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./mnist_data", one_hot=True)

# parameters
training_epochs = 1000
batch_size = 100
learning_rate = 0.05

# 변수들을 설정한다.
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

# cross-entropy 모델을 설정한다.
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

# 경사하강법으로 모델을 학습한다.
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
for i in range(training_epochs):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# 학습된 모델이 얼마나 정확한지를 출력한다.
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

'''
0.9181
'''


def get_hot_idx(arr):
    max_val = -1
    max_idx = -1
    for idx in range(0, len(arr)):
        e = arr[idx]
        if max_val < e:
            max_val = e
            max_idx = idx
    return max_idx


def save_to_json_file(filename, d):
    obj = open(filename, 'wb')
    with open(filename, 'w') as outfile:
        json.dump(d, outfile)
    obj.close()


# 내 데이터로 테스트
print("내 데이터로 테스트")

with open('./data/mnist_png_testing/images.json') as data_file:
    data = json.load(data_file)
images = np.zeros((len(data), 784))
for i in range(len(data)):
    images[i] = data[i]

with open('./data/mnist_png_testing/correctValues.json') as data_file:
    data = json.load(data_file)
correct_vals = np.zeros((len(data), 10))
for i in range(len(data)):
    correct_vals[i] = data[i]

prediction = tf.argmax(y, 1)

res = []
correct_times = 0
for i in range(0, len(correct_vals)):
    real = int(get_hot_idx(correct_vals[i]))
    predict = int(sess.run(prediction, feed_dict={x: [images[i]], y_: [correct_vals[i]]})[0])
    res.append([real, predict])
    if real == predict:
        correct_times = correct_times+1

save_to_json_file('./result/softmax_regression.json', res)

print(correct_times/len(correct_vals))
'''
0.8912
'''
