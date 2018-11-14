import numpy as np
import json
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def get_hot_idx(arr):
    max_val = -1
    max_idx = -1
    for idx in range(0, len(arr)):
        e = arr[idx]
        if max_val < e:
            max_val = e
            max_idx = idx
    return max_idx


# prepare data
mnist = input_data.read_data_sets("./mnist_data", one_hot=True)
tf.set_random_seed(777)  # reproducibility
np.random.seed(59)

# 1000 데이터 기반으로 100개 데이터 테스트
x_vals_train, y_vals_train = mnist.train.next_batch(2000)
x_vals_test, y_vals_test = mnist.test.next_batch(100)

n_feature = len(x_vals_train[0])
n_class = len(y_vals_train[0])

x_data_train = tf.placeholder(shape=[None, n_feature], dtype=tf.float32)
y_data_train = tf.placeholder(shape=[None, n_class], dtype=tf.float32)
x_data_test = tf.placeholder(shape=[None, n_feature], dtype=tf.float32)

# manhattan distance
distance = tf.reduce_sum(tf.abs(tf.sub(x_data_train, tf.expand_dims(x_data_test, 1))), 2)

# nearest k points
k = 3
_, top_k_indices = tf.nn.top_k(tf.neg(distance), k=k)
top_k_label = tf.gather(y_data_train, top_k_indices)

sum_up_predictions = tf.reduce_sum(top_k_label, 1)
prediction = tf.argmax(sum_up_predictions, dimension=1)


# sess = tf.Session()
# prediction_outcome = sess.run(prediction, feed_dict={x_data_train: x_vals_train,
#                                                      x_data_test: x_vals_test,
#                                                      y_data_train: y_vals_train})
#
# # evaluation
# accuracy = 0
# for pred, actual in zip(prediction_outcome, y_vals_test):
#     print(pred, actual)
#     if pred == np.argmax(actual):
#         accuracy += 1
#
# print('K is', k, ':', accuracy / len(prediction_outcome))


def save_to_json_file(filename, d):
    obj = open(filename, 'wb')
    with open(filename, 'w') as outfile:
        json.dump(d, outfile)
    obj.close()


# 내 데이터로 테스트
print("데이터 불러오는 중")

with open('./data/mnist_png_testing/images.json') as data_file:
    data = json.load(data_file)
images = np.zeros((len(data), 784))
for i in range(len(data)):
    images[i] = data[i]

with open('./data/mnist_png_testing/correctValues.json') as data_file:
    data = json.load(data_file)
correct_vals = []
for i in range(len(data)):
    correct_vals.append(get_hot_idx(data[i]))

# evaluation
print("세션 생성 및 결과 계산")
res = []
batch = 0
batch_size = 100

accuracy = 0
for batch in range(0, len(images), batch_size):
    next_batch = batch + batch_size
    sess = tf.Session()
    prediction_outcome = sess.run(prediction, feed_dict={x_data_train: x_vals_train,
                                                         y_data_train: y_vals_train,
                                                         x_data_test: images[batch:next_batch]})
    print("누적 성능 보기", correct_vals[batch])
    for pred, actual in zip(prediction_outcome, correct_vals[batch:next_batch]):
        res.append([int(actual), int(pred)])
        if pred == actual:
            accuracy += 1
    print(accuracy / next_batch)

print(res)
save_to_json_file('./result/knn.json', res)

'''
확실히 KNN 예측 속도가 엄청 느리긴 하다.
성능도 낮다.
0.875
'''
