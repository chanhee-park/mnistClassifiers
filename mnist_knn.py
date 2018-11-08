import numpy as np
import tensorflow as tf
import json

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./mnist_data", one_hot=True)

# 1000개 이미지 학습, 100개 이미지 테스트
train_pixels, train_list_values = mnist.train.next_batch(1000)
test_pixels, test_list_of_values = mnist.test.next_batch(100)

# 텐서 정의
train_pixel_tensor = tf.placeholder("float", [None, 784])
test_pixel_tensor = tf.placeholder("float", [784])

# 비용함수 정의 텐서의 차원을 탐색하며 개체들의 총합 계산
# _reduce_sum 함수: 텐서의 차원을 탐색하며 개체의 총합 계산
distance = tf.reduce_sum(tf.abs(tf.add(train_pixel_tensor, tf.neg(test_pixel_tensor))), reduction_indices=1)

# 비용함수 최소화를 위해 arg_min 사용 가장 작은 거리를 갖는 인덱스 리턴(최근접 이웃)
pred = tf.arg_min(distance, 0)

# 학습
accuracy = 0
init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    for i in range(len(test_list_of_values)):
        nn_index = sess.run(pred, feed_dict={train_pixel_tensor: train_pixels, test_pixel_tensor: test_pixels[i, :]})
        print("Test No. ", i, "Predict Class: ", np.argmax(train_list_values[nn_index]), "True class: ",
              np.argmax(test_list_of_values[i]))
        if np.argmax(train_list_values[nn_index]) == np.argmax(test_list_of_values[i]):
            accuracy += 1.0 / len(test_pixels)
    print("Result Accuracy =", accuracy)

'''
Result Accuracy = 0.9100000000000006
'''

# 내 데이터로 테스트
print("내 데이터로 테스트")

# with open('./data/test2000_images.json') as data_file:
#     data = json.load(data_file)
# images = np.zeros((len(data), 784))
# for i in range(len(data)):
#     images[i] = data[i]

# with open('./data/test2000_correctValues.json') as data_file:
#     data = json.load(data_file)
# correct_vals = np.zeros((len(data), 10))
# for i in range(len(data)):
#     correct_vals[i] = data[i]
#
# with tf.Session() as sess:
#     sess.run(init)
#     for i in range(len(correct_vals)):
#         nn_index = sess.run(pred, feed_dict={train_pixel_tensor: train_pixels, test_pixel_tensor: images[i, :]})
#         print("Test No. ", i, "Predict Class: ", np.argmax(train_list_values[nn_index]), "True class: ",
#               np.argmax(correct_vals[i]))
#         if np.argmax(train_list_values[nn_index]) == np.argmax(correct_vals[i]):
#             accuracy += 1.0 / len(images)
#     print("내 데이터로 테스트 ", accuracy)

'''
0.91287017
'''
