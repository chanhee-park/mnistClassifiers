# http://solarisailab.com/archives/1182
# 텐서플로우(TensorFlow)를 이용해서 MNIST 숫자 분류를 위한 Stacked Autoencoders 구현해보기

# http://solarisailab.com/archives/113
# [UFLDL Tutorial – 1. 오토인코더(Sparse Autoencoder) 1 – AutoEncoders & Sparsity]

import tensorflow as tf
import json
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./mnist_data", one_hot=True)

# 파라미터 설정
learning_rate_RMSProp = 0.01
learning_rate_Gradient_Descent = 0.5
training_epochs = 50  # epoch 횟수 (iteration)
softmax_classifier_iterations = 1000  # Softmax Classifier iteration 횟수
batch_size = 256
display_step = 5  # 몇 Step마다 log를 출력할지 결정한다.
examples_to_show = 10  # reconstruct된 이미지 중 몇개를 보여줄지를 결정한다.
n_hidden_1 = 200  # 첫번째 히든레이어의 노드 개수
n_hidden_2 = 200  # 두번째 히든레이어의 노드 개수
n_input = 784  # MNIST 데이터 input (이미지 크기: 28*28)


# Stacked Autoencoder를 생성한다.
def build_autoencoder():
    # 히든 레이어 1을 위한 Weights와 Biases
    Wh_1 = tf.Variable(tf.random_normal([n_input, n_hidden_1]))
    bh_1 = tf.Variable(tf.random_normal([n_hidden_1]))
    h_1 = tf.nn.sigmoid(tf.matmul(X, Wh_1) + bh_1)  # 히든레이어 1의 activation (sigmoid 함수를 사용)
    # 히든 레이어 2을 위한 Weights와 Biases
    Wh_2 = tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]))
    bh_2 = tf.Variable(tf.random_normal([n_hidden_2]))
    h_2 = tf.nn.sigmoid(tf.matmul(h_1, Wh_2) + bh_2)  # 히든레이어 2의 activation (sigmoid 함수를 사용)
    # Output 레이어를 위한 Weights와 Biases
    Wo = tf.Variable(tf.random_normal([n_hidden_2, n_input]))
    bo = tf.Variable(tf.random_normal([n_input]))
    X_reconstructed = tf.nn.sigmoid(tf.matmul(h_2, Wo) + bo)  # Output 레이어의 activation (sigmoid 함수를 사용)
    return X_reconstructed, h_2


# Softmax Classifier를 생성한다.
def build_softmax_classifier():
    # Softmax Classifier를 위한 파라미터들
    W = tf.Variable(tf.zeros([n_hidden_2, 10]))
    b = tf.Variable(tf.zeros([10]))
    y_pred = tf.nn.softmax(
        tf.matmul(extracted_features, W) + b)  # 예측된 Output : 두번째 히든레이어의 activation output을 input으로 사용한다.
    return y_pred


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


# 학습에 필요한 변수들 설정
X = tf.placeholder("float", [None, n_input])  # Input 데이터 설정
y_pred, extracted_features = build_autoencoder()  # Autoencoder의 Reconstruction 결과, 압축된 Features(h_2=200)
y_true = X  # Output 값(True Output)을 설정(=Input 값)
y = build_softmax_classifier()  # Predicted Output using Softmax Classifier
y_ = tf.placeholder(tf.float32, [None, 10])  # True Output

# Optimization을 위한 파라미터들
# Autoencoder Optimization을 위한 파라미터들
reconsturction_cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))  # squared error loss 함수
initial_optimizer = tf.train.RMSPropOptimizer(learning_rate_RMSProp).minimize(reconsturction_cost)

# Softmax Classifier Optimization을 위한 파라미터들
cross_entropy_cost = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))  # cross-entropy loss 함수
softmax_classifier_optimizer = tf.train.GradientDescentOptimizer(learning_rate_Gradient_Descent).minimize(
    cross_entropy_cost)

# Fine Tuning Optimization을 위한 파라미터들
finetuning_cost = cross_entropy_cost + reconsturction_cost
finetuning_optimizer = tf.train.GradientDescentOptimizer(learning_rate_Gradient_Descent).minimize(finetuning_cost)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    # 변수들을 초기화한다.
    sess.run(init)

    # Step 1: Stacked Autoencoder pre-training
    total_batch = int(mnist.train.num_examples / batch_size)
    # Training을 시작한다.
    for epoch in range(training_epochs):
        # 모든 배치들을 돌아가면서(Loop) 학습한다.
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # batch 데이터를 이용해서 트레이닝을 진행한다.
            _, cost_value = sess.run([initial_optimizer, reconsturction_cost], feed_dict={X: batch_xs})
        # 일정 epoch step마다 로그를 출력한다.
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(cost_value))
    print("Stacked Autoencoder pre-training Optimization Finished!")

    # Step 2: test 데이터셋을 autoencoder로 reconstruction 해본다.
    reconstructed_image = sess.run(y_pred, feed_dict={X: mnist.test.images[:examples_to_show]})

    # Step 3: Softmax Classifier를 학습한다.
    for i in range(softmax_classifier_iterations):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(softmax_classifier_optimizer, feed_dict={X: batch_xs, y_: batch_ys})
    print("Softmax Classifier Optimization Finished!")

    # Step 4: 학습된 모델이 얼마나 정확한지를 출력한다. (Before fine-tuning)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy(before fine-tuning): ")  # Accuracy ~ 0.9282
    print(sess.run(accuracy, feed_dict={X: mnist.test.images, y_: mnist.test.labels}))

    # Step 5: Fine-tuning softmax model
    # Training을 시작한다.
    for epoch in range(training_epochs):
        # 모든 배치들을 돌아가면서(Loop) 학습한다.
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # batch 데이터를 이용해서 트레이닝을 진행한다.
            _, cost_value = sess.run([finetuning_optimizer, finetuning_cost], feed_dict={X: batch_xs, y_: batch_ys})
        # 일정 epoch step마다 로그를 출력한다.
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(cost_value))
    print("Fine-tuning softmax model Optimization Finished!")

    # Step 6: 학습된 모델이 얼마나 정확한지를 출력한다. (After fine-tuning)
    print("Accuracy(after fine-tuning): ")  # Accuracy ~ 0.9714
    print(sess.run(accuracy, feed_dict={X: mnist.test.images, y_: mnist.test.labels}))

    '''
    Extracting ./mnist_data/train-images-idx3-ubyte.gz
    Extracting ./mnist_data/train-labels-idx1-ubyte.gz
    Extracting ./mnist_data/t10k-images-idx3-ubyte.gz
    Extracting ./mnist_data/t10k-labels-idx1-ubyte.gz
    Epoch: 0001 cost= 0.178233355
    Epoch: 0002 cost= 0.146527156
    Epoch: 0003 cost= 0.133671910
    Epoch: 0004 cost= 0.127620995
    Epoch: 0005 cost= 0.126519501
    Epoch: 0006 cost= 0.123915642
    Epoch: 0007 cost= 0.120452702
    Epoch: 0008 cost= 0.117654808
    Epoch: 0009 cost= 0.117769420
    Epoch: 0010 cost= 0.115769677
    Epoch: 0011 cost= 0.114197388
    Epoch: 0012 cost= 0.108472437
    Epoch: 0013 cost= 0.109850086
    Epoch: 0014 cost= 0.105722561
    Epoch: 0015 cost= 0.102819115
    Stacked Autoencoder pre-training Optimization Finished!
    Softmax Classifier Optimization Finished!
    Accuracy(before fine-tuning): 0.8693 (학습 많이하고 fine-tuning 하면 95% 이상 나옴)
    '''

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
    score = 0
    for i in range(0, len(correct_vals)):
        real = int(get_hot_idx(correct_vals[i]))
        predict = int(sess.run(prediction, feed_dict={X: images[i:i + 1], y_: correct_vals[i:1 + 1]})[0])
        if real == predict:
            score = score + 1
        res.append([real, predict])

    print(score / 10000)
    '''0.9493'''
    save_to_json_file('./result/stacked_autoencoder.json', res)
