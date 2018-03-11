'''train linear regression model for classification on the two-class notMNIST dataset
   using the stochastic gradient descent (SGD) algorithm. 
   
   using the validation set to tune the hyper-parameter of the weight decay regularizer.
'''
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

with np.load("/Users/zhaojiayi/Downloads/Assignment 2/notMNIST.npz") as data:
    Data, Target = data["images"], data["labels"]
    posClass = 2
    negClass = 9
    dataIndx = (Target == posClass) + (Target == negClass)
    Data = Data[dataIndx] / 255.
    Target = Target[dataIndx].reshape(-1, 1)
    Target[Target == posClass] = 1
    Target[Target == negClass] = 0
    np.random.seed(521)
    randIndx = np.arange(len(Data))
    np.random.shuffle(randIndx)
    Data, Target = Data[randIndx], Target[randIndx]
    trainData, trainTarget = Data[:3500], Target[:3500]
    validData, validTarget = Data[3500:3600], Target[3500:3600]
    testData, testTarget = Data[3600:], Target[3600:]

traindata_size = 3500
batch_size = 500
steps = 20000
learn_rate_set = [0.005, 0.001, 0.0001]

# model
x = tf.placeholder(tf.float32, [None, 784])
w = tf.Variable(tf.zeros([784, 1]))
b = tf.Variable(tf.zeros([1]))
y_ = tf.placeholder(tf.float32, [None, 1])
N = batch_size
y = tf.matmul(x, w) + b
learn_rate = tf.placeholder(tf.float32)

cost = tf.reduce_mean(tf.square(y_ - y)) * (1 / 2)

# train
train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(cost)
sess = tf.Session()

error = [[], [], []]
for j in range(3):
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(steps):
        batch_start_idx = (i * batch_size) % traindata_size
        batch_end_idx = batch_start_idx + batch_size
        if batch_end_idx > traindata_size:
            batch_end_idx = batch_end_idx - traindata_size
            batch_xs = trainData[batch_start_idx:traindata_size]
            batch_xs = np.append(trainData[:batch_end_idx])

            batch_xs = trainTarget[batch_start_idx:traindata_size]
            batch_xs = np.append(trainTarget[:batch_end_idx])
        else:
            batch_xs = trainData[batch_start_idx:batch_end_idx]
            batch_ys = trainTarget[batch_start_idx:batch_end_idx]

        batch_xs = np.reshape(batch_xs, [-1, 784])
        xs, ys = np.array(batch_xs), np.array(batch_ys)

        feed = {x: xs, y_: ys, learn_rate: learn_rate_set[j]}
        sess.run(train_step, feed_dict=feed)

        x_all = np.reshape(trainData, [-1, 784])
        y_all = np.array(trainTarget)
        feed_all = {x: x_all, y_: y_all}

        if i % 7 == 0:
            error_ = sess.run(cost, feed_dict=feed_all)
            error[j].append(error_)

x = len(error[0])
xx = range(x)
plt.xlabel('epoch')
plt.ylabel('training error')
plt.plot(xx, error[0], 'g--', label='learning rate at 0.005')
plt.plot(xx, error[1], 'r--', label='learning rate at 0.001')
plt.plot(xx, error[2], 'b--', label='learning rate at 0.0001')
plt.legend(loc='upper right')
plt.show()
