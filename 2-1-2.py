import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

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

X = tf.placeholder('float32', [None,784])
Y = tf.placeholder('float32')

W = tf.Variable(tf.truncated_normal(shape=[784, 1], stddev = 0.5))
b = tf.Variable(tf.zeros([1]))

learning_rate = 0.001
batch_size = 500
lambda_para = 0.01
steps = 5000
trainData_size = trainData.shape[0]
trainData = np.reshape(trainData, [-1, 784])

#model
pred = tf.matmul(X,W) + b
logistic_reg = tf.sigmoid(pred)

cross_entropy_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred,labels=Y))
weight_loss = lambda_para * tf.reduce_sum(tf.square(W))
total_loss = (cross_entropy_loss + weight_loss) * (1/2)

#train model
train_adam = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_loss)
train_SGD = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(total_loss)

loss_adam = []
loss_SGD = []

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
for i in range(steps):
    batch_start_idx = (batch_size * i) % trainData_size
    batch_end_idx = batch_start_idx + batch_size

    xx_ = trainData[batch_start_idx:batch_end_idx]
    yy_ = trainTarget[batch_start_idx:batch_end_idx]

    feed = {X: xx_, Y: yy_}
    sess.run(train_adam,feed_dict=feed)

    if i % 7 == 0:
        feed_all = {X:trainData, Y:trainTarget}
        loss_adam.append(sess.run(total_loss, feed_dict= feed))


init = tf.global_variables_initializer()
sess.run(init)

for j in range(steps):
    batch_start_idx1 = (batch_size * j) % trainData_size
    batch_end_idx1 = batch_start_idx1 + batch_size

    xx_1 = trainData[batch_start_idx1:batch_end_idx1]
    yy_1 = trainTarget[batch_start_idx1:batch_end_idx1]
    feed1 = {X: xx_1, Y: yy_1}

    sess.run(train_SGD, feed_dict= feed1)

    feed_all1 = {X: trainData, Y: trainTarget}
    if j % 7 == 0:
        loss_SGD.append(sess.run(total_loss, feed_dict= feed_all1))

plt.xlabel('epoch')
plt.ylabel('entropy cross loss')
plt.plot(range(len(loss_adam)), loss_adam, 'r--', label = 'Adam Optimizer')
plt.legend(loc='upper right')
plt.show()

plt.xlabel('epoch')
plt.ylabel('entropy cross loss')
plt.plot(range(len(loss_SGD)), loss_SGD,'b--', label = 'SGD')
plt.legend(loc='upper right')
plt.show()

plt.xlabel('epoch')
plt.ylabel('entropy cross loss')
plt.plot(range(len(loss_adam)), loss_adam, 'r--', label = 'Adam Optimizer')
plt.plot(range(len(loss_SGD)), loss_SGD,'b--', label = 'SGD')
plt.legend(loc='upper right')
plt.show()


