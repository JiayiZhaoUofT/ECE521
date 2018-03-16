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

W = tf.Variable(tf.zeros([784,1]))
b = tf.Variable(tf.zeros([1]))


learning_rate = 0.001
batch_size = 500

steps = 20000
trainData_size = trainData.shape[0]
trainData = np.reshape(trainData, [-1, 784])

pred = tf.matmul(X,W) + b
loss = tf.reduce_mean(tf.square(pred - Y))

train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
mse = []
train_linear_accuracy= []

for i in range(steps):
    batch_start_idx = (batch_size * i) % trainData_size
    batch_end_idx = batch_start_idx + batch_size

    xx_ = trainData[batch_start_idx:batch_end_idx]
    yy_ = trainTarget[batch_start_idx:batch_end_idx]

    feed = {X: xx_, Y: yy_}
    sess.run(train, feed_dict=feed)

    if i % 7 == 0:
        feed_all = {X: trainData, Y: trainTarget}
        mse.append(sess.run(loss, feed_dict= feed_all))

        train_linear_reg_pred = sess.run(pred, feed_dict=feed_all)
        train_linear_reg_tran = tf.sign(train_linear_reg_pred - 0.5) * (1 / 2) + (1 / 2)
        train_linear_accuracy_ = tf.cast(tf.equal(train_linear_reg_tran, trainTarget), 'float32')
        train_linear_accuracy.append(sess.run(tf.reduce_mean(train_linear_accuracy_)))

plt.title('linear regression mse')
plt.ylabel('mse')
plt.xlabel('epoch')
plt.plot(range(len(mse)),mse,'r--')
plt.show()


plt.title('linear regression accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.plot(range(len(train_linear_accuracy)),train_linear_accuracy,'b--')
plt.show()