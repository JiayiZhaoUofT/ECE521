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


learning_rate = 0.005
batch_size = 500
steps = 5000

trainData_size = trainData.shape[0]
trainData = np.reshape(trainData, [-1, 784])

#model
pred = tf.matmul(X,W) + b
logistic_reg = tf.sigmoid(pred)

cross_entropy_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred,labels=Y))

train_adam = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy_loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

loss_adam = []
train_log_accuracy = []
train_log_reg_pre = []
for i in range(steps):
    batch_start_idx = (batch_size * i) % trainData_size
    batch_end_idx = batch_start_idx + batch_size

    xx_ = trainData[batch_start_idx:batch_end_idx]
    yy_ = trainTarget[batch_start_idx:batch_end_idx]

    feed = {X: xx_, Y: yy_}
    sess.run(train_adam,feed_dict=feed)
    if i % 7 == 0:
        print(i)
        feed_all = {X: trainData, Y: trainTarget}
        loss_adam.append(sess.run(cross_entropy_loss, feed_dict=feed_all))
        print('loss')
        train_log_reg_pred = sess.run(logistic_reg, feed_dict=feed_all)
        print('train_log')
        train_log_reg_tran = tf.sign(train_log_reg_pred - 0.5) * (1 / 2) + (1 / 2)
        print('train_log1')
        train_log_accuracy_ = tf.cast(tf.equal(train_log_reg_tran, trainTarget), 'float32')
        print('train_log2')
        train_log_accuracy.append(sess.run(tf.reduce_mean(train_log_accuracy_)))
        print('train log3')



plt.title('logistic regression cross-entropy loss')
plt.ylabel('cross-entropy loss')
plt.xlabel('epoch')
plt.plot(range(len(loss_adam)),loss_adam,'r--')

plt.show()


plt.title('logistic regression accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.plot(range(len(train_log_accuracy)),train_log_accuracy,'b--')
plt.show()