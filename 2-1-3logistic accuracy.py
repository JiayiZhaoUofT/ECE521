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

train_adam = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_loss)

loss_adam = []


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

trainData = np.reshape(trainData,[-1,784])
validData = np.reshape(validData,[-1,784])
testData = np.reshape(testData,[-1,784])

feed1 = {X:trainData}
feed2 = {X:validData}
feed3 = {X:testData}

#train accuracy
train_linear_reg_pred = sess.run(logistic_reg,feed_dict= feed1)
train_linear_reg_tran = tf.sign(train_linear_reg_pred - 0.5) * (1/2) + (1/2)
train_linear_accuracy = tf.cast(tf.equal(train_linear_reg_tran, trainTarget),'float32')
train_linear_accuracy = tf.reduce_mean(train_linear_accuracy)
print('the logistic regression training accuracy is: ', sess.run(train_linear_accuracy))

#validation accuracy
valid_linear_reg_pred = sess.run(logistic_reg,feed_dict= feed2)
valid_linear_reg_tran = tf.sign(valid_linear_reg_pred - 0.5) * (1/2) + (1/2)
valid_linear_accuracy = tf.cast(tf.equal(valid_linear_reg_tran, validTarget),'float32')
valid_linear_accuracy = tf.reduce_mean(valid_linear_accuracy)
print('the logistic validation accuracy is: ',sess.run(valid_linear_accuracy))


#test accuracy
test_linear_reg_pred = sess.run(logistic_reg,feed_dict= feed3)
test_linear_reg_tran = tf.sign(test_linear_reg_pred - 0.5) * (1/2) + (1/2)
test_linear_accuracy = tf.cast(tf.equal(test_linear_reg_tran, testTarget),'float32')
test_linear_accuracy = tf.reduce_mean(test_linear_accuracy)
print('the logistic test accuracy is: ',sess.run(test_linear_accuracy))
