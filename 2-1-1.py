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


#initialize
X = tf.placeholder('float32',[None,784])
Y = tf.placeholder('float32',[None,1])
learning_rate = tf.placeholder('float32')

#initial w as 1
W = tf.Variable(tf.truncated_normal(shape=[784, 1], stddev = 0.5))
b = tf.Variable(tf.zeros([1]))

#data in training
lamda_para = 0.01
batch_size = 500
steps = 5000
trainData_size = trainData.shape[0]
trainData = np.reshape(trainData,[-1,784])
learning_rate_set = [0.005,0.001,0.0001]
#logistic pred
pred = tf.matmul(X,W) + b
logistic_reg = tf.sigmoid(pred)
#loss
cross_entropy_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred,labels=Y))
weight_loss = lamda_para * tf.reduce_sum(tf.square(W))
total_loss = (cross_entropy_loss + weight_loss) * (1/2)

#train model
train = tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)

#reshape data
xx = np.reshape(trainData,[-1,784])
yy = np.reshape(trainTarget,[-1,1])

#trianing
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

accuracy_set = [[],[],[]]
loss_set = [[],[],[]]


#training
for j in range(3):
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(steps):
        batch_start_idx = (batch_size * i)% trainData_size
        batch_end_idx = batch_start_idx + batch_size

        xx_ = trainData[batch_start_idx:batch_end_idx]
        yy_ = trainTarget[batch_start_idx:batch_end_idx]

        feed = {X: xx_, Y: yy_, learning_rate: learning_rate_set[j]}
        sess.run(train,feed_dict=feed)
  # validation
    validData = np.reshape(validData, [-1, 784])
    feed_valid = {X: validData, Y: validTarget, learning_rate: learning_rate_set[j]}
    valid_pred = sess.run(logistic_reg, feed_dict=feed_valid)
    valid_pred_tran = tf.sign(valid_pred - 0.5) * (1 / 2) + (1 / 2)
    accuracy_set[j] = tf.cast(tf.equal(valid_pred_tran, validTarget), 'float32')
    accuracy_set[j] = tf.reduce_mean(accuracy_set[j])
    loss_set[j] = sess.run(total_loss, feed_dict=feed_valid)
    print('when learning rate is',learning_rate_set[j])
    print('accuracy= ')
    print(sess.run(accuracy_set[j]))
    print('cross entropy loss= ')
    print(loss_set[j])

best_learning_rate = 0.005
train_loss =[]
train_accuracy = []
valid_loss = []
valid_accuracy = []

init = tf.global_variables_initializer()
sess.run(init)
for f in range(steps):
    print(f)
    batch_start_idx = (batch_size * f) % trainData_size
    batch_end_idx = batch_start_idx + batch_size

    xx_ = trainData[batch_start_idx:batch_end_idx]
    yy_ = trainTarget[batch_start_idx:batch_end_idx]

    feed2 = {X: xx_, Y: yy_, learning_rate: best_learning_rate}
    sess.run(train, feed_dict=feed2)

    feed3 = {X: trainData, Y: trainTarget,learning_rate:best_learning_rate}
    if f%7 == 0:
       loss = sess.run(total_loss, feed_dict=feed3)
       train_loss.append(loss)
       train_pred = sess.run(logistic_reg,feed_dict=feed3)
       train_pred_tran = tf.sign(train_pred - 0.5) * (1/2) + (1/2)
       train_accuracy1 = tf.cast(tf.equal(train_pred_tran, trainTarget),'float32')
       train_accuracy.append(sess.run(tf.reduce_mean(train_accuracy1)))

       feed4 = {X: validData, Y:validTarget, learning_rate:best_learning_rate}
       loss_valid = sess.run(total_loss, feed_dict= feed4)
       valid_loss.append(loss_valid)
       valid_pred1 = sess.run(logistic_reg,feed_dict=feed4)
       valid_pred_tran1 = tf.sign(valid_pred1 - 0.5) * (1/2) + (1/2)
       valid_accuracy1 = tf.cast(tf.equal(valid_pred_tran1, validTarget),'float32')
       valid_accuracy.append(sess.run(tf.reduce_mean(valid_accuracy1)))

print('test')
testData = np.reshape(testData,[-1,784])
print('1')
feed_test = {X: testData, Y:testTarget}
test_pred = sess.run(logistic_reg,feed_dict= feed_test)
print('2')
test_pred_tran = tf.sign(test_pred - 0.5) * (1/2) + (1/2)
test_accuracy1 = tf.cast(tf.equal(test_pred_tran, testTarget),'float32')
test_accuracy1 = tf.reduce_mean(test_accuracy1)
print('test accuracy: ',sess.run(test_accuracy1))


length = len(train_loss)

plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(range(length), train_loss,'b--',label = 'training loss')
plt.plot(range(length), valid_loss,'r--',label = 'valid loss')
plt.legend(loc='upper right')
plt.show()

plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.plot(range(length), train_accuracy,'b--',label = 'training accuracy')
plt.plot(range(length),valid_accuracy,'r--', label = 'valid accuracy')
plt.legend(loc='upper right')
plt.show()