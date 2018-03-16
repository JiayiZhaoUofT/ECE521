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



trainData = np.reshape(trainData,[-1,784])
one = np.ones([3500,1])
xx = np.column_stack((one,trainData))
yy = np.array(trainTarget)

validData = np.reshape(validData,[-1,784])
one1 = np.ones([100,1])
xx_1 = np.column_stack((one1,validData))
yy_1 = np.array(validTarget)


testData = np.reshape(testData,[-1,784])
d = testData.shape[0]
one2 = np.ones([d,1])
xx_2 = np.column_stack((one2,testData))
yy_2 = np.array(testTarget)

x = xx
y = yy
w_star = np.dot(np.dot(np.linalg.inv(np.dot(x.T,x)),x.T), yy)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

y1 = np.dot(x, w_star)
accuracy1 = 0
for i in range(3500):
    if y1[i] > 0.5 and trainTarget[i] == 1:
        accuracy1 = accuracy1 + 1
    if y1[i] < 0.5 and trainTarget[i] == 0:
        accuracy1 = accuracy1 + 1

print(accuracy1/3500)


y2 = np.dot(xx_1, w_star)

accuracy2 = 0
for i in range(100):
    if y2[i] > 0.5 and validTarget[i] == 1:
        accuracy2 = accuracy2 + 1
    if y2[i] < 0.5 and validTarget[i] == 0:
        accuracy2 = accuracy2 + 1

print(accuracy2/100)

y3 = np.dot(xx_2, w_star)
accuracy3 = 0
length = xx_2.shape[0]
for i in range(length):
    if y3[i] > 0.5 and testTarget[i] == 1:
        accuracy3 = accuracy3 + 1
    if y3[i] < 0.5 and testTarget[i] == 0:
        accuracy3 = accuracy3 + 1

print(accuracy3/length)


'''
#train accuracy
train_linear_reg_pred = xx * w_star
train_linear_reg_tran = tf.sign(train_linear_reg_pred - 0.5) * (1/2) + (1/2)
train_linear_accuracy = tf.cast(tf.equal(train_linear_reg_tran, trainTarget),'float32')
train_linear_accuracy = tf.reduce_mean(train_linear_accuracy)
print('the linear regression training accuracy is: ',sess.run(train_linear_accuracy))

#validation accuracy
feed_valid = {X:xx_1,Y:validTarget}
valid_linear_reg_pred = sess.run(liner_reg_pred,feed_dict= feed_valid)
valid_linear_reg_tran = tf.sign(valid_linear_reg_pred - 0.5) * (1/2) + (1/2)
valid_linear_accuracy = tf.cast(tf.equal(valid_linear_reg_tran, validTarget),'float32')
valid_linear_accuracy = tf.reduce_mean(valid_linear_accuracy)
print('the linear regression validation accuracy is: ',sess.run(valid_linear_accuracy))

#test accuracy

feed_test = {X:xx_2, Y:testTarget}
test_linear_reg_pred = sess.run(liner_reg_pred,feed_dict= feed_test)
test_linear_reg_tran = tf.sign(test_linear_reg_pred - 0.5) * (1/2) + (1/2)
test_linear_accuracy = tf.cast(tf.equal(test_linear_reg_tran, testTarget),'float32')
test_linear_accuracy = tf.reduce_mean(test_linear_accuracy)

print('the linear regression test accuracy is: ',sess.run(test_linear_accuracy))



'''


