# Deep-Learning-Course
import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf



boston_housing=tf.keras.datasets.boston_housing

(train_x,train_y),(test_x,test_y)=boston_housing.load_data()

# print(train_x)



num_train=len(train_x)

num_test=len(test_x)

train_x1=train_x[:,12]

test_x1=test_x[:,12]

train_x2=train_x[:,5]

test_x2=test_x[:,5]

train_x1=(train_x1-train_x1.min())/(train_x1.max()-train_x1.min())

test_x1=(test_x1-test_x1.min())/(test_x1.max()-test_x1.min())

train_x2=(train_x2-train_x2.min())/(train_x2.max()-train_x2.min())

test_x2=(test_x2-test_x2.min())/(test_x2.max()-test_x2.min())



train_x1=tf.reshape(train_x1,(-1,1))

train_x2=tf.reshape(train_x2,(-1,1))

test_x1=tf.reshape(test_x1,(-1,1))

test_x2=tf.reshape(test_x2,(-1,1))





train_x=tf.cast(tf.concat([train_x1,train_x2],axis=1),tf.float32)

test_x=tf.cast(tf.concat([test_x1,test_x2],axis=1),tf.float32)



x0_train=np.ones(num_train).reshape(-1,1)

x0_test=np.ones(num_test).reshape(-1,1)

train_X=tf.cast(tf.concat([x0_train,train_x],axis=1),tf.float32)

test_X=tf.cast(tf.concat([x0_test,test_x],axis=1),tf.float32)

train_Y=tf.constant(train_y.reshape(-1,1),tf.float32)

test_Y=tf.constant(test_y.reshape(-1,1),tf.float32)

# print(test_X)



# print(train_Y.shape)

# print(train_x)

# print(test_x)

learn_rate=0.1

iter=230

display_step=5

np.random.seed(612)

w=tf.Variable(np.random.randn(3,1),dtype=tf.float32)

mse_train=[]

mse_test=[]

for i in range(0,iter+1):

with tf.GradientTape() as tape:

pred_train=tf.matmul(train_X,w)

loss_train=0.5*tf.reduce_mean(tf.square(train_Y-pred_train))

pred_test=tf.matmul(test_X,w)

loss_test=0.5*tf.reduce_mean(tf.square(test_Y-pred_test))

mse_train.append(loss_train)

mse_test.append(loss_test)

dl_dw=tape.gradient(loss_train,w)

w.assign_sub(learn_rate*dl_dw)

if i%display_step==0:

print("i:%i,Train Loss:%f,Test Loss:%f"%(i,loss_train,loss_test))

plt.figure(figsize=(15,10))

plt.subplot(221)

plt.plot(mse_train,color="b",linewidth=3,label="train loss")

plt.plot(mse_test,color="r",linewidth=3,label="test loss")

plt.legend(loc="upper right")

plt.subplot(222)

plt.plot(train_y,color="b",marker="o",label="true_price")

plt.plot(pred_train,color="r",marker=".",label="predice_price")

plt.legend()

plt.subplot(223)

plt.plot(test_y,color="b",marker="o",label="true_price")

plt.plot(pred_test,color="r",marker=".",label="predice_price")

plt.legend()

plt.show()
