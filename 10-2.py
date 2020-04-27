# Deep-Learning-Course
import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf



boston_housing=tf.keras.datasets.boston_housing

(train_x,train_y),(test_x,test_y)=boston_housing.load_data()

# print(train_x)

# train_x=train_x[:,12]

# test_x=test_x[:,12]

num_train=len(train_x)

num_test=len(test_x)

train_x=(train_x-train_x.min())/(train_x.max()-train_x.min())

test_x=(test_x-test_x.min())/(test_x.max()-test_x.min())



x0_train=np.ones(num_train).reshape(-1,1)

x0_test=np.ones(num_test).reshape(-1,1)

train_X=tf.cast(tf.concat([x0_train,train_x],axis=1),tf.float32)

test_X=tf.cast(tf.concat([x0_test,test_x],axis=1),tf.float32)

train_Y=tf.constant(train_y.reshape(-1,1),tf.float32)

test_Y=tf.constant(test_y.reshape(-1,1),tf.float32)





# print(train_Y.shape)

# print(train_x)

# print(test_x)

learn_rate=1

iter=20000

display_step=1000

np.random.seed(612)

w=tf.Variable(np.random.randn(14,1),dtype=tf.float32)

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
