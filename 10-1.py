# Deep-Learning-Course
import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf



boston_housing=tf.keras.datasets.boston_housing

(train_x,train_y),(test_x,test_y)=boston_housing.load_data()

# print(train_x)

train_x=train_x[:,12]

test_x=test_x[:,12]

train_x=(train_x-train_x.min())/(train_x.max()-train_x.min())

test_x=(test_x-test_x.min())/(test_x.max()-test_x.min())



# print(train_x)

# print(test_x)

learn_rate=0.04

iter=1800

display_step=200

np.random.seed(612)

w=tf.Variable(np.random.randn())

b=tf.Variable(np.random.randn())

mse_train=[]

mse_test=[]

for i in range(0,iter+1):

with tf.GradientTape() as tape:

pred_train=w*train_x+b

loss_train=0.5*tf.reduce_mean(tf.square(train_y-pred_train))

pred_test=w*test_x+b

loss_test=0.5*tf.reduce_mean(tf.square(test_y-pred_test))

mse_train.append(loss_train)

mse_test.append(loss_test)

dl_dw,dl_db=tape.gradient(loss_train,[w,b])

w.assign_sub(learn_rate*dl_dw)

b.assign_sub(learn_rate*dl_db)

if i%display_step==0:

print("i:%i,Train Loss:%f,Test Loss:%f"%(i,loss_train,loss_test))

plt.figure(figsize=(15,10))

plt.subplot(221)

plt.scatter(train_x,train_y,color="b",label="data")

plt.plot(train_x,pred_train,color="r",label="model")

plt.legend(loc="upper left")

plt.subplot(222)

plt.plot(mse_train,color="b",linewidth=3,label="train loss")

plt.plot(mse_test,color="r",linewidth=3,label="test loss")

plt.legend(loc="upper right")

plt.subplot(223)

plt.plot(train_y,color="b",marker="o",label="true_price")

plt.plot(pred_train,color="r",marker=".",label="predice_price")

plt.legend()

plt.subplot(224)

plt.plot(test_y,color="b",marker="o",label="true_price")

plt.plot(pred_test,color="r",marker=".",label="predice_price")

plt.legend()

plt.show()
