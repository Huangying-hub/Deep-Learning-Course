# Deep-Learning-Course
import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf

# from sklearn.utils import shuffle

# from sklearn.preprocessing import scale

#导入数据

boston_housing=tf.keras.datasets.boston_housing

(train_x,train_y),(test_x,test_y)=boston_housing.load_data()

train_x=train_x[:,:12]

test_x=test_x[:,:12]

# print(train_x.shape)

# print(train_y)

#数据归一化

train_x1=(train_x-train_x.min(axis=0))/(train_x.max(axis=0)-train_x.min(axis=0))

test_x1=(test_x-test_x.min(axis=0))/(test_x.max(axis=0)-test_x.min(axis=0))

# print(train_x1)

#转化类型为float32

x_train=tf.cast(train_x1,dtype=tf.float32)

x_valid=tf.cast(test_x1,dtype=tf.float32)

#定义模型

def model(x,w,b):

return tf.matmul(x,w)+b

#定义w,v两个参数

w=tf.Variable(tf.random.normal([12,1],mean=0.0,stddev=1.0,dtype=tf.float32))

b=tf.Variable(tf.zeros(1),dtype=tf.float32)



# print(w)

# print(b)

#设置超参数

training_epochs=150#迭代次数

learning_rate=0.02#学习率

batch_size=10#批量训练一次的样本数

#定义均方误差损失函数

def loss(x,y,w,b):

err=model(x,w,b)-y

squared_err=tf.square(err)

return tf.reduce_mean(squared_err)

#定义梯度计算函数

def grad(x,y,w,b):

with tf.GradientTape() as tape:

loss_=loss(x,y,w,b)

return tape.gradient(loss_,[w,b])

#选择优化器,优化器可以帮助根据算出的求导结果更新模型参数，从而最小化损失函数

optimizer=tf.keras.optimizers.SGD(learning_rate)#创建优化器，指定学习率

#训练过程

loss_list_train=[]#保存训练集loss值的列表

loss_list_valid=[]#保存测试集loss值的列表

for epoch in range (training_epochs):

for step in range(40):

xs=x_train[step*batch_size:(step+1)*batch_size,:]

ys=train_y[step*batch_size:(step+1)*batch_size]

grads=grad(xs,ys,w,b)

optimizer.apply_gradients(zip(grads,[w,b]))

loss_train=loss(x_train,train_y,w,b).numpy()

loss_valid=loss(x_valid,test_y,w,b).numpy()

loss_list_train.append(loss_train)

loss_list_valid.append(loss_valid)


