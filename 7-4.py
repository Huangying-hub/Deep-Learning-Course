# Deep-Learning-Course
import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt



# 导入数据

mnist=tf.keras.datasets.mnist

(train_images,train_labels),(test_images,test_labels)=mnist.load_data()

# print(train_labels[1])

# plt.imshow(train_images[1],cmap="binary")

# plt.show()

# 划分验证集

total_num=len(train_images)

valid_split=0.2

train_num=int(total_num*(1-valid_split))

train_x=train_images[:train_num]

train_y=train_labels[:train_num]

valid_x=train_images[train_num:]

valid_y=train_labels[train_num:]

test_x=test_images

test_y=test_labels

# print(valid_x.shape)

# 把28*28的图片变成1*784的图片数据

train_x=train_x.reshape(-1,784)

valid_x=valid_x.reshape(-1,784)

test_x=test_x.reshape(-1,784)

# 特征数据归一化

train_x=tf.cast(train_x/255.0,tf.float32)

valid_x=tf.cast(valid_x/255.0,tf.float32)

test_x=tf.cast(test_x/255.0,tf.float32)

# 数据标签独热编码

train_y=tf.one_hot(train_y,depth=10)

valid_y=tf.one_hot(valid_y,depth=10)

test_y=tf.one_hot(test_y,depth=10)

# print(test_y)

# 构建模型y=wx+b

def model(x,w,b):

pred=tf.matmul(x,w)+b

return tf.nn.softmax(pred)# S型函数，生成概率



#定义变量w,b

w=tf.Variable(tf.random.normal([784,10],mean=0.0,stddev=1.0,dtype=tf.float32))

b=tf.Variable(tf.zeros([10],dtype=tf.float32))

# 定义交叉熵损失函数

def loss(x,y,w,b):

pred=model(x,w,b)#预测

loss_=tf.keras.losses.categorical_crossentropy(y_true=y,y_pred=pred)

return tf.reduce_mean(loss_)#平均值，得出均方差

#定义训练超参数

trainning_epochs=20#训练轮数

batch_size=50#单次训练样本数（批次大小）

learning_rate=0.001#学习率

#定义梯度计算函数

def grad(x,y,w,b):

with tf.GradientTape() as tape:

loss_=loss(x,y,w,b)

return tape.gradient(loss_,[w,b])#求偏导

optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)

#定义准确率

def accurace(x,y,w,b):

pred=model(x,w,b)#预测值

correct_prediction=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))

return tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

# 训练模型

total_step=int(train_num/batch_size)#一轮训练有多少批次

loss_list_train=[]

loss_list_valid=[]

acc_list_train=[]

acc_list_valid=[]

for epoch in range(trainning_epochs):

for step in range(total_step):

xs=train_x[step*batch_size:(step+1)*batch_size]

ys=train_y[step*batch_size:(step+1)*batch_size]

grads=grad(xs,ys,w,b)

optimizer.apply_gradients(zip(grads,[w,b]))

loss_train=loss(train_x,train_y,w,b).numpy()

loss_valid=loss(valid_x,valid_y,w,b).numpy()

acc_train=accurace(train_x,train_y,w,b).numpy()

acc_valid=accurace(valid_x,valid_y,w,b).numpy()

loss_list_train.append(loss_train)

loss_list_valid.append(loss_valid)

acc_list_train.append(acc_train)

acc_list_valid.append(acc_valid)

print("epoch={:3d},train_loss={:.4f},train_acc={:.4f},val_loss={:.4f},val_acc={:.4f}".format(epoch+1,loss_train,acc_train,loss_valid,acc_valid))

# acc_test=accurace(test_x,test_y,w,b).numpy()

# print("Test accuracy",acc_test)

# def predict(x,w,b):

# pred=model(x,w,b)#计算模型预测值

# result=tf.argmax(pred,1).numpy()

# return result

# pred_test=predict(test_x,w,b)

# print(pred_test[0])

#可视化

# def plot_images_labels_prediction(images,labels,preds,index=0,num=10):

# fig=plt.gcf()

# fig.set_size_inches(10,4)

# if num>10:

# num=10

# for i in range(0,num):

# ax=plt.subplot(2,5,i+1)

# ax.imshow(np.reshape(images[index],(28,28)),cmap="binary")

# title="label="+str(labels[index])

# if len(preds)>0:

# title+=",predict="+str(labels[index])

# ax.set_title(title,fontsize=10)

# ax.set_yticks([])

# ax.set_xticks([])

# index=index+1

# plt.show()

# plot_images_labels_prediction(test_images,test_labels,pred_test,10,10)
