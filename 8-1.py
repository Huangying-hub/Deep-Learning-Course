# Deep-Learning-Course
import tensorflow as tf 
import numpy as np 
x=tf.constant([64.3,99.6,145.45,63.75,135.46,92.85,86.97,144.76,59.3,116.03]) 
y=tf.constant([62.55,82.42,132.62,73.31,131.05,86.57,85.49,127.44,55.25,104.84]) 
x_=tf.reduce_mean(x)
#求x的平均值 
y_=tf.reduce_mean(y)
#求y的平均值 
x_x_=x-x_
#求x中每个元素减去x的平均值 
x_x_2=x_x_**2
#求x中每个元素减去x的平均值的平方 
y_y_=y-y_
#求y中每个元素减去y的平均值 
x__y=x_x_*y_y_
#求x的中每个元素减去x的均值乘以y中每个元素减去y的均值 
fenm=tf.reduce_sum(x__y) 
fenz=tf.reduce_sum(x_x_2)  
w=fenm/fenz 
b=y_-w*x_  
print("w的值为：{}".format(w)) 
print("b的值为：{}".format(b)) 

