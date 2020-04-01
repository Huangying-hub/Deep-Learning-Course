# Deep-Learning-Course
import tensorflow as tf 
import numpy as np 
x=tf.constant([64.3,99.6,145.45,63.75,135.46,92.85,86.97,144.76,59.3,116.03],dtype=tf.float32) 
y=tf.constant([62.55,82.42,132.62,73.31,131.05,86.57,85.49,127.44,55.25,104.84],dtype=tf.float32) 
n=tf.constant(tf.size(x)) 
n=tf.cast(n,dtype=tf.float32)
 #求x中的每个元素乘以y中的每个元素  
x1=x*y
 #把x乘以y的每个元素求和
x1=tf.reduce_sum(x1) 
 #求n乘以x中每个元素乘以y中每个元素的乘积的和
x1n=n*x1 
 #求x中各个元素的和 
xsum=tf.reduce_sum(x)
 #求y中各个元素的和 
ysum=tf.reduce_sum(y)
 #求xsum与ysum的乘积 
xsum_ysum=xsum*ysum
 #求分子 
fenm=x1n-xsum_ysum
 #求x中所有元素的平方
x2=xsum**2 
 #求n乘以x中所有元素的和的平方
x2n=n*tf.reduce_sum((x**2))
 #求分母 
fenz=x2n-x2 
w=fenm/fenz 
b=(ysum-(w*xsum))/n 
print("w的值为：{}".format(w)) 
print("b的值为：{}".format(b)) 
