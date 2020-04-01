# Deep-Learning-Course

import tensorflow as tf
import numpy as np
aera=np.array([137.97,104.50,100.00,124.32,79.20,99.00,124.00,114.00,106.69,138.05,53.75,46.91,68.00,63.02,81.26,86.21])
roomnum=np.array([3.0,2.0,2.0,3.0,1.0,2.0,3.0,2.0,2.0,3.0,1.0,1.0,1.0,1.0,2.0,2.0])
price=np.array([145.00,110.00,93.00,116.00,65.32,104.00,118.00,91.00,62.00,133.00,51.00,45.00,78.50,69.65,75.69,95.30])
taera=tf.constant(aera)
troomnum=tf.constant(roomnum)
tprice=tf.constant(price)
tone=tf.ones((16,),dtype=tf.float64)
#创建TensorFlow张量
X=tf.stack((tone,taera,troomnum),axis=1)
Y=tf.reshape(tprice,(16,-1))

Xt=tf.transpose(X)
Xtx=tf.matmul(Xt,X)
Xtx_1=tf.linalg.inv(Xtx)
X_Xt=tf.matmul(Xtx_1,Xt)
w=tf.matmul(X_Xt,Y)

print("房价预测系统:")
while(1):
    neara=float(input("请输入房屋面积（20-500之间的实数）："))
    nroomnum=float(input("请输入房间数（1-10之间的整数）:"))
    if neara>=20 and neara<=500 and nroomnum>=1 and nroomnum<=10:
        break
    else:
        print("输入房屋面积或房间数范围错误！！！")
yprice=w[1]*neara+w[2]*nroomnum+w[0]

print("房屋面积：{}\n房间数：{}\n预测的房价为：{:.2f}万元".format(neara,nroomnum,yprice[0]))

