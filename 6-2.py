# Deep-Learning-Course

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

boston_housing=tf.keras.datasets.boston_housing
(train_x,train_y),(test_x,test_y)=boston_housing.load_data(test_split=0)

titles=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B-1000','LSTAT','MEDV']
plt.figure(figsize=(12,12))
zhu=int(input("请输入功能键（1表示显示全部属性图表，2表示选择某一属性的图表）："))
if zhu==1:
   for i in range(13):
       plt.subplot(4,4,(i+1))
       plt.scatter(train_x[:,i],train_y,8)
       plt.xlabel(titles[i])
       plt.ylabel("Price($1000's)")
       plt.title(str(i+1)+"."+titles[i]+" - Price")
else:
   inputnumber=int(input("\n1--CRIM\n2--ZN\n3--INDUS\n4--CHAS\n5--NOX\n6--RM\n7--AGE\n8--DIS\n9--RAD\n10-TAX\n11-PTRATIO\n12-B\n13-LSTAT\n请输入属性："))
   plt.scatter(train_x[:,inputnumber-1],train_y,8)
   plt.xlabel(titles[inputnumber-1])
   plt.ylabel("Price($1000's)")
   plt.title(str(inputnumber)+"."+titles[inputnumber-1]+" - Price")
plt.tight_layout(pad=7, w_pad=2.0, h_pad=4.0)
plt.suptitle("各个属性与房屋的关系",fontsize=20)
plt.show()
