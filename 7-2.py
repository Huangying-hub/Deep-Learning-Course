# Deep-Learning-Course
import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
from PIL import Image  
plt.rcParams["font.sans-serif"]="SimHei" 
mnist=tf.keras.datasets.mnist 
(trian_x,train_y),(text_x,text_y)=mnist.load_data(r"E:\python\picture\mnist.npz") 
 #把数据放到numpy数组中  
plt.figure(figsize=(6,6)) 
plt.suptitle("MNIST测试集样本",fontsize=20,color='r') 
 #建立画布（6X6），并设置全局标题，字体20，颜色为红色 
  #取随机值，随机显示MNIST数据集中的样本 
for i in range(16): 
    n=np.random.randint(0,60000)
    plt.subplot(4,4,i+1)
    plt.axis("off")
    plt.title("标签值:"+str(train_y[n]),fontsize=14)
    plt.imshow(trian_x[n],cmap="gray")
  
    
plt.tight_layout(rect=[0,0,1,0.95]) 
plt.show() 
