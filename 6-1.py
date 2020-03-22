# Deep-Learning-Course
import matplotlib.pyplot as sp
import numpy as np

sp.rcParams['font.sans-serif']='SimHei'
sp.rcParams['axes.unicode_minus']=False
x=np.array([137.97,104.50,100.00,124.32,79.20,99.00,124.00,114.00,106.69,138.05,53.75,46.91,68.00,63.02,81.26,86.21])
y=np.array([145.00,110.00,93.00,116.00,65.32,104.00,118.00,91.00,62.00,133.00,51.00,45.00,78.50,69.65,75.69,95.30])
sp.scatter(x,y,color="red")
sp.title("商品房销售记录",color="blue",fontsize=16)
sp.xlim(20,150)
sp.ylim(20,150)
sp.xlabel('面积（平方米）',fontsize=14)
sp.ylabel('价格（万元）',fontsize=14)
sp.show()
