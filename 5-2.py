# Deep-Learning-Course

import numpy as np
x=np.array([64.3,99.6,145.45,63.75,135.46,92.85,86.97,144.76,59.3,116.03])
y=np.array([62.55, 84.42,132.62,73.31,131.05,86.57,85.49,127.44,55.25,104.84])
a=x.mean()
y1=y.mean()
b=x-a
c=b**2
y2=y-y1
f=b*y2
fs=np.sum(f)
m=np.sum(c)
w=fs/m
b=y1-w*a
print("w= {}".format(w))
print("b= {}".format(b))
