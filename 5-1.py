import numpy as num
a=int(input("(1-100)a="))
x=0
y=1
num.random.seed(612)
while x<1000:
    s=num.random.uniform(0,1)
    if x%a==0:
        print("%d\t%d\t%f\t"%(y,x,s))
        y=y+1
    x=x+1
