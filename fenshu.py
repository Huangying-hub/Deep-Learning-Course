# Deep-Learning-Course


import math 
a=float(input('a: '))
b=float(input('b: ')) 
c=float(input('c: ')) 
d=(b**2)-(4ac)
if d >0: print("此函数有两个解") 
   sol1=(-b-math.sqrt(d))/(2a) 
   sol2=(-b+math.sqrt(d))/(2a) 
   print("x1=",sol1,"\t","x2=",sol2) 
elif d ==0: 
    print("此函数有一个解") 
    print("x=",(-b-math.sqrt(d))/(2*a)) 
elif d <0: 
    print("此函数无解")
