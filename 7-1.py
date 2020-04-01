# Deep-Learning-Course
import matplotlib.pyplot as plt 
from PIL import Image 
plt.rcParams["font.sans-serif"]="SimHei" 
img=Image.open("E:\python\picture\lena.tiff") 
rgb_r,rgb_g,rgb_b=img.split() 
plt.figure(figsize=(4,4)) 
plt.suptitle("图像基本操作",fontsize=20,color='b') 
plt.subplot(221) 
plt.title("R-缩放",fontsize=14) 
plt.axis("off") 
imgg=rgb_r.resize((50,50)) 
plt.imshow(imgg,cmap="gray") 
plt.subplot(222) 
plt.title("G-镜像+旋转",fontsize=14) 
imgg=rgb_g.transpose(Image.FLIP_LEFT_RIGHT) 
imgg=imgg.transpose(Image.ROTATE_270) 
plt.imshow(imgg,cmap="gray") 
 #对分离出来的G通道图片进行水平翻转，再进行顺时针旋转90度（逆时针旋转270度） 
plt.subplot(223) 
plt.title("B-剪裁",fontsize=14) 
plt.axis("off") 
imgg=rgb_b.crop((0,0,300,300)) 
plt.imshow(imgg,cmap="gray") 
 #对B通道图片进行剪裁坐标为(0,0),(300,300) 
plt.subplot(224) 
plt.title("RGB",fontsize=14) 
plt.axis("off") 
imgg=Image.merge("RGB",[rgb_r,rgb_g,rgb_b]) 
plt.imshow(imgg) 
imgg.save(r"E:\python\picture\test.png") 
 #合并R,G,B三通道图片为RGB24位真彩色图片，并保存合并后的图片为test.png 
plt.tight_layout(rect=[0,0,1,0.9]) 
plt.show() 
