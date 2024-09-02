#导入所需的库
import cv2

#读取输入图像
image = cv2.imread('demo.jpg')

#定义alpha和beta
alpha = 3 #对比度控制
beta  = 0  #亮度控制

#调用convertScaleAbs函数
adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

#显示输出图像
cv2.namedWindow('adjusted',cv2.WINDOW_NORMAL)
cv2.imshow('adjusted', adjusted)
cv2.waitKey()
cv2.destroyAllWindows()
