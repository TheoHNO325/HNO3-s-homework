import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(r"C:\Users\xxt\Desktop\ddl\HNO3's homework\3\xiaohei.jpeg", 0) # 转为单通道灰度图
h, w = img.shape

hist = np.zeros(256) # 一共256个灰度值
for i in range(h):
    for j in range(w):
        hist[img[i,j]] += 1 # 该灰度值的数量+1

cdf = np.zeros(256)
cdf[0] = hist[0]
for i in range(1, 256):
    cdf[i] = cdf[i-1] + hist[i] # 构建离散概率分布函数

cdf_min = cdf[np.nonzero(cdf)[0][0]] # 最小的像素值
cdf_norm = (cdf - cdf_min) * 255 / (h*w - cdf_min) # 映射到0-255 cdf最大值为h*w，最小为cdf_min

equalized = np.zeros_like(img)
for i in range(h):
    for j in range(w):
        equalized[i,j] = cdf_norm[img[i,j]]

# ai作图
fig, axes = plt.subplots(2, 2, figsize=(12,10))

axes[0,0].imshow(img, cmap='gray')
axes[0,0].set_title('Original Image')

axes[0,1].hist(img.flatten(), 256, [0,256])
axes[0,1].set_title('Original Histogram')

axes[1,0].imshow(equalized, cmap='gray')
axes[1,0].set_title('Equalized Image')

axes[1,1].hist(equalized.flatten(), 256, [0,256])
axes[1,1].set_title('Equalized Histogram')

plt.tight_layout()
plt.show()