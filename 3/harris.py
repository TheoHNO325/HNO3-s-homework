import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(r"C:\Users\xxt\Desktop\ddl\HNO3's homework\3\xiaohei.jpeg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

# 求梯度
blurred = cv2.GaussianBlur(gray, (5,5), 0.1)
Ix = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
Iy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
Ix2 = Ix * Ix
Iy2 = Iy * Iy
Ixy = Ix * Iy

sigma = 2 # 滤波
Ix2_blur = cv2.GaussianBlur(Ix2, (5,5), sigma)
Iy2_blur = cv2.GaussianBlur(Iy2, (5,5), sigma)
Ixy_blur = cv2.GaussianBlur(Ixy, (5,5), sigma)

alpha = 0.04 # 行列式算R
R = np.zeros_like(gray)
for i in range(gray.shape[0]):
    for j in range(gray.shape[1]):
        M = np.array([[Ix2_blur[i,j], Ixy_blur[i,j]],
                      [Ixy_blur[i,j], Iy2_blur[i,j]]])
        det = np.linalg.det(M)
        trace = np.trace(M)
        R[i,j] = det - alpha * (trace ** 2)
# 画
fig, axes = plt.subplots(3, 4, figsize=(20,10))

axes[0,0].imshow(gray, cmap='gray')
axes[0,0].set_title('Original Image')

axes[0,1].imshow(Ix, cmap='gray')
axes[0,1].set_title('Ix Gradient')

axes[0,2].imshow(Iy, cmap='gray')
axes[0,2].set_title('Iy Gradient')

axes[0,3].imshow(R, cmap='hot')
axes[0,3].set_title('Harris Response R')

# opencv库
window_sizes = [3, 5, 7, 9]
for idx, size in enumerate(window_sizes):
    dst_temp = cv2.cornerHarris(gray, size, 3, 0.04)
    dst_temp = cv2.dilate(dst_temp, None)
    temp_corners = dst_temp > 0.01 * dst_temp.max()
    
    temp_img = img.copy()
    temp_img[temp_corners] = [0,0,255]
    axes[1,idx].imshow(cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB))
    axes[1,idx].set_title(f'Window Size: {size}')

# 手动实现（AI实现）
T = 0.01 * R.max()
candidate_corners = R > T #候选

# NMS 好像写错了
corners = np.zeros_like(R)
for idx, size in enumerate(window_sizes):
    offset = size // 2
    for i in range(offset, gray.shape[0]-offset):
        for j in range(offset, gray.shape[1]-offset):
            if candidate_corners[i,j]:
                local_window = R[i-offset:i+offset+1, j-offset:j+offset+1]
                if R[i,j] == np.max(local_window):
                    corners[i,j] = R[i,j]

    temp_img = img.copy()
    temp_img[corners > 0] = [0,0,255]
    axes[2,idx].imshow(cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB))
    axes[2,idx].set_title(f'Window Size: {size}')

plt.tight_layout()
plt.show()
