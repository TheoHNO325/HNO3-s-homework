import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

path = r"C:\Users\xxt\Desktop\ddl\HNO3's homework\3\xiaohei.jpeg"
img = Image.open(path).convert('L')
pix = np.array(img)
img_1 = cv2.GaussianBlur(pix,(5,5),1)
h,w = pix.shape

# or 用在2里的 （太慢了）
# def gaussian_filter(size, sigma):
#         x_size = int((size[0] - 1) / 2)
#         y_size = int((size[1] - 1) / 2)
#         x = np.array(range(-x_size, x_size + 1))
#         y = np.array(range(-y_size, y_size + 1))
#         [xx, yy] = np.meshgrid(x, y)
#         kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
#         return kernel / np.sum(kernel)

# kernel = gaussian_filter((5,5),1)

# padded = np.pad(pix, ((2,2),(2,2)), mode='constant')
# img_1 = np.zeros_like(padded, dtype=float)

# for i in range(2, h+2):
#     for j in range(2, w+2):
#         img_1[i,j] = np.sum(kernel * padded[i-2:i+3, j-2:j+3])

# img_1 = img_1[2:h+2, 2:w+2]

plt.imshow(img_1)
plt.title("afterblur")

# 把sobellvbo里的封装成函数

def sobel(pix):
    h,w = pix.shape
    out = np.zeros((h, w))
    angle = np.zeros((h, w))
    gx = np.array([[-1, 0, 1], 
            [-2, 0, 2], 
            [-1, 0, 1]])
    gy = np.array([[-1, -2, -1],
            [0, 0, 0], 
            [1, 2, 1]])

    for i in range(1, h-1):
        for j in range(1, w-1):
            dx = np.sum(gx * img_1[i-1:i+2, j-1:j+2])
            dy = np.sum(gy * img_1[i-1:i+2, j-1:j+2])
            out[i,j] = np.sqrt(dx**2 + dy**2)
            angle[i,j] = np.arctan2(dy, dx)
    return out,angle

out,angle = sobel(img_1)

nms = np.zeros_like(out)
angle_deg = np.degrees(angle) % 180

for i in range(1, h-1):
    for j in range(1, w-1):
        if (0 <= angle_deg[i,j] < 22.5) or (157.5 <= angle_deg[i,j] <= 180):
            neighbors = [out[i,j-1], out[i,j+1]]
        elif 22.5 <= angle_deg[i,j] < 67.5:
            neighbors = [out[i-1,j+1], out[i+1,j-1]]
        elif 67.5 <= angle_deg[i,j] < 112.5:
            neighbors = [out[i-1,j], out[i+1,j]]
        else:
            neighbors = [out[i-1,j-1], out[i+1,j+1]]
        
        if out[i,j] >= max(neighbors):
            nms[i,j] = out[i,j]

high_threshold = np.max(nms) * 0.1
low_threshold = high_threshold * 0.5
strong_edges = nms > high_threshold
weak_edges = (nms >= low_threshold) & (nms <= high_threshold)

final_edges = strong_edges.copy()
changed = True
while changed:
    changed = False
    for i in range(1, h-1):
        for j in range(1, w-1):
            if weak_edges[i,j] and np.any(final_edges[i-1:i+2, j-1:j+2]):
                final_edges[i,j] = True
                changed = True

fig, axes = plt.subplots(2, 3, figsize=(15,10))
axes[0,0].imshow(pix, cmap='gray')
axes[0,0].set_title('Original')
axes[0,1].imshow(img_1, cmap='gray')
axes[0,1].set_title('Gaussian Filtered')
axes[0,2].imshow(out, cmap='gray')
axes[0,2].set_title('Gradient Magnitude')
axes[1,0].imshow(angle, cmap='hsv')
axes[1,0].set_title('Gradient Direction')
axes[1,1].imshow(nms, cmap='gray')
axes[1,1].set_title('After NMS')
axes[1,2].imshow(final_edges, cmap='gray')
axes[1,2].set_title('Final Edges')
plt.tight_layout()
plt.show()