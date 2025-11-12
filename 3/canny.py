import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from sobellvbo import sobel

path = r"C:\Users\xxt\Desktop\ddl\HNO3's homework\3\xiaohei.jpeg"
img = cv2.imread(r"C:\Users\xxt\Desktop\ddl\HNO3's homework\3\xiaohei.jpeg")
pix = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_1 = cv2.GaussianBlur(pix,(5,5),1)

print("hewe")
out,angle = sobel(img_1)
print("hewe")

h, w = out.shape
nms = np.zeros_like(out)
angle_deg = np.degrees(angle) % 180

for i in range(1, h-1): # NMS
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

print("hewe")

high_threshold = np.max(nms) * 0.1
low_threshold = high_threshold * 0.5
strong_edges = nms > high_threshold # 筛选出强边和弱边
weak_edges = (nms >= low_threshold) & (nms <= high_threshold)

final_edges = strong_edges.copy()
changed = True
while changed:
    changed = False
    for i in range(1, h-1):
        for j in range(1, w-1):
            if weak_edges[i,j] and np.any(final_edges[i-1:i+2, j-1:j+2]) and final_edges[i,j] != True: # 附近有强边的弱边都True
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
plt.savefig("task2-canny.jpg")
plt.show()
