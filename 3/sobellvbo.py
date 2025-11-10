import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

path = r"C:\Users\xxt\Desktop\ddl\HNO3's homework\3\xiaohei.jpeg"
img = Image.open(path).convert('L')
pix = np.array(img)
h, w = pix.shape

out = np.zeros((h, w))

gx = np.array([[-1, 0, 1], 
        [-2, 0, 2], 
        [-1, 0, 1]])
gy = np.array([[-1, -2, -1],
        [0, 0, 0], 
        [1, 2, 1]])

for i in range(1, h-1):
    for j in range(1, w-1):
        dx = 0
        dy = 0
        for m in range(3):
            for n in range(3):
                dx += pix[i+m-1, j+n-1] * gx[m][n]
                dy += pix[i+m-1, j+n-1] * gy[m][n]
        out[i, j] = np.sqrt(dx*dx + dy*dy)
plt.imshow(out, cmap='gray')
plt.axis('off')
plt.show()
