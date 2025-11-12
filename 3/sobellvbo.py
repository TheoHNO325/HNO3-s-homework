import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def sobel(img_1):
    h,w = img_1.shape
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

path = r"C:\Users\xxt\Desktop\ddl\HNO3's homework\3\xiaohei.jpeg"
img = Image.open(path).convert('L')
pix = np.array(img)
h, w = pix.shape
out,_ = sobel(pix) #被封装进utils了
plt.imshow(out, cmap='gray')
plt.axis('off')
plt.savefig('task1-sobel.jpg')
plt.show()
