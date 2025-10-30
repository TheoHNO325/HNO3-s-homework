import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import signal
SIGMA = 10 # 直接调，三组
SIZE = 21 # 直接调
WIDTH = 200 # padding 宽度
plt.rcParams['font.family'] = 'SimHei' # 设置中文字体为黑体
plt.rcParams['font.size'] = 12 # 设置字体大小
plt.rcParams['font.weight'] = 'bold' # 设置字体粗细

def gaosi(path=None):
    def gaussian_filter(size, sigma):
        x_size = int((size[0] - 1) / 2)
        y_size = int((size[1] - 1) / 2)
        x = np.array(range(-x_size, x_size + 1))
        y = np.array(range(-y_size, y_size + 1))
        [xx, yy] = np.meshgrid(x, y)
        kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        return kernel / np.sum(kernel)
    if path:
        img = cv2.imread(path)
    else:
        img = cv2.imread(r"C:\Users\xxt\Desktop\ddl\HNO3's homework\2\xiaohei.jpeg")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    kernel = gaussian_filter([SIZE, SIZE], SIGMA)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(kernel, cmap='gray')
    plt.title('My高斯滤波')

    img_res = np.zeros_like(img_rgb, dtype=np.float64)
    for i in range(3): 
        img_res[:, :, i] = signal.convolve2d(img_rgb[:, :, i], kernel, mode='same', boundary='symm')

    img_res = np.clip(img_res, 0, 255).astype(np.uint8)

    plt.subplot(1, 3, 2)
    plt.imshow(img_rgb)
    plt.title('Xiaohei')

    plt.subplot(1, 3, 3)
    plt.imshow(img_res)
    plt.title('Xiaohei?')

    plt.tight_layout()
    plt.show()

    img_blur = cv2.GaussianBlur(img_rgb, (SIZE, SIZE), SIGMA)

    kukernel_1 = cv2.getGaussianKernel(SIZE, SIGMA)
    kukernel_2 = np.outer(kukernel_1 , kukernel_1 .T)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(kukernel_2, cmap='gray')
    plt.title('库高斯滤波')

    plt.subplot(1, 3, 2)
    plt.imshow(img_rgb)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(img_blur)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# padding 

def padbynumpy(path=None):
    if path:
        img = cv2.imread(path)
    else:
        img = cv2.imread(r"C:\Users\xxt\Desktop\ddl\HNO3's homework\2\xiaohei.jpeg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # 发现出现变色，故添加

    def padding(img, pad_width, mode):

        if isinstance(pad_width, int):
            top = bottom = left = right = pad_width
        elif len(pad_width) == 2:
            top = bottom = pad_width[0]
            left = right = pad_width[1]
        elif len(pad_width) == 4:
            top, bottom, left, right = pad_width
        
        if img.ndim == 2:
            h, w = img.shape
            padded = np.pad(img, ((top, bottom), (left, right)), mode=mode)
        else:
            h, w, c = img.shape
            padded = np.pad(img, ((top, bottom), (left, right), (0, 0)), mode=mode)
    
        return padded
    
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title('Xiaohei')
    
    pad1 = padding(img, WIDTH, 'edge')
    pad2 = padding(img, WIDTH, 'symmetric')

    plt.subplot(1, 3, 2)
    plt.imshow(pad1)
    plt.title('Xiaohei-fat')

    plt.subplot(1, 3, 3)
    plt.imshow(pad2)
    plt.title('Xiaohei-reflect')

    plt.tight_layout()
    plt.show()


def padbymyself(path=None):
    if path:
        img = cv2.imread(path)
    else:
        img = cv2.imread(r"C:\Users\xxt\Desktop\ddl\HNO3's homework\2\xiaohei.jpeg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # 发现出现变色，故添加

    def padding(img, pad_width, mode):

        if isinstance(pad_width, int):
            top = bottom = left = right = pad_width
        elif len(pad_width) == 2:
            top = bottom = pad_width[0]
            left = right = pad_width[1]
        elif len(pad_width) == 4:
            top, bottom, left, right = pad_width
        
        if mode == 'edge':
            if img.ndim == 2:
                h, w = img.shape
                padded = np.zeros((h + top + bottom, w + left + right), dtype=img.dtype)
                
                padded[top:top+h, left:left+w] = img
                
                for i in range(top):
                    padded[i, left:left+w] = img[0, :]
                for i in range(bottom):
                    padded[top+h+i, left:left+w] = img[-1, :]
                for j in range(left):
                    padded[:, j] = padded[:, left]
                
                for j in range(right):
                    padded[:, left+w+j] = padded[:, left+w-1]
            elif img.ndim == 3:
                h, w, c = img.shape
                padded = np.zeros((h + top + bottom, w + left + right, c), dtype=img.dtype)
                
                padded[top:top+h, left:left+w, :] = img
                
                for i in range(top):
                    padded[i, left:left+w, :] = img[0, :, :]
                for i in range(bottom):
                    padded[top+h+i, left:left+w, :] = img[-1, :, :]
                for j in range(left):
                    padded[:, j, :] = padded[:, left, :]
                for j in range(right):
                    padded[:, left+w+j, :] = padded[:, left+w-1, :]
        elif mode == 'symmetric':
            if img.ndim == 2:
                h, w = img.shape
                padded = np.zeros((h + top + bottom, w + left + right), dtype=img.dtype)
                
                padded[top:top+h, left:left+w] = img
                
                for i in range(top):
                    padded[i, left:left+w] = img[top - i, :] if top - i < h else img[0, :]
                
                for i in range(bottom):
                    src_row = h - 2 - (i % h) if i >= h else h - 2 - i
                    padded[top+h+i, left:left+w] = img[src_row, :]
                
                for j in range(left):
                    padded[:, j] = padded[:, 2*left - j] if 2*left - j < left + w else padded[:, left]
                
                for j in range(right):
                    src_col = left + w - 2 - (j % w) if j >= w else left + w - 2 - j
                    padded[:, left+w+j] = padded[:, src_col]
            
            elif img.ndim == 3:
                h, w, c = img.shape
                padded = np.zeros((h + top + bottom, w + left + right, c), dtype=img.dtype)
                
                padded[top:top+h, left:left+w, :] = img
                
                for i in range(top):
                    src_row = top - i if top - i < h else 0
                    padded[i, left:left+w, :] = img[src_row, :, :]
                
                for i in range(bottom):
                    src_row = h - 2 - (i % h) if i >= h else h - 2 - i
                    padded[top+h+i, left:left+w, :] = img[src_row, :, :]
                
                for j in range(left):
                    src_col = 2*left - j if 2*left - j < left + w else left
                    padded[:, j, :] = padded[:, src_col, :]
                
                for j in range(right):
                    src_col = left + w - 2 - (j % w) if j >= w else left + w - 2 - j
                    padded[:, left+w+j, :] = padded[:, src_col, :]

        return padded
    
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title('Xiaohei')
    
    pad1 = padding(img, WIDTH, 'edge')
    pad2 = padding(img, WIDTH, 'symmetric')

    plt.subplot(1, 3, 2)
    plt.imshow(pad1)
    plt.title('Xiaohei-fat')

    plt.subplot(1, 3, 3)
    plt.imshow(pad2)
    plt.title('Xiaohei-reflect')

    plt.tight_layout()
    plt.show()



gaosi()
padbynumpy()
padbymyself()