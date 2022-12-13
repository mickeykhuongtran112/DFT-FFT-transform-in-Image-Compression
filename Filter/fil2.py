# -*- coding: utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread('lena.png', 0)


dft = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)
fshift = np.fft.fftshift(dft)


rows, cols = img.shape
crow,ccol = int(rows/2), int(cols/2) 
mask = np.zeros((rows, cols, 2), np.uint8)
mask[crow-30:crow+30, ccol-30:ccol+30] = 1


f = fshift * mask
print(f.shape, fshift.shape, mask.shape)



ishift = np.fft.ifftshift(f)
iimg = cv2.idft(ishift)
res = cv2.magnitude(iimg[:,:,0], iimg[:,:,1])


plt.subplot(121), plt.imshow(img, 'gray'), plt.title('Original Image')
plt.axis('off')
plt.subplot(122), plt.imshow(res, 'gray'), plt.title('Result Image')
plt.axis('off')
plt.show()
