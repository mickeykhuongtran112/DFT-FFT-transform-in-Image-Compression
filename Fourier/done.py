# -*- coding: utf-8 -*-
import numpy as np
import cv2
from matplotlib import pyplot as plt

#Doc hinh anh
img = cv2.imread('bird.png', 0)

#Bien doi Fourier
dft = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)
dftshift = np.fft.fftshift(dft)
res1= 20*np.log(cv2.magnitude(dftshift[:,:,0], dftshift[:,:,1]))

#craete a mask first, center square is 1, remaining all zeros
rows, cols = img.shape
crow,ccol = rows/2 , cols/2
mask = np.zeros((rows,cols,2),np.uint8)
mask[int(crow)+30:int(rows), int(ccol)+30:int(cols)] = 1
#apply mask and inverse DFT
fshift = dftshift*mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
res2 = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])


#Hien thi hinh anh
plt.subplot(131), plt.imshow(img, 'gray'), plt.title('Original Image')
plt.axis('off')

plt.subplot(132), plt.imshow(res1, 'gray'), plt.title('Fourier Image')
plt.axis('off')

plt.subplot(133), plt.imshow(res2, 'gray'), plt.title('Processed Fourier Image')
plt.axis('off')
plt.show()