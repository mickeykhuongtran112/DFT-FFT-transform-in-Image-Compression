# -*- coding: utf-8 -*-
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

#Read image
img = cv.imread('lena.png', 0)

#FFT convert to frequency domain
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

#DC
rows, cols = img.shape
crow,ccol = int(rows/2), int(cols/2)
fshift[crow-30:crow+30, ccol-30:ccol+30] = 0

#FFT inverse
ishift = np.fft.ifftshift(fshift)
iimg = np.fft.ifft2(ishift)
iimg = np.abs(iimg)

#Convert image
plt.subplot(121), plt.imshow(img, 'gray'), plt.title('Original Image')
plt.axis('off')
plt.subplot(122), plt.imshow(iimg, 'gray'), plt.title('Result Image')
plt.axis('off')
plt.show()
