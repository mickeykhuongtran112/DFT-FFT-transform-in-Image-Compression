# -*- coding: utf-8 -*-
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

#Doc anh
img = cv.imread('dogg.jpg', 0)

#FFT de lay phan bo tan so
f = np.fft.fft2(img)

#Vi tri trung tam ket qua mac dinh nam o goc tren ben trai,
#Goi ham fftshift de dich chuyen trung tam ve vi tri giua
fshift = np.fft.fftshift(f)       

#fft ket qua la mot so phuc, ket qua gia tri tuyet doi cua no la bien do
fimg = np.log(np.abs(fshift))

#Hien thi ket qua
plt.subplot(121), plt.imshow(img, 'gray'), plt.title('Original Fourier')
plt.axis('off')
plt.subplot(122), plt.imshow(fimg, 'gray'), plt.title('Fourier Fourier')
plt.axis('off')
plt.show()