# -*- coding: utf-8 -*-
import numpy as np
import cv2
from matplotlib import pyplot as plt

#Doc anh
img = cv2.imread('grid1.jpg', 0)

#Bien doi Fourier
dft = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)

#Di chuyen pho tan so thap tu goc tren ben trai den vi tri trung tam
dft_shift = np.fft.fftshift(dft)

#Chuyen doi phuc hop kenh doi hinh anh quang pho thanh khoang 0-255
result = 20*np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]))

#Hien thi hinh anh
plt.subplot(121), plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(result, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()