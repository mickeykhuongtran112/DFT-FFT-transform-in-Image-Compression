# -*- coding: utf-8 -*-
import numpy as np
import cv2
from matplotlib import pyplot as plt

#Doc anh
img = cv2.imread('lena.png', 0)

#Bien doi Fourier
dft = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)

#Di chuyen pho tan so thap tu goc tren ben trai den vi tri trung tam
dft_shift = np.fft.fftshift(dft)
#create a mask first, center square is 1, remaining all zeros
rows, cols = img.shape
crow,ccol = rows/2 , cols/2
mask = np.zeros((rows,cols,2),np.uint8)
mask[int(crow)-40:int(crow)+40, int(ccol)-40:int(ccol)+40] = 1

#apply mask and inverse DFT
fshift = dft_shift*mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

#Chuyen doi phuc hop kenh doi hinh anh quang pho thanh khoang 0-255
result = 20*np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]))
rs = 20*np.log(cv2.magnitude(fshift[:,:,0], fshift[:,:,1]))
#Hien thi hinh anh
plt.subplot(121), plt.imshow(result, cmap = 'gray')
plt.title('FFT'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(rs, cmap = 'gray')
plt.title('FFT*mask'), plt.xticks([]), plt.yticks([])
plt.show()