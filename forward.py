import tensorflow as tf
import cv2
import numpy as np


'''
    仿照matlab代码自己写的一个物理传播过程，由相位得到衍射强度图
'''



phase = cv2.imread('./phase.jpg', cv2.IMREAD_GRAYSCALE)
phase = phase / 255.
phase *= np.pi

lambd = 0.6328e-3
deltaX = 8e-3
deltaY = 8e-3
k = 2 * np.pi / lambd
d3 = 10

shape = phase.shape

N = shape[0]
deltaL = N * deltaX

M1 = tf.complex(1., 0.)
a1 = M1 * tf.exp(tf.complex(0., 1.) * tf.complex(np.pi, 0.) * tf.cast(phase, tf.complex64))
A1 = tf.signal.fftshift(tf.signal.fft2d(tf.signal.fftshift(a1, name='ffs1'), name='fft1'), name='ffs2')

r1 = tf.linspace(-shape[0]/2, shape[0]/2-1, shape[0])
s1 = tf.linspace(-shape[0]/2, shape[0]/2-1, shape[0])
deltaFX = 1 / deltaL * r1
deltaFY = 1 / deltaL * s1
meshgrid = tf.meshgrid(deltaFX, deltaFY)

h3 = tf.exp(tf.complex(0., 1.)*tf.cast(k*d3*tf.sqrt(1 - tf.pow(lambd * meshgrid[0], 2) - tf.pow(lambd * meshgrid[1], 2)), tf.complex64))

U1 = tf.signal.ifftshift(tf.signal.ifft2d(tf.signal.ifftshift(tf.multiply(A1, h3), name='iffs1'), name='ifft1'), name='iffs2')

UA10 = tf.multiply(tf.math.conj(U1), U1)
UA10 = tf.math.real(UA10)
maxnum = tf.reduce_max(UA10)
minnum = tf.reduce_min(UA10)

M5 = tf.divide((UA10 - minnum), maxnum - minnum)
M5 *= 255.

M5 = np.array(M5, dtype='uint8')
cv2.imwrite('./my_image.png', M5)
cv2.imshow('m5', M5)
cv2.waitKey(0)
