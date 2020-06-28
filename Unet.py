import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, LeakyReLU
from tensorflow.keras import Model


'''
    unet的网络结构在这个类中定义，
'''

class Unet(Model):
    def __init__(self):
        super(Unet, self).__init__()
        self.inititalizers = tf.initializers.he_uniform(666)

        '''
            sequential相当于将几个网络零件依次堆在一起（卷积、激活、降采样等均可）
            
            Conv2d ：卷积层
            BatchNormalization ：批归一化处理层，
            LeakyRelu ：激活函数
            Sequanntial中的传播过程即为 卷积->bn->激活->卷积->bn->激活
            Maxpool2d ：最大池化层
            Conv2DTranspose ： 转置卷积层，反卷积，用于上采样
            
            
            self.XXX  这些是自己定义的类内成员，后面定义正向过程时用到
        '''
        self.down1 = tf.keras.models.Sequential([
            Conv2D(filters=16, kernel_size=3, strides=1, padding='same', kernel_initializer=self.inititalizers),
            BatchNormalization(),
            LeakyReLU(),
            Conv2D(filters=16, kernel_size=3, strides=1, padding='same', kernel_initializer=self.inititalizers),
            BatchNormalization(),
            LeakyReLU()
        ])

        self.down2 = tf.keras.Sequential([
            MaxPool2D(pool_size=(2, 2), strides=2, padding='same'),
            Conv2D(filters=32, kernel_size=3, strides=1, padding='same', kernel_initializer=self.inititalizers),
            BatchNormalization(),
            LeakyReLU(),
            Conv2D(filters=32, kernel_size=3, strides=1, padding='same', kernel_initializer=self.inititalizers),
            BatchNormalization(),
            LeakyReLU()
        ])

        self.down3 = tf.keras.Sequential([
            MaxPool2D(pool_size=(2, 2), strides=2, padding='same'),
            Conv2D(filters=64, kernel_size=3, strides=1, padding='same', kernel_initializer=self.inititalizers),
            BatchNormalization(),
            LeakyReLU(),
            Conv2D(filters=64, kernel_size=3, strides=1, padding='same', kernel_initializer=self.inititalizers),
            BatchNormalization(),
            LeakyReLU()
        ])

        self.down4 = tf.keras.Sequential([
            MaxPool2D(pool_size=(2, 2), strides=2, padding='same'),
            Conv2D(filters=128, kernel_size=3, strides=1, padding='same', kernel_initializer=self.inititalizers),
            BatchNormalization(),
            LeakyReLU(),
            Conv2D(filters=128, kernel_size=3, strides=1, padding='same', kernel_initializer=self.inititalizers),
            BatchNormalization(),
            LeakyReLU()
        ])

        self.down5 = tf.keras.Sequential([
            MaxPool2D(pool_size=(2, 2), strides=2, padding='same'),
            Conv2D(filters=256, kernel_size=3, strides=1, padding='same', kernel_initializer=self.inititalizers),
            BatchNormalization(),
            LeakyReLU(),
            Conv2D(filters=256, kernel_size=3, strides=1, padding='same', kernel_initializer=self.inititalizers),
            BatchNormalization(),
            LeakyReLU(),
            Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding='same', output_padding=1),
            BatchNormalization(),
            LeakyReLU()
        ])

        self.up4 = tf.keras.models.Sequential([
            Conv2D(filters=128, kernel_size=3, strides=1, padding='same', kernel_initializer=self.inititalizers),
            BatchNormalization(),
            LeakyReLU(),
            Conv2D(filters=128, kernel_size=3, strides=1, padding='same', kernel_initializer=self.inititalizers),
            BatchNormalization(),
            LeakyReLU(),
            Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', output_padding=1, kernel_initializer=self.inititalizers),
            BatchNormalization(),
            LeakyReLU()
        ])

        self.up3 = tf.keras.models.Sequential([
            Conv2D(filters=64, kernel_size=3, strides=1, padding='same', kernel_initializer=self.inititalizers),
            BatchNormalization(),
            LeakyReLU(),
            Conv2D(filters=64, kernel_size=3, strides=1, padding='same', kernel_initializer=self.inititalizers),
            BatchNormalization(),
            LeakyReLU(),
            Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', output_padding=1, kernel_initializer=self.inititalizers),
            BatchNormalization(),
            LeakyReLU()
        ])

        self.up2 = tf.keras.models.Sequential([
            Conv2D(filters=32, kernel_size=3, strides=1, padding='same', kernel_initializer=self.inititalizers),
            BatchNormalization(),
            LeakyReLU(),
            Conv2D(filters=32, kernel_size=3, strides=1, padding='same', kernel_initializer=self.inititalizers),
            BatchNormalization(),
            LeakyReLU(),
            Conv2DTranspose(filters=16, kernel_size=3, strides=2, padding='same', output_padding=1, kernel_initializer=self.inititalizers),
            BatchNormalization(),
            LeakyReLU()
        ])

        self.up1 = tf.keras.models.Sequential([
            Conv2D(filters=1, kernel_size=3, strides=1, padding='same', kernel_initializer=self.inititalizers),
            BatchNormalization(),
            Activation('relu')
        ])

        '''
            call  为自定义的网络正向传播过程，用到了上面的类内成员
            concat  将两个张量连接起来
        '''

    def call(self, input):
        d1 = self.down1(input)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        u4 = self.up4(tf.concat([d4, d5], axis=3))
        u3 = self.up3(tf.concat([d3, u4], axis=3))
        u2 = self.up2(tf.concat([d2, u3], axis=3))
        u1 = self.up1(tf.concat([d1, u2], axis=3))
        return u1




