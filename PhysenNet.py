from Unet import *


'''
    physennet  文章中的网络
    
    self.unet = Unet() 用我们定义的unet网络结构中构建了一个成员，作为文章中unet的网络部分，
    
    其余类内成员是用于物理传播过程的，命名均依照了matlab中的代码
    
    tf.cast()用于强制类型转换
'''


class PhysenNet(Model):
    def __init__(self, size=256, d=10, lambd=0.6328e-3, deltax=8e-3, deltay=8e-3, m=1):
        super(PhysenNet, self).__init__()
        self.unet = Unet()

        self.pi = np.pi
        self.size = size
        self.d = d
        self.lambd = lambd
        self.k = 2 * self.pi / self.lambd
        self.deltax = deltax
        self.deltay = deltay
        self.deltal = size * deltax
        self.m = m
        self.r1 = tf.linspace(-size/2, size/2-1, size)
        self.s1 = tf.linspace(-size/2, size/2-1, size)
        self.deltafx = 1 / self.deltal * self.r1
        self.deltafy = 1 / self.deltal * self.s1
        self.meshgrid = tf.meshgrid(self.deltafx, self.deltafy)
        self.h = tf.exp(tf.complex(0., 1.)*tf.cast(self.k*self.d*tf.sqrt(1 - tf.pow(lambd * self.meshgrid[0], 2) - tf.pow(lambd * self.meshgrid[1], 2)), tf.complex64))
        # self.H = self.h[tf.newaxis, :, :, tf.newaxis]

    def call(self, input):

        # zeros = tf.zeros((1, 256, 256, 1), dtype=tf.float32)

        '''
            定义physennet的正向传播过程
            p1 = self.unet(input), 输入经过unet网络输出得到浮点类型p1
            p1.cast（）将浮点类型强转为复数类型得到的是（p1, 0i）
        '''
        p1 = self.unet(input)
        p1 = tf.cast(p1, tf.complex64)
        # p1 = tf.complex(p1, zeros)
        # print('p1', p1)
        # a1 = self.m * tf.exp(1 * 3.1415 * p1)

        '''
            下列为物理传播过程，用到了上面定义的类内成员
            tf.complex(0., 1.)构造了一个复部 i
        '''
        a1 = self.m * tf.exp(tf.complex(0., 1.) * self.pi * p1)

        A1 = tf.signal.fftshift(tf.signal.fft2d(tf.signal.fftshift(a1[0, :, :, 0], name='ffs1'), name='fft1'), name='ffs2')
        U1 = tf.signal.ifftshift(tf.signal.ifft2d(tf.signal.ifftshift(tf.multiply(A1, self.h), name='iffs1'), name='ifft1'), name='iffs2')
        # A1 = tf.signal.fft2d(a1, name='fft')
        # U1 = tf.signal.ifft2d(A1, name='ifft')


        UA10 = tf.multiply(tf.math.conj(U1), U1)

        '''
            UA10仍为复数，tf.math.real()取实数部
        '''
        UA10 = tf.math.real(UA10)

        maxnum = tf.reduce_max(UA10)
        minnum = tf.reduce_min(UA10)
        M5 = (UA10 - minnum) / (maxnum - minnum)

        # M5 = tf.divide((UA10 - minnum), maxnum - minnum)
        # M5 *= 255.
        '''
            这里是为了输出一个1*256*256*1的张量，匹配真值的shape
        '''
        M5 = M5[tf.newaxis, :, :, tf.newaxis]
        M5 *= 255.
        return M5


    '''
        pre为自定的函数，输入衍射强度图，得到仅经过训练好的unet后的相位图。
    '''
    def pre(self, image):
        return self.unet(image)
