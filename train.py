from PhysenNet import *
import os
import cv2

'''
    训练网络
'''


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

'''
    读入强度图并复制一份作为真值，分别转化 1*256*256*1的张量
'''
image = cv2.imread('./my_image.png', -1)
image = np.array(image, dtype='float32')
gt = np.copy(image)
image = tf.convert_to_tensor(image)
image = image[tf.newaxis, :, :, tf.newaxis]
gt = tf.convert_to_tensor(gt)
gt = gt[tf.newaxis, :, :, tf.newaxis]



# input = tf.random.uniform([1, 256, 256, 1])
# input = tf.constant(input)
# gt = tf.random.uniform([1, 256, 256, 1])
# gt = tf.constant(gt)


'''
    构造physennet网络对象用于训练
'''
model = PhysenNet()
tf.initializers.he_normal(666)
# model.load_weights('./check_points/physennet.ckpt')
model.load_weights('./check_points/physennet_my_image1.ckpt')




# model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.01), loss=tf.losses.mean_squared_error)


'''
    网络的损失以及优化函数在此定义，用到Adam作为更新权重的方法，MSE作为损失
'''

# my_optimizer = tf.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.01), loss=tf.keras.losses.MSE)

# model.fit(image, gt, batch_size=1, epochs=1000)

# history = model.fit(data_gen.flow(image, batch_size=1), epochs=50)

'''
    网络训练过程
'''

criterism = tf.keras.losses.MeanSquaredError()
val_loss = 15000

epoch = 10000
for e in range(epoch):
    '''
        每次迭代在输入强度图和真值强度图上均加上 0到1/30的平均噪声， 文章中有这一步操作，使网络收敛得更快
    '''
    noice = tf.random.uniform(image.shape, 0, 1/30)
    # noice = tf.cast(noice, tf.complex64)
    x = image + noice
    y = gt + noice
    model.fit(x, y)
    print('\tepoch: {}/{}'.format(e + 1, epoch))
    if(e%1000 == 999):
        model.save_weights('./0611/weight_{}.ckpt'.format(e))
        print('\tweight_{} saved'.format(e))

    output = model.predict(image)
    temp_val_loss = criterism(output, gt)
    print('\ttemp_val_loss: {}'.format(temp_val_loss))

    if temp_val_loss < val_loss:
        model.save_weights('./0611/best_weight.ckpt')
        val_loss = temp_val_loss
        print('\tbest_model saved')
    print('\t'+'=='*30+'\n'*2)


'''
    训练完后，将网络中的权重打印到一个txt文件中
'''
file = open('./my_image_weights.txt', 'w')
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

'''
    保存训练好的网络的权重数据
'''
model.save_weights('./check_points/physennet_my_image2.ckpt')
model.summary()

'''
    这里是尝试预测physennet的输出结果，理想结果应该为强度图
'''
output = model.predict(image)
img = np.array(output[0, :, :, 0], dtype='uint8')
cv2.imshow('img', img)
cv2.waitKey(0)












