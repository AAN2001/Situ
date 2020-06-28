from PhysenNet import *
import cv2

'''
    网络预测可视化，输入强度图分别输出了unet的结果和physennet的结果
    理想情况下unet应输出相位图，physennet应输出强度图
'''


os.environ["CUDA_VISIBLE_DEVICES"]="-1"

image = cv2.imread('./my_image.png', -1)
image = np.array(image, dtype='float32')
gt = np.copy(image)
image = tf.convert_to_tensor(image)
image = image[tf.newaxis, :, :, tf.newaxis]
gt = tf.convert_to_tensor(gt)
gt = gt[tf.newaxis, :, :, tf.newaxis]

model = PhysenNet()
# model.load_weights('./check_points/physennet2.ckpt')
model.load_weights('./0611/best_weight.ckpt')


model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.01), loss='mse')
unet_pre = model.pre(image)
print(unet_pre)
print(unet_pre.dtype, unet_pre.shape)
physennet_predict = model.predict(image)
print(physennet_predict)
print(physennet_predict.dtype, physennet_predict.shape)
img = np.array(unet_pre[0, :, :, 0] * 255, dtype='uint8')
img2 = np.array(physennet_predict[0, :, :, 0], dtype='uint8')
# dis = np.abs(img-output2)
# print(dis)
cv2.imshow('phase', img)
cv2.imshow('magnitude', img2)
# cv2.imshow('dis', dis)
cv2.waitKey(0)




