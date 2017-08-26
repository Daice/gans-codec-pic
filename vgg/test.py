import numpy as np
import scipy.misc
import cv2
import glob
import os
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
sys.path.append('../../utils')
sys.path.append('../../vgg19')
from testsrgan import SRGAN
from vgg import VGG19

model = VGG19(h,w,False)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

saver = tf.train.Saver()
saver.restore(sess, '../backup/latest')

img = cv2.imread('baboon.bmp')
h, w, c = img.shape
input_ = np.zeros((1,h,w,c))
rbg_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = np.array(rbg_img) / 127.5 - 1
input_[0] = img
_, x_phi = sess.run([model.out, model.phi], feed_dict={model.x:input_, model.is_training:False})


for i in range(len(x_phi)):
    ximg = x_phi[i][0][:,:,0]
    ximg = np.uint8((ximg+1)*127.5)
    ximg =cv2.cvtColor(ximg,cv2.COLOR_BGR2RGB)
    cv2.imwrite(str(i)+'.png',ximg) 

'''
paths = glob.glob(os.path.join('test', '*'))
paths = sorted(paths)
for path in paths:
    bgr_img = cv2.imread(path)
    h,w ,c = bgr_img.shape
    input_ = np.zeros((1,h,w,c)) 
    rbg_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    img = np.array(rbg_img) / 127.5 - 1
    input_[0] = img
    mos,fake = sess.run(
        [model.downscaled,model.imitation],
        feed_dict={model.x:input_,model.is_training:False})
    #imos = np.uint8((mos[0]+1)*127.5)
    ifake = np.uint8((fake[0]+1)*127.5)
    name = path.split('/')[1]
    #imos = cv2.cvtColor(imos, cv2.COLOR_BGR2RGB)
    ifake = cv2.cvtColor(ifake, cv2.COLOR_BGR2RGB)
    #cv2.imwrite('Set5/input_'+name,imos)
    cv2.imwrite('testdc/dc_'+name,ifake)
'''

