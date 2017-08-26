import os
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from load import load
from srgan import SRGAN

learning_rate = 1e-4
batch_size = 32
img_dim = 96

vgg_model = '../vgg19/backup/latest'

def main:
	    model = SRGAN()
    sess = tf.Session()
    
    sess.run(tf.global_variables_initializer())
    var_ = tf.global_variables()

    # Restore the VGG-19 network
    vgg_var = [var for var in var_ if "vgg19" in var.name]
    saver = tf.train.Saver(vgg_var)
    saver.restore(sess, vgg_model)

    # Restore the SRGAN network
    if tf.train.get_checkpoint_state('backup/'):
        saver = tf.train.Saver()
        saver.restore(sess, 'backup/latest')

if __name__ == '__main__':
    train()
