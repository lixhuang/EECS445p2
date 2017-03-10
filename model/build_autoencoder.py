"""
EECS 445 - Introduction to Machine Learning
Winter 2017 - Project 2
Build Autoencoder - Skeleton
Build TensorFlow computation graph for autoencoder network
Usage: `from model.build_autoencoder import autoencoder`
Additionally, a naive compression scheme can be imported by:
`from model.build_autoencoder import naive`
"""

import tensorflow as tf
from utils.config import get


def shrink(x, in_length=32, scale=2):
    ''' Resize given image by shrinking by `scale` in linear scale '''
    as_image = tf.reshape(x, [-1, in_length, in_length, 1])
    pooled = tf.nn.avg_pool(as_image, ksize=[1, scale, scale, 1], strides=[1, scale, scale, 1], padding='SAME')
    as_vector = tf.reshape(pooled, [-1, (in_length//scale)**2]) 
    return as_vector


def grow(x, channel_dim=16, in_length=18, scale=2, out_length=32):
    ''' ``Deconvolution layer'': magnify small image with many channels to
        large, 1-channel image. Then crop.
    '''
    magnified_length = in_length * scale 
    crop_offset = (magnified_length-out_length) // 2

    W = tf.Variable(tf.random_normal([5, 5, 1, channel_dim], stddev=0.01))
    b = tf.Variable(tf.constant(0.00, shape=[1]))

    as_image = tf.reshape(x, [-1, in_length, in_length, channel_dim])
    conv = b + tf.nn.conv2d_transpose(as_image, W, [tf.shape(as_image)[0], magnified_length, magnified_length, 1], strides=[1, scale, scale, 1])
    crop = tf.slice(conv, [0, crop_offset, crop_offset, 0], [tf.shape(conv)[0], out_length, out_length, 1])
    as_vector = tf.reshape(crop, [-1, out_length**2])
    return as_vector


def normalize(x): 
    ''' Set mean to 0.0 and standard deviation to 1.0 via affine transform '''
    shifted = (x - tf.reduce_mean(x)) 
    scaled = shifted / tf.sqrt(tf.reduce_mean(tf.multiply(shifted, shifted)))
    return scaled

# TODO: if you write helper functions for building your neural nets, place
#       those helper functions here. Above are three functions that we used
#       when solving this problem. You may find them helpful.


def autoencoder():
    ''' Autoencoder architecture (see specs) '''
    # TODO: implement the architecture specified in the project document.
    #      Return the input, compressed, and output layers. You may choose
    #      to rely on the 'MODEL.' parameters found in `config.json`, or you
    #      may hardcode constants. For easy development, we recommend 
    #      creating helper functions. See the implementation of `naive`
    #      below for an example of correct TensorFlow. This model,
    #      `autoencoder`, may be somewhat longer. 
    #
    #      As stated in project docs, you may modify the body of this function
    #      by adding, removing, or changing lines. 
    orig = tf.placeholder(tf.float32, shape=[None, 1024])  # TODO
    compressed = shrink(orig, scale=2)  # TODO
    recon = None  # TODO
    recon = grow(recon, channel_dim=16, in_length=18, scale=2, out_length=32) 
    recon = normalize(recon) # Note: this makes `grow`'s bias variable have no effect;
                             #       we decide to include a bias variable in `grow` 
                             #       for generality.
    return orig, compressed, recon


def naive():
    ''' Compress by down-sampling; decompress by up-sampling '''
    orig = tf.placeholder(tf.float32, shape=[None, 1024])
    compressed = shrink(orig, scale=32//get('MODEL.SQRT_REPR_DIM'))
    recon = normalize(compressed)
    return orig, compressed, recon

