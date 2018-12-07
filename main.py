#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 13:55:40 2018

@author: me
"""

import tensorflow as tf
from model import RDN

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean("is_train", True, "if the train")
flags.DEFINE_boolean("matlab_bicubic", False, "using bicubic interpolation in matlab")
flags.DEFINE_integer("image_size", 64, "the size of image input")
flags.DEFINE_integer("c_dim", 5, "the size of channel")
flags.DEFINE_integer("scale", 4, "the size of scale factor for preprocessing input image")
flags.DEFINE_integer("stride", 16, "the size of stride")
flags.DEFINE_integer("epoch", 20, "number of epoch")
flags.DEFINE_integer("batch_size", 32, "the size of batch")
flags.DEFINE_integer("num_h5_file", 100, "number of training h5 files")
flags.DEFINE_integer("num_patches", 160, "number of patches in each h5 file")
flags.DEFINE_float("learning_rate", 1e-4 , "the learning rate")
flags.DEFINE_boolean("is_eval", True, "if the evaluation")
flags.DEFINE_string("test_img", "", "test_img")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "name of the checkpoint directory")
flags.DEFINE_string("result_dir", "result", "name of the result directory")
flags.DEFINE_string("train_set", "DIV2K_train_HR", "name of the train set")
flags.DEFINE_string("test_set", "Set5", "name of the test set")
flags.DEFINE_string("data_path", "./data_generation/h5data/", "The path of h5 files")
flags.DEFINE_integer("D", 4, "D")
flags.DEFINE_integer("C", 5, "C")
flags.DEFINE_integer("G", 32, "G")
flags.DEFINE_integer("G0", 32, "G0")
flags.DEFINE_integer("kernel_size", 3, "the size of kernel")



def main(_):
    rdn = RDN(tf.Session(),
              is_train = FLAGS.is_train,
              is_eval = FLAGS.is_eval,
              image_size = FLAGS.image_size,
              c_dim = FLAGS.c_dim,
              scale = FLAGS.scale,
              batch_size = FLAGS.batch_size,
              D = FLAGS.D,
              C = FLAGS.C,
              G = FLAGS.G,
              G0 = FLAGS.G0,
              kernel_size = FLAGS.kernel_size
              )

    if rdn.is_train:
        rdn.train(FLAGS)
    else:
        if rdn.is_eval:
            rdn.eval(FLAGS)
        else:
            rdn.test(FLAGS)

if __name__=='__main__':
    tf.app.run()
