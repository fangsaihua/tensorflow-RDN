#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 11:02:31 2018

@author: me
"""

import tensorflow as tf
import time
import cv2
import numpy as np
import os
import h5py
import scipy.ndimage as ndi
import tensorflow.contrib.layers as ly


def read_data(file):
    with h5py.File(file, 'r') as hf:
        ms = hf.get('data')
        pan_4x = hf.get('pan_x4')
        label_4x = hf.get('label_x4')
#        pan = cv2.resize(np.array(pan_4x), None, fx = 1.0 / 4, fy = 1.0 / 4, interpolation = cv2.INTER_LINEAR)
        pan = ndi.zoom(pan_4x, (1, 1, 1.0 / 4, 1.0 / 4), order = 1)
        
        return np.array(ms), np.array(pan), np.array(label_4x), np.array(pan_4x) 


class RDN(object):
    def __init__(self,
             sess,
             is_train,
             is_eval,
             image_size,
             c_dim,
             scale,
             batch_size,
             D,
             C,
             G,
             G0,
             kernel_size
             ):

        self.sess = sess
        self.is_train = is_train
        self.is_eval = is_eval
        self.image_size = image_size             #64
        self.c_dim = c_dim                       #5
        self.scale = scale                       #4
        self.batch_size = batch_size
        self.D = D
        self.C = C
        self.G = G
        self.G0 = G0
        self.kernel_size = kernel_size


    def create_kernel(self, name, shape):
#        new_variables = tf.Variable(tf.random_normal(shape, stddev=0.01), name=name)
        initializer=tf.contrib.layers.xavier_initializer()
        regularizer = tf.contrib.layers.l2_regularizer(scale=1e-5)
        new_variables = tf.get_variable(name=name, shape=shape, initializer=initializer,
                                    regularizer=regularizer, trainable=True)
        
        return new_variables
        
    
    def create_bias(self, name, shape):
        new_variables = tf.Variable(tf.zeros(shape, name=name))
        return new_variables
        
    
    def SFEParams(self):
        G = self.G
        G0 = self.G0
        ks = self.kernel_size
        weightsS = {
            'w_S_1': self.create_kernel('w_S_1', [ks, ks, self.c_dim, G0]),
            'w_S_2': self.create_kernel('w_S_2', [ks, ks, G0, G])
        }
        biasesS = {
            'b_S_1': self.create_bias('b_S_1', [G0]),
            'b_S_2': self.create_bias('b_S_2', [G])
        }

        return weightsS, biasesS
    
    
#    def RDBParams(self):
#        weightsR = {}
#        biasesR = {}
#        D = self.D
#        C = self.C
#        G = self.G
#        ks = self.kernel_size
#
#        for i in range(1, D+1):
#            for j in range(1, C+1):
#                weightsR.update({'w_R_%d_%d' % (i, j): self.create_kernel('w_R_%d_%d' % (i, j), [ks, ks, G * j, G])}) 
#                biasesR.update({'b_R_%d_%d' % (i, j): self.create_bias('b_R_%d_%d' % (i, j), [G])})
#            weightsR.update({'w_R_%d_%d' % (i, C+1): self.create_kernel('w_R_%d_%d' % (i, C+1), [1, 1, G * (C+1), G])})
#            biasesR.update({'b_R_%d_%d' % (i, C+1): self.create_bias('b_R_%d_%d' % (i, C+1), [G])})
#
#        return weightsR, biasesR
    
    
    def DFFParams(self):
        D = self.D
        G = self.G
        G0 = self.G0
        ks = self.kernel_size
        weightsD = {
            'w_D_1': self.create_kernel('w_D_1', [1, 1, G * D, G0]),
            'w_D_2': self.create_kernel('w_D_2', [ks, ks, G0, G0])
        }
        biasesD = {
            'b_D_1': self.create_bias('b_D_1', [G0]),
            'b_D_2': self.create_bias('b_D_2', [G0])
        }

        return weightsD, biasesD

    
    def UPNParams(self):
        G0 = self.G0
        weightsU = {
            'w_U_1': self.create_kernel('w_U_1', [5, 5, G0, 64]),
            'w_U_2': self.create_kernel('w_U_2', [3, 3, 64, 32]),
#            'w_U_3': tf.Variable(tf.random_normal([3, 3, 32, self.c_dim * self.scale * self.scale ], stddev=np.sqrt(2.0/9/32)), name='w_U_3')
            'w_U_3': self.create_kernel('w_U_3', [3, 3, 32, self.c_dim * self.scale * self.scale]) 
        }
        biasesU = {
            'b_U_1': self.create_bias('b_U_1', [64]),
            'b_U_2': self.create_bias('b_U_2', [32]),
            'b_U_3': self.create_bias('b_U_3', [self.c_dim * self.scale * self.scale])
        }

        return weightsU, biasesU

    
    def build_model(self, images_shape, labels_shape):
        self.images = tf.placeholder(tf.float32, images_shape, name='images')
        self.labels = tf.placeholder(tf.float32, labels_shape, name='labels')
        self.lr = tf.placeholder(tf.float32, shape=[])
        
        self.weightsS, self.biasesS = self.SFEParams()
#        self.weightsR, self.biasesR = self.RDBParams()
        self.weightsD, self.biasesD = self.DFFParams()
        self.weightsU, self.biasesU = self.UPNParams()
#        self.weight_final = tf.Variable(tf.random_normal([self.kernel_size, self.kernel_size, self.c_dim, self.c_dim], stddev=np.sqrt(2.0/9/3)), name='w_f')
        self.weight_final = self.create_kernel('w_f', [self.kernel_size, self.kernel_size, self.c_dim, self.c_dim])
        self.bias_final = self.create_bias('b_f', [self.c_dim])
        
        self.pred = self.model()
        self.loss = tf.reduce_mean(tf.square(self.labels - self.pred))
        self.summary = tf.summary.scalar('loss', self.loss)
        self.saver = tf.train.Saver()
        
        
    def UPN(self, input_layer):
        x = tf.nn.conv2d(input_layer, self.weightsU['w_U_1'], strides=[1,1,1,1], padding='SAME') + self.biasesU['b_U_1']
        x = tf.nn.relu(x)
        x = tf.nn.conv2d(x, self.weightsU['w_U_2'], strides=[1,1,1,1], padding='SAME') + self.biasesU['b_U_2']
        x = tf.nn.relu(x)
        x = tf.nn.conv2d(x, self.weightsU['w_U_3'], strides=[1,1,1,1], padding='SAME') + self.biasesU['b_U_3']

        x = self.PS(x, self.scale)

        return x
    
    
#    def RDBs(self, input_layer):
#        rdb_concat = list()
#        rdb_in = input_layer
#        for i in range(1, self.D+1):
#            x = rdb_in
#            for j in range(1, self.C+1):
#                tmp = tf.nn.conv2d(x, self.weightsR['w_R_%d_%d' %(i, j)], strides=[1,1,1,1], padding='SAME') + self.biasesR['b_R_%d_%d' % (i, j)]
#                tmp = tf.nn.relu(tmp)
#                x = tf.concat([x, tmp], axis=3)
#
#            x = tf.nn.conv2d(x, self.weightsR['w_R_%d_%d' % (i, self.C+1)], strides=[1,1,1,1], padding='SAME') +  self.biasesR['b_R_%d_%d' % (i, self.C+1)]
#            rdb_in = tf.add(x, rdb_in)
#            rdb_concat.append(rdb_in)
#
#        return tf.concat(rdb_concat, axis=3)
        
    
    def RDBs(self, input_layer):
        rdb_concat = list()
        rdb_in = input_layer
        for i in range(1, self.D+1):
            x = rdb_in
            rdb_in = self.RDB(x)
            rdb_concat.append(rdb_in)
            
        return tf.concat(rdb_concat, axis=3)
        
        
    
    def RDB(self, input_layer):
        weight_decay = 1e-5
        G = self.G
        num_res = 2;
        rs = ly.conv2d(input_layer, num_outputs = G, kernel_size = 3, stride = 1, 
                          weights_regularizer = ly.l2_regularizer(weight_decay), 
                          weights_initializer = ly.xavier_initializer(),activation_fn = tf.nn.relu)
        for i in range(num_res):
            rs1 = ly.conv2d(rs, num_outputs = G, kernel_size = 3, stride = 1, 
                          weights_regularizer = ly.l2_regularizer(weight_decay), 
                          weights_initializer = ly.xavier_initializer(),activation_fn = tf.nn.relu)
            rs1 = ly.conv2d(rs1, num_outputs = G, kernel_size = 3, stride = 1, 
                          weights_regularizer = ly.l2_regularizer(weight_decay), 
                          weights_initializer = ly.xavier_initializer(),activation_fn = None)
            rs = tf.add(rs,rs1)
        
        rs = ly.conv2d(rs, num_outputs = G, kernel_size = 3, stride = 1, 
                          weights_regularizer = ly.l2_regularizer(weight_decay), 
                          weights_initializer = ly.xavier_initializer(),activation_fn = None)
        return rs
    
    
    def model(self):
        F_1 = tf.nn.conv2d(self.images, self.weightsS['w_S_1'], strides=[1,1,1,1], padding='SAME') + self.biasesS['b_S_1']
        F0 = tf.nn.conv2d(F_1, self.weightsS['w_S_2'], strides=[1,1,1,1], padding='SAME') + self.biasesS['b_S_2']

        FD = self.RDBs(F0)

        FGF1 = tf.nn.conv2d(FD, self.weightsD['w_D_1'], strides=[1,1,1,1], padding='SAME') + self.biasesD['b_D_1']
        FGF2 = tf.nn.conv2d(FGF1, self.weightsD['w_D_2'], strides=[1,1,1,1], padding='SAME') + self.biasesD['b_D_2']

        FDF = tf.add(FGF2, F_1)      
        
        FU = self.UPN(FDF)
        # FU = self.UPN(F_1)
        IHR = tf.nn.conv2d(FU, self.weight_final, strides=[1,1,1,1], padding='SAME') + self.bias_final

        return IHR
    
    
    def _phase_shift(self, I, r):
        # Helper function with main phase shift operation
        bsize, a, b, c = I.get_shape().as_list()
        bsize = tf.shape(I)[0]
        X = tf.reshape(I, [bsize, a, b, r, r])
#        X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
        X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
        X = tf.concat([tf.squeeze(x, axis=1) for x in X], 2)  # bsize, b, a*r, r
        X = tf.split(X, b, 1)  # b, [bsize, a*r, r]
        X = tf.concat([tf.squeeze(x, axis=1) for x in X], 2)  # bsize, a*r, b*r
        return tf.reshape(X, (bsize, a*r, b*r, 1))


    # NOTE: test without batchsize
    def _phase_shift_test(self, I ,r):
        bsize, a, b, c = I.get_shape().as_list()
        X = tf.reshape(I, (1, a, b, r, r))
        X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
        X = tf.concat([tf.squeeze(x) for x in X], 1)  # bsize, b, a*r, r
        X = tf.split(X, b, 0)  # b, [bsize, a*r, r]
        X = tf.concat([tf.squeeze(x) for x in X], 1)  # bsize, a*r, b*r
        return tf.reshape(X, (1, a*r, b*r, 1))


    def PS(self, X, r):
        # Main OP that you can arbitrarily use in you tensorflow code
        Xc = tf.split(X, self.c_dim, 3)
        if self.is_train:
            X = tf.concat([self._phase_shift(x, r) for x in Xc], 3) # Do the concat RGB
        else:
            X = tf.concat([self._phase_shift_test(x, r) for x in Xc], 3) # Do the concat RGB
        return X


    def train(self, config):
        print("\nPrepare Data...\n")        
        images_shape = [None, self.image_size, self.image_size, self.c_dim]
        labels_shape = [None, self.image_size * self.scale, self.image_size * self.scale, self.c_dim]

        self.build_model(images_shape, labels_shape)
#        self.train_op = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(self.loss)
        self.train_op = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(self.loss)
        tf.global_variables_initializer().run(session=self.sess) 
        
        counter = self.load(config.checkpoint_dir)
        time_ = time.time()
        print("\nNow Start Training...\n")
#        if tf.train.get_checkpoint_state('./model/'):  # load previous trained model
#            ckpt = tf.train.latest_checkpoint('./model/')
#            saver.restore(self.sess, ckpt)
#            ckpt_num = re.findall(r"\d", ckpt)
#            if len(ckpt_num) == 3:
#                start_point = 100 * int(ckpt_num[0]) + 10 * int(ckpt_num[1]) + int(ckpt_num[2])
#            elif len(ckpt_num) == 2:
#                start_point = 10 * int(ckpt_num[0]) + int(ckpt_num[1])
#            else:
#                start_point = int(ckpt_num[0])
#            print("Load success")
#            print(start_point)
#
#        else:
#            print("re-training")
#            start_point = 0
        data_path = config.data_path
        for ep in range(config.epoch):
            if ep + 1 > (config.epoch / 3):  # reduce learning rate
                config.learning_rate = config.learning_rate * 0.1
            if ep + 1 > (2 * config.epoch / 3):
                config.learning_rate = config.learning_rate * 0.01
            for num in range(config.num_h5_file):
                train_data_name = "train" + str(num + 1) + ".h5"
                train_ms, train_pan, train_ms4, train_pan4 = read_data(data_path + train_data_name)
                train_input = np.concatenate((train_ms, train_pan), axis = 1)
                train_label = np.concatenate((train_ms4, train_pan4), axis = 1)
                
                train_input = np.transpose(train_input, (0, 2, 3, 1))  
                train_label = np.transpose(train_label, (0, 2, 3, 1))  
                
                indexs = np.arange(config.num_patches)
                np.random.shuffle(indexs)
                data_size = int(config.num_patches / config.batch_size)
                for i in range(data_size):
#                    rand_index = np.arange(int(i * config.batch_size), int((i + 1) * config.batch_size))
                    rand_index = indexs[int(i * config.batch_size) : int((i + 1) * config.batch_size)]
                    batch_images, batch_labels = train_input[rand_index,:,:,:], train_label[rand_index,:,:,:]
                    counter += 1
                    
                    _, err = self.sess.run([self.train_op, self.loss], feed_dict={self.lr: config.learning_rate, self.images: batch_images, self.labels: batch_labels})
                    if counter % 10 == 0:
                        print("Epoch: [%2d], batch: [%2d/%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" % ((ep+1), i, data_size, counter, time.time()-time_, err))
                    if counter % 100 == 0:
                        self.save(config.checkpoint_dir, counter)
                        # summary_str = self.sess.run(merged_summary_op)
                        # summary_writer.add_summary(summary_str, counter)
                    
#                    if counter > 0 and counter == data_size * config.epoch:
#                        return


    def load(self, checkpoint_dir):
        print("\nReading Checkpoints.....\n")
        model_dir = "%s_%s_%s_%s_x%s" % ("rdn", self.D, self.C, self.G, self.scale)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
 
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_path = str(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(os.getcwd(), ckpt_path))
            step = int(os.path.basename(ckpt_path).split('-')[1])
            print("\nCheckpoint Loading Success! %s\n" % ckpt_path)
        else:
            step = 0
            print("\nCheckpoint Loading Failed! \n")

        return step
                
    
    def save(self, checkpoint_dir, step):
        model_name = "RDN.model"
        model_dir = "%s_%s_%s_%s_x%s" % ("rdn", self.D, self.C, self.G, self.scale)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
             os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    