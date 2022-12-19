#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 16:40:31 2020

@author: jeremyjohnston
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers

class AMPNet(tf.keras.Model):
  
  def __init__(self, p, num_stages, *args, **kwargs):
    super().__init__()

    self.N = p.N
    self.M = p.M
    self.Ng = p.Ng
    self.Phi = p.Phi
    self.Layers=[]
    istabundle = istaBundle(p, num_stages['ista'])

    for i in range(num_stages['amp']):
      self.Layers.append(ampBundle(p, istabundle, i))

    self.loss_fcn = tf.keras.losses.MeanSquaredError()

  def call(self, Y):
    Yre, Yim = Y

    Xre = tf.zeros((self.N,self.M), dtype=tf.float32)
    Xim = tf.zeros((self.N,self.M), dtype=tf.float32)
    
    Zre = tf.zeros((self.Ng, self.N))
    Zim = tf.zeros((self.Ng, self.N))


    Rre = Yre
    Rim = Yim
    for l in self.Layers:
      Xre, Xim, Rre, Rim, Zre, Zim = l(Yre, Yim, Xre, Xim, Rre, Rim, Zre, Zim)

    return Xre, Xim

  @tf.function
  def train_step(self, x, y_true):
    with tf.GradientTape() as tape:
      y_pred = self(x, training=True)
      loss = self.loss_fcn(y_pred[0], y_true[0]) + self.loss_fcn(y_pred[1], y_true[1])

    trainable_variables = self.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss

class ampBundle(layers.Layer):
  def __init__(self, p, istabundle, layerid, *args):
    super().__init__()
    self.l1 = ampLayer1(p)
    self.l2 = ampLayer2(p,layerid)
    self.ista = istabundle
    M3 = p.A.T.conj()
    self.M3re = tf.Variable(initial_value=np.float32(M3.real),
                         trainable=True, name='M3re')
    self.M3im = tf.Variable(initial_value=np.float32(M3.imag),
                         trainable=True, name='M3im')

  def call(self, Yre, Yim, Xre, Xim, Rre, Rim, Zre, Zim):

    Xtildere, Xtildeim = self.l1(Xre, Xim, Rre, Rim, self.M3re, self.M3im)


    Xre, Xim, Zre, Zim = self.ista(Xtildere, Xtildeim, Zre, Zim)

    Rre, Rim = self.l2(Yre, Yim, Xre, Xim, Rre, Rim, self.M3re, self.M3im)

    return Xre, Xim, Rre, Rim, Zre, Zim

class istaBundle(layers.Layer):
  def __init__(self, p, num_stages, *args):
    super().__init__()
    self.Layers = []
    self.p = p
    for i in range(num_stages):
      self.Layers.append(istaStage(p))
    
  def call(self, Xre, Xim, Zre, Zim):
    
    for l in self.Layers:
      Zre, Zim, M2 = l(Xre, Xim, Zre, Zim)

    M2re, M2im = M2

    Xhatre = tf.transpose(tf.matmul(tf.transpose(M2re)/self.p.alpha0, Zre) - tf.matmul(tf.transpose(-M2im)/self.p.alpha0, Zim), perm=(0,2,1))
    Xhatim = tf.transpose(tf.matmul(tf.transpose(-M2im)/self.p.alpha0, Zre) + tf.matmul(tf.transpose(M2re)/self.p.alpha0, Zim), perm=(0,2,1))
    
    return Xhatre, Xhatim, Zre, Zim

class istaStage(layers.Layer):

  def __init__(self, p, *args):
    super().__init__()

    self.p = p
    M1 = np.eye(p.Ng) - p.alpha*np.matmul(p.Phi.T.conj(),p.Phi)
    M2 = p.alpha*p.Phi.T.conj()

    self.M1re = tf.Variable(initial_value=np.float32(M1.real),
                         trainable=True, name='M1re')
    self.M1im = tf.Variable(initial_value=np.float32(M1.imag),
                         trainable=True, name='M1im')
    
    self.M2re = tf.Variable(initial_value=np.float32(M2.real),
                         trainable=True, name='M2re')
    self.M2im = tf.Variable(initial_value=np.float32(M2.imag),
                         trainable=True, name='M2im')

    self.lam = tf.Variable(initial_value=np.float32(p.alpha0*p.lam),
                         trainable=True, name='lam')
  
  def soft_threshold_complex(self, x_re, x_im, kappa):
    # x is a shape (N,2) array whose rows correspond to complex numbers
    # returns shape (N,2) array corresponding to complex numbers

    norm = tf.sqrt(x_re**2 + x_im**2)
    x_re_normalized = tf.math.divide_no_nan(x_re,norm)
    x_im_normalized = tf.math.divide_no_nan(x_im,norm)
    maxterm = tf.maximum(norm - kappa,0)
    z_re = tf.math.multiply(x_re_normalized, maxterm)
    z_im = tf.math.multiply(x_im_normalized, maxterm)
    return z_re, z_im

  def call(self, Xre, Xim, Zre, Zim):
    Are = tf.matmul(self.M1re, Zre) - tf.matmul(self.M1im, Zim)
    Aim = tf.matmul(self.M1im, Zre) + tf.matmul(self.M1re, Zim)

    Bre = tf.matmul(Xre, tf.transpose(self.M2re)) - tf.matmul(Xim, tf.transpose(self.M2im))
    Bim = tf.matmul(Xre, tf.transpose(self.M2im)) + tf.matmul(Xim, tf.transpose(self.M2re))

    Bre = tf.transpose(Bre, perm=(0,2,1))
    Bim = tf.transpose(Bim, perm=(0,2,1))

    Cre = Are + Bre
    Cim = Aim + Bim

    M2 = self.M2re, self.M2im
    Zre, Zim = self.soft_threshold_complex(Cre, Cim, self.lam)
    return Zre, Zim, M2

class ampLayer1(layers.Layer):

  def __init__(self, p, *args):
    super().__init__()

  def call(self, Xre, Xim, Rre, Rim, M3re, M3im):
    Xtildere = tf.matmul(M3re, Rre) - tf.matmul(M3im, Rim) + Xre
    Xtildeim = tf.matmul(M3im, Rre) + tf.matmul(M3re, Rim) + Xim

    return Xtildere, Xtildeim

class ampLayer2(layers.Layer):

  def __init__(self, p, layerid, *args):
    super().__init__()

    self.damp = tf.Variable(initial_value=np.float32(p.damp_init),
                            trainable=True, name='damp_{}'.format(layerid))

    
  def call(self, Yre, Yim, Xre, Xim, Rre, Rim, M3re, M3im):

    Rnewre = Yre - (tf.matmul(tf.transpose(M3re), Xre) - tf.matmul(-tf.transpose(M3im), Xim))
    Rnewim = Yim - (tf.matmul(tf.transpose(-M3im), Xre) + tf.matmul(tf.transpose(M3re), Xim))

    Rre = self.damp*Rre + (1-self.damp)*Rnewre
    Rim = self.damp*Rim + (1-self.damp)*Rnewim

    return Rre, Rim
