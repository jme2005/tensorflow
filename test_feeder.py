# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 11:14:09 2016

@author: edvinj
"""

import tensorflow as tf
import numpy as np


def score(Y,Yhat):
    score = 0
    for n,(y_, y_h) in enumerate(zip(Y,Yhat)):
        if np.argmax(y_) == np.argmax(y_h):
            score += 1
        else:
            print n
    return score


def modifyX(X):
    
    for line in X:
        flip = np.random.random_integers(0,3,1)
        if flip == 1 or flip == 2 or flip == 3:
            n = 1 #np.random.random_integers(0,9)
            m = np.random.random_integers(0,8)
            m2 = np.random.random_integers(m,9)
            line[n+(m*5)] = -1
            if flip == 2:
                line[n+(m2*5)] = -1
            if flip == 3:
                line[n+(m2*5)+1] = -1
    return X      

def generateYmod(X):
    output = []
    for line in X:
        start = False
        for n,word in enumerate(np.split(line,10)):
            
            tmp = np.zeros(10)
            
            if word.sum() == 5 and not start:
                tmp[0] = 1
            elif word.sum() < 5 and not start:
                tmp[0] = 1
                start = True
                ctr = 1
            elif word.sum() < 5 and start:
                tmp[ctr] = 1
                start = False
            elif start:
                tmp[ctr] = 1
                ctr += 1
            output.append(tmp)
    return output

def generateY(y):
    output = []
    for y_ in y:
        tmp = np.zeros(10)
        tmp[y_] = 1
        output.append(tmp)
    return output

config = Config.create_default()
#X = np.random.uniform(0,1,25*50).reshape(50,25)
X = np.random.random_integers(1,1,50*400).reshape(400,50)
X = modifyX(X)
#y = np.random.random_integers(0,4,50*5)
#y = [np.linspace(0,9,10)]*50
#y = [int(item) for sublist in y for item in sublist]
#Y = generateY(y)
Y = generateYmod(X)

with tf.Graph().as_default(), tf.Session() as session:
    initializer = tf.random_uniform_initializer(-1,1)
    with tf.variable_scope("model", reuse=None, initializer=initializer):
        m = CUDARNN(is_training=True, config=config)
    with tf.variable_scope("model", reuse=True, initializer=initializer):
        mvalid = CUDARNN(is_training=False, config=config)
      

    tf.initialize_all_variables().run()
    
    for i in range(2000):
        
       cost = session.run([m._cost, m._train_op],feed_dict = {m._input_data:X,m._targets:Y})
       print i, cost
    X_test = np.random.random_integers(1,1,50*400).reshape(400,50)
    X_test = modifyX(X_test)
    Y_test = generateYmod(X_test)
    valid = session.run(mvalid.softmax,feed_dict = {mvalid._input_data:X_test})
    
"""    
tmp = np.array([0.7,0.3,0.7,0.3,0.7,0.3]).reshape(3,2)
with tf.Graph().as_default(), tf.Session() as session:
    initializer = tf.random_uniform_initializer(-1,1)
    with tf.variable_scope("CRF", reuse=None, initializer=initializer):
        m = CRFCell(tmp.shape[1],tmp.shape[0],session)
        
    
    init_op = tf.initialize_all_variables()
    session.run(init_op)
    fwd = session.run(m(),feed_dict={m._obs: tmp})
"""    