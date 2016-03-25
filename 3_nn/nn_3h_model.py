# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 10:35:22 2015

@author: edvinj
"""

import math
import tensorflow.python.platform
import tensorflow as tf

VECTOR_SIZE = 500
WINDOW_SIZE = 9
INPUT_VEC = VECTOR_SIZE * WINDOW_SIZE
NUM_CLASSES = 7



def inference(x, h_1_u,h_2_u,h_3_u,h_4_u,h_5_u,keep_prob):
    
    with tf.name_scope("h_1") as scope:
        weights = tf.Variable(
            tf.truncated_normal([INPUT_VEC,h_1_u],
                                stddev=0.1),name='weights')
        biases = tf.Variable(tf.constant(0.1,shape=[h_1_u]),
                                name='biases')
        h_1 = tf.nn.tanh(tf.matmul(x,weights)+biases)
    
   
    with tf.name_scope("h_2") as scope:
        weights = tf.Variable(
            tf.truncated_normal([h_1_u,h_2_u],
                                stddev=0.1),name='weights')
        biases = tf.Variable(tf.constant(0.1,shape=[h_2_u]),
                             name='biases')
        h_2 = tf.nn.tanh(tf.matmul(h_1,weights)+biases)
       
    with tf.name_scope("h_3") as scope:
        weights = tf.Variable(
            tf.truncated_normal([h_2_u,h_3_u],
                                stddev=0.1),name='weights')
        biases = tf.Variable(tf.constant(0.1,shape=[h_3_u]),
                             name='biases')
        h_3 = tf.nn.tanh(tf.matmul(h_2,weights)+biases)
    
    with tf.name_scope("h_4") as scope:
        weights = tf.Variable(
            tf.truncated_normal([h_3_u,h_4_u],
                                stddev=0.1),name='weights')
        biases = tf.Variable(tf.constant(0.1,shape=[h_4_u]),
                             name='biases')
        h_4 = tf.nn.tanh(tf.matmul(h_3,weights)+biases)    
    
    with tf.name_scope("h_5") as scope:
        weights = tf.Variable(
            tf.truncated_normal([h_4_u,h_5_u],
                                stddev=0.1),name='weights')
        biases = tf.Variable(tf.constant(0.1,shape=[h_5_u]),
                             name='biases')
        h_5 = tf.nn.tanh(tf.matmul(h_4,weights)+biases)
        
    with tf.name_scope("D_O") as scope:
        h_5_d = tf.nn.dropout(h_5, keep_prob)
    
    with tf.name_scope("softmax") as scope:
        weights = tf.Variable(
            tf.truncated_normal([h_5_u,NUM_CLASSES],
                                stddev=0.1),name="weights")
        biases = tf.Variable(tf.constant(0.1,shape=[NUM_CLASSES]),
                             name="biases")
        logits = tf.nn.softmax(tf.matmul(h_5_d,weights) + biases)
   
    return logits
    
def loss(logits, y):
    """
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits,
                                                            y, 
    
    loss = tf.reduce_mean(cross_entropy, name ='x_entropy_mean')                                                        name="softmax_ce")
    """
    cross_entropy = -tf.reduce_sum(y*tf.log(tf.clip_by_value(logits,1e-8,1.0)
                                             ))
    
     
    return cross_entropy   
    
    
def training(loss,learning_rate):
    
    tf.scalar_summary(loss.op.name, loss)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    global_step = tf.Variable(0, name='global_step',trainable=False)
    train_op = optimizer.minimize(loss,global_step=global_step)
    return train_op
    
def evaluation(logits, labels):
    """
    correct_prediction = tf.equal(tf.argmax(logits,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
    return accuracy.eval(feed_dict = {logits:logits,y:y})
    """
    correct = tf.equal(tf.argmax(logits,1),tf.argmax(labels,1))
  # Return the number of true entries.
    return tf.reduce_sum(tf.cast(correct, tf.int32))
    
    
    
                                




        