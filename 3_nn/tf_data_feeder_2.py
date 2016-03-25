# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 11:41:39 2015

@author: edvinj
"""

import os.path
import time
import csv
import nn_3h_model
import tensorflow.python.platform
import numpy as np
from six.moves import xrange
import tensorflow as tf
import utils

VECTOR_SIZE = 500
WINDOW_SIZE = 9
INPUT_VEC = VECTOR_SIZE * WINDOW_SIZE
NUM_CLASSES = 7


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float("learning_rate",0.0001,"initital learning rate")
flags.DEFINE_integer("max_steps",2000,"max iterations")
flags.DEFINE_integer('h_1_u',500,"number of units in hidden layer 1")
flags.DEFINE_integer('h_2_u',500,"number of units in hidden layer 2")
flags.DEFINE_integer('h_3_u',500,"number of units in hidden layer 3")
flags.DEFINE_integer('h_4_u',500,"number of units in hidden layer 4")
flags.DEFINE_integer('h_5_u',500,"number of units in hidden layer 5")
flags.DEFINE_integer('batch_size',1000,'Batch size')
flags.DEFINE_string('train_dir','../train_dir/','Directory of training data')
flags.DEFINE_string('name','model','Name of the model')
flags.DEFINE_float('dropout',1.0,'keep probability before readout')



def placeholder_inputs(batch_size):
    
    x = tf.placeholder(tf.float32, shape = [batch_size,INPUT_VEC])
    y = tf.placeholder(tf.float32, shape = [batch_size,NUM_CLASSES])
    return x, y
    
def fill_feed_dict(x_data,y_data, x, y):
    #n = np.random.randint(0,len(xdata) - batch_size)
    x_feed, y_feed  = x_data, y_data
    feed_dict = {
        x: x_feed,
        y: y_feed
        }
        
    return feed_dict
    
    
def generatesample(xdata, ydata, batch_size,rev_dict):
    """
    person = getindex(ydata,tuple([['0.0', '1.0', '0.0', '0.0', '0.0'],
                                  ['0.0', '0.0', '0.0', '0.0', '1.0']]))
                                  
    address =  getindex(ydata,tuple([['0.0', '0.0', '1.0', '0.0', '0.0'],
                                   ['0.0', '0.0', '0.0', '1.0', '0.0']]))                              
    
    """
    other,cuda_type = getindex(ydata,tuple([['0.0', '1.0', '0.0', '0.0', '0.0', '0.0',
                                             '0.0']]))
    try:
        cudaix = np.random.choice(cuda_type,size=int(batch_size*0.3)).tolist()
    except:
        cudaix = []
    try:
        otherix = np.random.choice(other,size=batch_size-len(cudaix),replace=False).tolist()
    except Exception as e:
        print e
        otherix = []
    xdataout = []
    ydataout = []
    indexlist = list(cudaix + otherix)
    for idx in indexlist:
        xdataout.append(transform(xdata[idx],rev_dict))
        ydataout.append(ydata[idx])
        
    return xdataout, ydataout

def transform(xdata,rev_dict):
    output = np.array([])
    for idx in xdata:
        output = np.append(output,rev_dict[int(idx)])
    return output  


def generatesampleeval(xdata, ydata, batch_size,classname,rev_dict):
    
    
    other,cuda_type = getindex(ydata,tuple([['0.0', '1.0', '0.0', '0.0', '0.0',
                               '0.0', '0.0']]))
    replace = True
    if classname == "other":
        setidx = np.random.choice(other,size=int(batch_size),replace=replace).tolist()
    else:
        setidx = np.random.choice(cuda_type,size=int(batch_size),replace=replace).tolist()
    xdataout = []
    ydataout = []
    for idx in setidx:
        xdataout.append(transform(xdata[idx],rev_dict))
        ydataout.append(ydata[idx])
        
    return xdataout, ydataout



    
def getindex(inlist,entry):
    indeces = []
    non_indeces = []
    for n,item in enumerate(inlist):
        if item in entry:
            indeces.append(n)
        else:
            non_indeces.append(n)
    return indeces,non_indeces


    
def do_eval(sess,
            eval_correct,
            x, y,
            x_data, y_data,
            datasize):
                
    true_count = 0
    steps_per_epoch = len(x_data) // datasize
    num_examples = steps_per_epoch * datasize
    for step in xrange(steps_per_epoch):
        feed_dict = fill_feed_dict(x_data,
                                   y_data,
                                   x,y)
    true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = float(true_count) / num_examples

    print('  Num examples: %d   Num correct: %d  Precision:  %0.04f \n' %
         (num_examples, true_count, precision))
         
def run_training(x_train, y_train,
                 x_test, y_test, rev_dict):
    
    with tf.Graph().as_default():
        x, y = placeholder_inputs(FLAGS.batch_size)
        #x_s, y_s = placeholder_inputs(1000)
        logits = nn_3h_model.inference(x,
                                       FLAGS.h_1_u,
                                       FLAGS.h_2_u,
                                       FLAGS.h_3_u,
                                       FLAGS.h_4_u,
                                       FLAGS.h_5_u,
                                       FLAGS.dropout)
               
        
        
        loss = nn_3h_model.loss(logits, y)
        
        train_op = nn_3h_model.training(loss, FLAGS.learning_rate)
        
        eval_correct = nn_3h_model.evaluation(logits, y)

        summary_op = tf.merge_all_summaries()
        
        saver = tf.train.Saver()
        
        sess = tf.Session() 
        
        init = tf.initialize_all_variables()
        
        sess.run(init)
        
        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir,
                                                graph_def=sess.graph_def)
        
        for step in xrange(FLAGS.max_steps):
            
            start_time = time.time()
            x_step, y_step = generatesample(x_train,y_train,FLAGS.batch_size,rev_dict)
            
            feed_dict = fill_feed_dict(x_step, y_step, x, y)
            
            _, loss_value = sess.run([train_op, loss],
                                     feed_dict=feed_dict)
            
            duration = time.time() - start_time

            if step % 100 == 0:
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
    
                summary_str = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
            
            if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                print('Training Data Eval:')
                do_eval(sess,
                    eval_correct,
                    x, y,
                    x_step,
                    y_step,
                    FLAGS.batch_size)
                    
                for classname in ["other","cuda"]:
                    print "Class: ", classname
                    x_test_s, y_test_s = generatesampleeval(x_test,y_test,
                                                            FLAGS.batch_size,
                                                            classname,rev_dict)
                    
                        
                    print('Test Data Eval:')
                    do_eval(sess,
                            eval_correct,
                            x, y,
                            x_test_s,
                            y_test_s,
                            FLAGS.batch_size)
                            
                saver.save(sess, FLAGS.train_dir+FLAGS.name, 
                           global_step = step)
                
                
def main(_):
    print "Loading data sets ..."
    with open("../data/x_train_9_BI_num.csv",'r') as fp:
        reader = csv.reader(fp)
        x_train = list(reader)       
    with open("../data/y_train_9_BI_num.csv","r") as fp:
        reader = csv.reader(fp)
        y_train = list(reader)
    with open("../data/x_test_9_BI_num.csv","r") as fp:
        reader = csv.reader(fp)
        x_test = list(reader)
    with open("../data/y_test_9_BI_num.csv","r") as fp:
        reader = csv.reader(fp)
        y_test = list(reader)
    w_emb = utils.read_vectors('../../trunk/vectorspadword.txt')
    int_dict, rev_dict = utils.create_int_dict(w_emb)
    del int_dict
    print "done.\n"
    print len(y_train[0])
    run_training(x_train, y_train, x_test, y_test,rev_dict)
    
if __name__ == "__main__":
    tf.app.run()
                
        
