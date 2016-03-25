# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 10:29:19 2015

@author: edvinj
"""


import os.path
import time
import nn_3h_model
import tensorflow.python.platform
import numpy as np
from six.moves import xrange
import tensorflow as tf
import nltk
import json
import os
import sys
import re

VECTOR_SIZE = 500


NUM_CLASSES = 3
WORD_EMB = "/home/edvinj/trunk/vectorspadword.txt"

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
flags.DEFINE_string('input','No input string','The inputstring to be tagged')
flags.DEFINE_string("name","model.mod-1999","name of model")
flags.DEFINE_integer('window',5,'window size')
flags.DEFINE_boolean("viterbi",True,"when true uses viterbi algo")
flags.DEFINE_float('dropout',1.0,'keep probability not really useful for testing here')
WINDOW_SIZE = FLAGS.window
INPUT_VEC = VECTOR_SIZE * WINDOW_SIZE

def placeholder_inputs(batch_size):
    
    x = tf.placeholder(tf.float32, shape = [batch_size,INPUT_VEC])
    return x
    
def fill_feed_dict(x_data,x):
    #n = np.random.randint(0,len(xdata) - batch_size)
    x_feed = x_data
    feed_dict = {
        x: x_feed
        }
        
    return feed_dict
    

         
def run_inference(x_input,tokens,tp=None):
    
    with tf.Graph().as_default():
        
        x_in = placeholder_inputs(len(x_input))
        
        logits = nn_3h_model.inference(x_in,
                                       FLAGS.h_1_u,
                                       FLAGS.h_2_u,
                                       FLAGS.h_3_u,
                                       FLAGS.h_4_u,
                                       FLAGS.h_5_u,
                                       FLAGS.dropout)
               
        saver = tf.train.Saver()
        cn_dict = {2:"B-ADDRESS",3:"I-ADDRESS",1:"B-PERSON",0:"OTHER",4:"I-PERSON"}
        with tf.Session() as sess:
            init = tf.initialize_all_variables()
            sess.run(init)
            saver.restore(sess, "../train_dir/"+FLAGS.name)
            
            logits_eval = tf.Print(logits,[logits])
            output = logits_eval.eval(feed_dict = {x_in: x_input})
            
            for item in output:
                x = max(enumerate(item),key=lambda x: x[1])
                print x[0],round(x[1],3)
            """
            states = ('b-person','i-person','other','b-address','i-address')
            obs_dict = {"b-person":1,"other":0,"i-address":3,
                        "b-address":2,"i-person":4}
            prob,path = viterbi(states,tp,output,obs_dict)
            for tok,label in zip(tokens,path):
                print tok, label
            print prob
    
            for items in output:
                conf,class_id = max([(prob, n) for n, prob in enumerate(items)])
                print cn_dict[class_id], round(conf,3), [round(item,3) for item in items]
            """

def viterbi(states,trans_dict,observations,obs_dict):
    path = {}
    V=[{}]
    #print observations
    # base case        
    for n, y in enumerate(states):
        V[0][y] = 1./3*observations[0][obs_dict[y]]
        path[y] = [y]
    
    for n,obs in enumerate(observations[1:]):
        V.append({})
        newpath = {}
        for i, y in enumerate(states):
            #print [V[n][y0] for y0 in states]
            prob,state = max([(V[n][y0] * trans_dict[y0][y] * (obs[obs_dict[y]]), y0) for y0 in states])
            V[-1][y] = prob
            newpath[y] = path[state] + [y]
            
        path = newpath
    prob,state = max((V[-1][y], y) for y in states)
    return (prob,path[state])
    
    
def create_x_input(inputstring,dv):
    tokens = nltk.word_tokenize(inputstring)
    return create_ngram_string(dv,tokens,n_grams=WINDOW_SIZE),tokens
    
    
def create_ngram_string(dv,tokens,n_grams = 5):
    tokens = [removeDigits(tok.lower()) for tok in tokens]
    x = []
    for n,tok in enumerate(tokens):
        tmplist = []
        start = n - n_grams/2
        end = n + n_grams/2 + 1
        for i in range(start,end):
            if i < 0 or i > len(tokens):
                tmplist.append(dlookup('P_A_D',dv))
                
            else:
                try:
                    tmplist.append(dlookup(tokens[i],dv))
                except:
                    tmplist.append(dlookup('UNKNOWN',dv))
        x.append(np.append(tmplist[0],tmplist[1:]))
    return x
    
    
def removeDigits(tok):
    return re.sub('[0-9]','#',tok)
    
    
def read_vectors(path):
    dictionary = {}
    with open(path,'rb') as fp:
        lines = fp.readlines()
        
    for item in lines:
        try:
            dictionary[item.split()[0]] = np.array([float(n) for n in item.split()[1:-1]])
        except:
            print item
            
    for key, value in dictionary.items():
        if np.isnan(np.sum(value)) or len(value) != 500:
            del dictionary[key]
            
    sumarr = np.array([0.0]*500)
    for n,value in enumerate(dictionary.values()):
        try:
            sumarr += value
        except Exception as E:
            print E, n
    dictionary['UNKNOWN'] = sumarr/(float(len(dictionary)))
    return dictionary

def dlookup(key,dictionary):
    try:
        return dictionary[key]
    except:
        return dictionary['UNKNOWN']

                
                
def main(_):
    
    #inputstring = FLAGS.input
    w_emb = read_vectors(WORD_EMB)
    if FLAGS.viterbi == True:
        with open("tp_BI.json",'rb') as fp:
            tp = json.load(fp)
            print "tp loaded"
            print json.dumps(tp)
    EXIT = False
    while not EXIT:
        inputstring  = raw_input("Type input string or EXIT: ")
        if inputstring == "EXIT":
            EXIT = True
        elif inputstring == "":
            continue
        else:
            x_input,tokens = create_x_input(inputstring, w_emb)
            if FLAGS.viterbi:
                run_inference(x_input,tokens,tp)
            else:
                run_inference(x_input,tokens)
    
if __name__ == "__main__":
    tf.app.run()
                
    