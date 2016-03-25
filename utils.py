# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 13:14:55 2015

@author: edvinj
"""
import numpy as np
import json
import os
import re

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
    
def read_json(path):
    with open(path,'rb') as fp:
        return json.load(fp)

def extend_y(y):
    tlen = max([len(n) for n in y])
    y = [testlen(y_i,tlen) for y_i  in y]
    return y

def testlen(y_i,tlen):
    if len(y_i) == tlen:
        return y_i
    else:
        return np.append(y_i, ([0] * (tlen-len(y_i))))

       
def dlookup(key,dictionary):
    try:
        return dictionary[key]
    except:
        return dictionary['UNKNOWN']

def create_int_dict(dictionary):
    int_dict = {}
    rev_dict = {}
    for n,(key,value) in enumerate(dictionary.items()):
        #key = keyvalue[0]
        #value = keyvalue[1]
        int_dict[key] = n
        rev_dict[n] = value
    return int_dict, rev_dict

def exclude(tag):
    """
    Filters out the tags you want to include in your model
    """
    if tag == 'O':
        return tag
    elif tag[2:] == 'PERSON' or tag[2:] == 'CARDHOLDER':
        return tag[:2] + "PERSON"
    elif tag[2:] == 'RADDRESS':
        return tag[:2] + "STREETADD"
    elif tag[2:] in {'SSN','PHONE','ACCNUM'}:
        return tag[:2]
    else:
        return 'O'

def removeDigits(tok):
    """
    removes digits from data and replaces them with #
    """
    if tok.isalpha():
        return tok
    else:
        return re.sub('[0-9]','#',tok)
    
def create_ngram(tmpdir,dv,x,y,tagdict,n_grams = 5):
    
    for item in tmpdir['transcript']['sent_tokenized']:
        tokens = item['tokens']
        tokens = [removeDigits(tok.lower()) for tok in tokens]
        tags = [exclude(tag) for tag in item['tags']]
        
        tagset = set(tags)
        for item in tagset:
            m = len(tagdict)
            if item not in tagdict:
                tagdict[item] = m
                
        for n,tag in enumerate(tags):
            tmplist = []
            start = n - n_grams/2
            end = n + n_grams/2 + 1
            for i in range(start,end):
                if i < 0 or i > len(tags):
                    tmplist.append(dlookup('P_A_D',dv))
                else:
                    try:
                        tmplist.append(dlookup(tokens[i].lower(),dv))
                    except:
                        tmplist.append(dlookup('UNKNOWN',dv))
            x.append(np.append(tmplist[0],tmplist[1:]))
            
            
            
            
            y_array = np.zeros(len(tagdict))
            y_array[tagdict[tag]] = 1
            y.append(y_array)
    return x,y

def create_ngram_int(tmpdir,dv,x,y,tagdict,n_grams = 5):
    
    for item in tmpdir['transcript']['sent_tokenized']:
        tokens = item['tokens']
        tokens = [removeDigits(tok.lower()) for tok in tokens]
        tags = [exclude(tag) for tag in item['tags']]
        
        tagset = set(tags)
        for item in tagset:
            m = len(tagdict)
            if item not in tagdict:
                tagdict[item] = m
                
        for n,tag in enumerate(tags):
            tmplist = []
            start = n - n_grams/2
            end = n + n_grams/2 + 1
            for i in range(start,end):
                if i < 0 or i > len(tags):
                    tmplist.append(dlookup('P_A_D',dv))
                else:
                    try:
                        tmplist.append(dlookup(tokens[i].lower(),dv))
                    except:
                        tmplist.append(dlookup('UNKNOWN',dv))
            x.append(np.append(tmplist[0],tmplist[1:]))
            
            
            
            
            y_array = np.zeros(len(tagdict))
            y_array[tagdict[tag]] = 1
            y.append(y_array)
    return x,y  

def makepersontest(testdata_x,testdata_y):
    idx = findindex(testdata_y,np.array([1.,0,0.]))
    output_x = []
    output_y = []
    for n in idx:
        #print n
        output_x.append(testdata_x[n])
        output_y.append(testdata_y[n])
    return output_x,output_y

def makeothertest(testdata_x,testdata_y):
    idx = findindex(testdata_y,np.array([0.,1,0.]))
    output_x = []
    output_y = []
    for n in idx:
        #print n
        output_x.append(testdata_x[n])
        output_y.append(testdata_y[n])
    return output_x,output_y
    
def makeaddresstest(testdata_x,testdata_y):
    idx = findindex(testdata_y,np.array([0.,0,1.]))
    output_x = []
    output_y = []
    for n in idx:
        #print n
        output_x.append(testdata_x[n])
        output_y.append(testdata_y[n])
    return output_x,output_y   

def trainmodel(train_step,bf,n,x_train,y_train):    
    for i in range(n):
        print sess.run(bf)
        xdata,ydata = samplebatch(x_train,y_train,1000)
        train_step.run(feed_dict = {x: xdata, y_: ydata})
        print i

def samplebatch(x_train,y_train,m):
    n = np.random.randint(0,len(y_train)-m)
    return x_train[n:n+m], y_train[n:n+m]
    
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def findindex(array,condition):
    output = []
    for n,item in enumerate(array):
        if np.array_equal(item,condition):
            output.append(n)
    return output
    


def setupmodel():
    
    sess = tf.InteractiveSession()
    x = tf.placeholder("float", shape=[None, 2500])
    y_ = tf.placeholder("float", shape=[None, 2])
    W = tf.Variable(tf.zeros([2500,2]))
    b = tf.Variable(tf.zeros([2]))
    sess.run(tf.initialize_all_variables())
    y = tf.nn.softmax(tf.matmul(x*W)+b)
    cross_entropy =-tf.reduce_sum(y_*tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)  
    return sess,x,y_,W,b,y,cross_entropy,train_step
       
def create_windows(dv,path,testpath=None, window_size = 9):

    """
    Creates data of window_size-windows of the entire dataset.
    If testpath is included it also creates a test data set.
    The data needs to be in the format that we used for the cuda 
    crf model (json) as found in the /home/edvinj/Data/secondtagging/
    """

    filelist = [item for item in os.listdir(path) if item[-4:] == 'json']
    x = []
    y = []
    tagdict = {}
    for filename in filelist:
        try:
            tmpdir = read_json(path+filename)
            x,y = create_ngram_int(tmpdir,dv,x,y,tagdict,n_grams = window_size)
        except:
            continue
    y = extend_y(y)
    if testpath:
        filelist = [item for item in os.listdir(testpath) if item[-4:] == 'json']
        x_test = []
        y_test = []
    
        for filename in filelist:
            try:
                tmpdir = read_json(testpath+filename)        
                x_test,y_test = create_ngram(tmpdir,dv,x_test,y_test,tagdict, n_grams = 9)
            except:
                continue
        y_test = extend_y(y_test)
    if testpath:
        return x, y, x_test, y_test, tagdict
    else:
        return x, y, tagdict
    
