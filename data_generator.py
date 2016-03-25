# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 12:02:52 2016

@author: edvinj
"""

import numpy as np

def generate_data(x_len,y_len):
    x = []
    y = np.random.random_integers(0,1,y_len)
    y = [one_hot(y_) for y_ in y]
    
    for y_ in y:
        if (y_ == [0,1]).all():
            x.append(np.random.normal(loc=1,scale=2,size=x_len))
        else:
            x.append(np.random.normal(loc=-1,scale=2,size=x_len))
            
    return np.array(y), np.array(x)
    
def one_hot(n):
    output = np.zeros(2)
    output[n] = 1    
    return output

def generate_set(ex):
    #X = None
    #Y = None
    for i in range(ex):
        tmp_y,tmp_x = generate_data(10,10)
        try:
            X = np.vstack((X,tmp_x))
            Y = np.vstack((Y,tmp_y))
        except Exception as e:
            print e
            X = tmp_x
            Y = tmp_y
    return np.split(X,ex), np.split(Y,ex)