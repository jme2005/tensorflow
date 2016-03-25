# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 09:54:19 2015
Another comment again:
script to create training and test data from CUDA json format
@author: edvinj
"""
import csv
import numpy as np
import utils
import json

save_dir = "data/"

w_emb = utils.read_vectors('/home/edvinj/trunk/vectorspadword.txt')
int_dict, rev_dict = utils.create_int_dict(w_emb)
with open('int_dict.json', 'wb') as fp:
    json.dump(int_dict,fp)

with open('rev_dict.json', 'wb') as fp:
    json.dump(rev_dict,fp)
       
print "read vectors"
x_train, y_train, x_test, y_test,tagdict = utils.create_windows(int_dict,'/home/edvinj/Data/secondtagging/newtrainingdata_withlines/',
                                                testpath = '/home/edvinj/Data/secondtagging/testsetALLlines/')
print "created the training and test sets"
"""
x_train_p, y_train_p = makepersontest(x_train,y_train)
x_test_p, y_test_p = makepersontest(x_test,y_test)
x_train_o, y_train_o = makeothertest(x_train,y_train)
x_test_o, y_test_o = makeothertest(x_test,y_test)
x_train_a,y_train_a = makeaddresstest(x_train,y_train)
x_test_a,y_test_a = makeaddresstest(x_test,y_test)
"""
savelist = [
           (x_train, "x_train_9_BI_num.csv"),
           (y_train,"y_train_9_BI_num.csv"),
           (x_test, "x_test_9_BI_num.csv"),
           (y_test, "y_test_9_BI_num.csv")
           ]

for obj,name in savelist:
    with open(save_dir+name,"w") as fp:
    	writer = csv.writer(fp)
    	writer.writerows(obj)

