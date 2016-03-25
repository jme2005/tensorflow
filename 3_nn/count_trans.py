# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 13:59:24 2015

@author: edvinj
"""
from collections import defaultdict
import csv
import json
import numpy as np
import sys

def counts(y_train,pad):
    tp = defaultdict(lambda: defaultdict(lambda: pad))
    old = str(y_train[0])
    for label in y_train[1:]:
        label = str(label)
        tp[old][label] += 1
        old = label
    for key, value in tp.items():
        
        for item in tp["['0.0', '0.0', '0.0', '1.0', '0.0']"].keys():
            if item not in value.keys():
                value[item]
                
    """    
    tot = 0    
    for item in tp.values():
        for count in item.values():
            tot += count
        for key in item.keys():
            item[key] = float(item[key])/tot
    """        
    return tp
    

def get_prob(tp,pad):
    name = {
            "['0.0', '0.0', '0.0', '1.0', '0.0']": "other",
            "['0.0', '0.0', '1.0', '0.0', '0.0']": "b-person",
            "['0.0', '0.0', '0.0', '0.0', '1.0']": "i-person",
            "['1.0', '0.0', '0.0', '0.0', '0.0']": "b-address",
            "['0.0', '1.0', '0.0', '0.0', '0.0']": "i-address"
            }
    out = {}
    for key, value in tp.items():
        tot = 0
        label = name[key]
        if label == "other":
            pad = pad#100000
        else:
            pad = 0
        newvalue = {}
        for count in value.values():
            tot += count
        
        for key in value.keys():
            keylabel = name[key]
            if keylabel[:2] == "b-":
                newvalue[keylabel] = float(value[key]+pad)/(tot+pad)
            else:
                newvalue[keylabel] = float(value[key])/(tot+pad)
        out[label] = newvalue
    return out
    

def create_matrix(tp):
    matrix = np.zeros([len(tp),len(tp)])
    for i,item1 in enumerate(tp.values()):
        for j, count in enumerate(item1.values()):
            matrix[i,j] = count
    
    return matrix
    


if __name__ == "__main__":
    
    print "Reading Data..."
    with open("y_train_9_BI.csv","r") as fp:
        reader = csv.reader(fp)
        y_train = list(reader)
    if len(sys.argv) > 1:
        pad = int(sys.argv[1])
    else:
        pad = 0
    print "done."
    print "do counts"
    tp = counts(y_train,1)
    tp = get_prob(tp,pad)
    print "write file"
    with open("tp_BI.json",'wb') as fp:
        json.dump(tp,fp)
