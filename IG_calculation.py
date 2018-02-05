# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 00:36:26 2018

@author: Deepesh
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 01:10:29 2018

@author: Deepesh
"""

import csv
import math

def calculateEntropy(data,attr):
    count = {}
    
    for i in data: 
        if(i[attr] in count):
            count[i[attr]] = count[i[attr]]+1
        else:
            count[i[attr]]=1
    h=0.0   
    for j in count.values():
        h= h+ ((-1)*(j/len(data))* math.log2(j/len(data)))
    return h

def calculateIG(data,attr,classLabel):
    countsInSplit = {}
    entropyBefore = calculateEntropy(data,classLabel)
    entropyAfter = 0.0  
    for i in data:
        if(i[attr] in countsInSplit):
            countsInSplit[i[attr]] = countsInSplit[i[attr]]+1
        else:
            countsInSplit[i[attr]]=1
     
    for valOfAttr in countsInSplit:
        subdata = [r for r in data if(r[attr] == valOfAttr)]
        entropyAfter = entropyAfter + ((countsInSplit[valOfAttr]/sum(countsInSplit.values()))* calculateEntropy(subdata,classLabel))
    return entropyBefore-entropyAfter

def bestAttribute(data,header,classLabel):
    data = data[:]
    
    for testAttr in header:
        if(testAttr == header[len(header)-1]):
            continue
        maxIG =0.0
        attr = ""
        IG = calculateIG(data,testAttr,header[len(header)-1])
        if(IG>maxIG):
            maxIG=IG
            attr = testAttr
    return attr


#print("Max IG = {} for attribute = {}".format(maxIG,attr))

trainFile = "D:\\MyStudy\\UTD\\sem2\\ML\\Assignment\\assign 2\\training_set - Copy.csv"

f = open(trainFile,"r")
reader = csv.reader(f)
data = []
rownum=0

for row in reader:
    if(rownum == 0):
        header = row
        rownum = rownum+1
    else:
        if any(row):
            row = [int(i) for i in row]
            data.append(dict(zip(header,row)))