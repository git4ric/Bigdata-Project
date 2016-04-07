'''
Created on Mar 22, 2016

@author: Ripul
'''
import re
import numpy as np

class Cluster(object):
    
    def __init__(self, category,clusterID,documentCount):
        self.category = category
        self.clusterID = clusterID
        self.documentCount = documentCount
        
    def __repr__(self):
        string = ""      
        string += "{} {} {}".format(self.category, self.clusterID,self.documentCount)
        return string 

def loadData(fileName):
    file = open(fileName)
    clusterArr = []
    for line in file:
        matchObj = re.search(r"([.a-z]*),CompactBuffer\((.*)\)\)",line)      
        category = matchObj.group(1)
        dataStr = matchObj.group(2)       
        data = [int(x) for x in dataStr.split(",")[0:]]
        unique, counts = np.unique(np.array(data), return_counts=True)
        x = np.array(unique).tolist()
        y = np.array(counts).tolist()       
        index1 = y.index(max(y))
        index = x[index1]
        cluster = Cluster(category,index,max(y))
        clusterArr.append(cluster)
    return clusterArr

def main():
    clusters = loadData("b.txt")     
    clusters.sort(key= lambda x: x.clusterID, reverse=False)
    
    for elem in clusters:
        print(elem)
    
main()