import sys

sys.path.insert(0,'../.')

from lenskit import batch, topn, util
from lenskit import crossfold as xf
from lenskit.algorithms import Recommender, als, item_knn as knn

import pandas as pd

import matplotlib
import time


ratings = pd.read_csv('../ml-100k/u.data', sep='\t', 
                    names=['user','item','rating','timestamp'])

print(ratings.head())

#Define Algorithms
alg_li = knn.ItemItem(20)
alg_als = als.BiasedMF(50)

#Evaluation
def eval(aname, algo, train, test):
    fittable = util.clone(algo)
    fittable = Recommender.adapt(fittable)
    results = fittable.fit(train)

    return 
    #users = test.user.unique()
    #recs = batch.recommend(fittable,users,100)
    
    #recs['Algorithm'] = aname
    #return recs

all_recs = []
test_data = []
print(ratings.shape)
#for train,test in xf.partition_users(ratings[['user','item','rating']],1000,xf.SampleFrac(0.2)):
    
    #test_data.append(test)
    #print(train.shape)
    #break
    #all_recs.append(eval('ItemItem',alg_li,train,test))
    #all_recs.append(eval('ALS',alg_als,train,test))

#import csv 
#output_file = open('output.csv',mode='w')
#output_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

#for n in range(200,10000,100):

#for n in range(10000,100000,1000):
for n in [200]:
    print(n)
    for i in range(10):
        train = ratings[['user','item','rating']][:n]
        #print(train)
        #train = train.drop([141],axis=0)
        #print(n,i)
        eval('ItemItem',alg_li,train,train)
        #output_writer.writerow([n,native_learn, learn_unlearn, unlearn])

#all_recs = pd.concat(all_recs,ignore_index=True)
#print(all_recs.head())

#test_data = pd.concat(test_data,ignore_index=True)
#print(test_data.head())