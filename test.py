#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 23:52:30 2019

@author: chenjie
"""
from loaddata import load_dataset
import random
import SVD
import SVDPP
import numpy as np
import  math
import matplotlib.pyplot as plt
from SVD import RMSE

# load_dataset.loaddata('./data-new/test.txt','1')
# load_dataset.loaddata('./data-new/train.txt','0')
def split_train_and_test(dataset,n_split,seed,index):
    """
    generate the function to iterate over trainsets and testsets.
    :param dataset: 训练集和验证集data
    :param n_split: 将数据分成n_split 份
    :param seed: 随机种子
    :param index: 选取第index份数据作为测试数据
    :return: trainset and testset
    """
    random.seed(seed)
    transet=[]
    testset=[]
    for i in range(len(dataset)):
        if random.randint(0,n_split)==index:
            testset.append(dataset[i])
        else:
            transet.append(dataset[i])
    return transet,testset


# def RMSE(predictions,targets):
#     """
#
#     :param predictions:
#     :param targets:
#     :return:  RMSE=(r'-r)**2/n
#     """
#
#     assert  len(predictions)==len(targets)
#     length=len(predictions)
#     sum=0
#     for i in range(length):
#         sum+=(predictions[i]-targets[i])**2
#     sum/=length
#     return math.sqrt(sum)


def testSVD():
    dataset=load_dataset.loaddata('./data-new/train.txt','0')
    trainset,testset=split_train_and_test(dataset,10,1,1)
    # testset = load_dataset.loaddata('./data-new/test.txt','1')
    trainset=load_dataset.construct_trainset(dataset)
    svd=SVD.SVD()
    svd.fit(trainset)

    targets=[score for (_,_,score)in testset]
    items=[[userid,itemid]for (userid,itemid,score) in testset]
    # items = [[userid,itemid]for (userid,itemid) in testset]
    predictions=svd.predict_all(items)
    p=[score for (_,_,score)in predictions]


    # fid=open('result1.txt','w')
    #
    # index=0
    # for ruid,riid,score in predictions:
    #     fid.write(ruid+','+riid+',')
    #     fid.write(str(score))
    #     fid.write(',')
    #     fid.write(str(targets[index]))
    #     index+=1
    #     fid.write('\n')
    #
    #
    # fid.close()
    score = RMSE(p, targets)
    print(score)

"""
    predict and return result.txt

"""
    # fid=open('result.txt','w')
    # for ruid,riid,score in predictions:
    #     s=ruid+','+riid+','
    #     fid.write(s)
    #     fid.write(str(score))
    #     fid.write('\n')
    # fid.close()

    # items=[[userid,itemid] for (userid,itemid,score)in testset]
    # predictions=svd.predict_all(items)

    # score=RMSE(predictions,targets)

    # print(score)


def testSVDpp():
    dataset=load_dataset.loaddata('./data-new/train.txt','0')
    trainset,testset=split_train_and_test(dataset,10,1,1)
    # testset = load_dataset.loaddata('./data-new/test.txt','1')
    trainset=load_dataset.construct_trainset(dataset)
    svd=SVDPP.SVDpp()
    svd.fit(trainset)

    targets=[score for (_,_,score)in testset]
    items=[[userid,itemid]for (userid,itemid,score) in testset]
    # items = [[userid,itemid]for (userid,itemid) in testset]
    predictions=svd.predict_all(items)
    p=[score for (_,_,score)in predictions]


    fid=open('result2.txt','w')

    index=0
    for ruid,riid,score in predictions:
        fid.write(ruid+','+riid+',')
        fid.write(str(score))
        fid.write(',')
        fid.write(str(targets[index]))
        index+=1
        fid.write('\n')


    fid.close()
    score = RMSE(p, targets)
    print(score)

if __name__=='__main__':
    # testSVD()
    testSVDpp()
