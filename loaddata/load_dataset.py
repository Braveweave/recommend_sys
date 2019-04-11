#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 01:52:30 2019

@author: chenjie

This module descibes how to load a dataset from a  file.
fileid = 0 trains
       = 1 test
"""
import sys,os
sys.path.append(os.path.abspath(os.path.join('..')))
from collections import defaultdict
import Transet

def loaddata(filename,fileid):
    with open(filename,'r') as f:
        lines=f.readlines()
        f.close()
    data_list=[]

    index=0
    while index<len(lines):
        #get user id eg:'0|41'..
        line=lines[index].strip()
        index+=1
        if line=="":
            continue
        if '|' not in line:
            continue
        userid,n_rating=line.split('|')

        n=int(n_rating)

        #get score
        for i in range(n):
            line=lines[index+i].strip()
            if line=="":
                index+=1
                i-=1
                continue
            if fileid == '1':
                itemid=line
                data_list.append((userid,itemid))
                # if userid == '1':
                #     print(userid)
                # # print(userid)
            else:
                itemid,score=line.split('  ')
                data_list.append((userid,itemid,float(score)))

        index+=n
    print('load data finished')
    return data_list


# load data from file and construct train set
def construct_trainset(dataset):

    raw_users_id = dict()
    raw_items_id = dict()
    ur = defaultdict(list)
    ir = defaultdict(list)
    u_index = 0
    i_index = 0

    for ruid, riid, rating in dataset:
        # build user_id map-table(raw user id -> inner user id)
        try:
            uid = raw_users_id[ruid]
        except KeyError:
            uid = u_index
            raw_users_id[ruid] = uid
            u_index += 1

        # build item_id map-table(raw item id -> inner item id)
        try:
            iid = raw_items_id[riid]
        except KeyError:
            iid = i_index
            raw_items_id[riid] = iid
            i_index += 1

        ur[uid].append([iid, rating])
        ir[iid].append([uid, rating])
    n_users = len(ur)
    n_items = len(ir)
    n_ratings = len(dataset)
    rating_scale = (0, 100)
    trainset = Transet.Trainset(ur, ir, n_users, n_items, n_ratings, raw_users_id, raw_items_id, rating_scale)
    return trainset


if __name__ == "__main__":
    loaddata('../data-new/train.txt','0')