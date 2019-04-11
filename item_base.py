#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  11 9:05:22 2019

@author: chenjie
"""
import  numpy as np
def load_item_data(filepath):
    with open("./data-new/itemAttribute.txt",'r')as f:
        lines=f.readlines()
        f.close()
    data_list=[]

    for i in range(len(lines)):
        line=lines[i].strip()
        itemid,x,y=line.split('|')
        if x=='None':
            x=0
        if y=='None':
            y=0
        data_list.append((int(x),int(y)))
    print('load data finished')
    return data_list

def pro_data(data_list):
    sum_x=0
    i_x=0
    sum_y=0
    i_y=0
    for rx,ry in data_list:
        if rx !=0:
            sum_x+=rx
            i_x+=1
        if ry!=0:
            sum_y+=ry
            i_y+=1
    sum_x/=i_x
    sum_y/=i_y
    # print(sum_x)
    # print(sum_y)

    pro_data=[]
    for rx,ry in data_list:
        if rx==0:
            rx=sum_x
        if ry==0:
            ry=sum_y
        pro_data.append((rx,ry))

    print('process success')
    return pro_data

def cal_sim(data_list):
    index=1
    length= len(data_list)
    sim=[]
    for x, y in data_list:
        v1 = np.zeros(2, np.double)
        v1[0],v1[1]=x,y
        for i in range(length-index):
            v2 = np.zeros(2, np.double)
            v2[0],v2[1]=data_list[i+index]
            sim.append(np.linalg.norm(v1 - v2))

        index+=1
    return sim

if __name__ == "__main__":
    dataset=load_item_data('./data-new/itemAttribute.txt')
    item_data=pro_data(dataset)
    sim=cal_sim(item_data)
    fid=open('item_sim.txt','w')
    for d in sim:
        print(d)
        fid.write(d)
        fid.write('\n')

    fid.close()
