#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 16:05:22 2019

@author: chenjie
"""
import random
def process_data(filename):
    with open(filename,'r') as f:
        lines=f.readlines()
        f.close()
    data_list=[]
    fid = open('process_traindata.txt', 'w')
    fid2=open('process_validata.txt','w')
    index = 0
    random.seed(1)
    userid='0'
    while index < len(lines) and int(userid)<250:
        # get user id eg:'0|41'..
        line = lines[index].strip()
        index += 1
        if line == "" or '|' not in line:
            continue
        userid, n_rating = line.split('|')

        n = int(n_rating)

        # get score
        for i in range(n):
            line = lines[index + i].strip()
            if line == "":
                index += 1
                i -= 1
                continue


            itemid, score = line.split('  ')
            if int(itemid)>10000:
                continue

            str=userid+',' +itemid+','+score
            # itemid=line
            # str=userid+','+itemid

            if random.randint(0, 10) == 1:

                fid2.write(str)
                fid2.write('\n')
            else:

                fid.write(str)
                fid.write('\n')
            # data_list.append((userid, itemid))
            # data_list.append((userid, itemid, float(score)))

    fid2.close()
    fid.close()

if __name__ == "__main__":
    process_data('./data-new/train.txt')
    # process_data('./data-new/test.txt')

