#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 10:42:34 2019

@author: chenjie
"""

import numpy as np
from tqdm import tqdm
from time import sleep
import math
import matplotlib.pyplot as plt

class SVD(object):
    """
        SVD algorithm

        math:
            \hat{ r}_{ui} = \mu + b_u + b_i + q_i^Tp_u

        math:
            \sum_{r_{ui} \in R_{train}} \left(r_{ui} - \hat{r}_{ui} \\right)^2 +
            \lambda\\left(b_i^2 + b_u^2 + ||q_i||^2 + ||p_u||^2\\right)

        args:
            trainset: .
            n_factors: the number of factors, default is 5
            n_epochs: the number of iteration in SGD procedure, default is 20
            l_rate: learning rate, default is 0.005
            t_regular: ragularization term, default is 0.02
            init_mean: the mean of the normal distribution for factor vectors initialization, default is 0
            init_std_dev: the standard deviation of the normal distribution for factor vectors initialization, default is 0.1


        attributes:
            pu: the user factors
            qi: the item factors
            bu: the user biases
            bi: the item biases
    """

    def __init__(self, n_factors=9, n_epochs=10, l_rate=0.003, t_regular=0.015, init_mean=0.0, init_std_dev=0.05):
        self.trainset = None
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.l_rate = l_rate
        self.t_regular = t_regular
        self.init_mean = init_mean
        self.init_std_dev = init_std_dev
        self.bu = None
        self.bi = None
        self.pu = None
        self.qi = None

    def fit(self, trainset):
        self.trainset = trainset
        self.SGD(trainset)
        return


    def SGD(self,trainset):
        # initial pu qi bu bi
        global_mean = self.trainset.global_mean
        bu = np.zeros(self.trainset.n_users, np.double)
        bi = np.zeros(self.trainset.n_items, np.double)

        randstate = np.random.RandomState(0)
        pu = randstate.normal(self.init_mean, self.init_std_dev, (self.trainset.n_users, self.n_factors))
        qi = randstate.normal(self.init_mean, self.init_std_dev, (self.trainset.n_items, self.n_factors))

        lowbound, highbound = trainset.rating_scale


        plot_y = np.zeros(self.n_epochs)

        for cur_epoch in range(self.n_epochs):
            l_rate = self.l_rate / (2.1 ** cur_epoch - 0.1 * cur_epoch)
            print('%d round of fitting...' % (cur_epoch + 1))
            sleep(1)

            for uid, iid, rate in tqdm(self.trainset.get_all_ratings()):

                dot = 0
                for f in range(self.n_factors):
                    dot += qi[iid, f] * pu[uid, f]

                predict = global_mean + bu[uid] + bi[iid] + dot
                predict = max(predict, lowbound)
                predict = min(predict, highbound)
                err = rate - predict

                bu[uid] += l_rate * (err - self.t_regular * bu[uid])
                bi[iid] += l_rate * (err - self.t_regular * bi[iid])
                for f in range(self.n_factors):
                    puf = pu[uid, f]
                    qif = qi[iid, f]
                    pu[uid, f] += l_rate * (err * qif - self.t_regular * puf)
                    qi[iid, f] += l_rate * (err * puf - self.t_regular * qif)

            plot_y[cur_epoch] = err

        plt.plot(range(self.n_epochs), plot_y, marker='o', label='Training Data')
        plt.title('SGD-WR Learning Curve')
        plt.xlabel('Number of Epochs')
        plt.ylabel('error')
        plt.legend()
        plt.grid()
        plt.show()


        self.bu = bu
        self.bi = bi
        self.pu = pu
        self.qi = qi

    def baseline(self,trainset):
        self.trainset=trainset
        global_mean = self.trainset.global_mean
        bu = np.zeros(self.trainset.n_users, np.double)
        bi = np.zeros(self.trainset.n_items, np.double)

        randstate = np.random.RandomState(0)
        pu = randstate.normal(self.init_mean, self.init_std_dev, (self.trainset.n_users, self.n_factors))
        qi = randstate.normal(self.init_mean, self.init_std_dev, (self.trainset.n_items, self.n_factors))

        lowbound, highbound = trainset.rating_scale

        for cur_epoch in range(self.n_epochs):
            l_rate = self.l_rate / (2.1 ** cur_epoch - 0.1 * cur_epoch)
            print('%d round of fitting...' % (cur_epoch + 1))
            sleep(1)

            for uid, iid, rate in tqdm(self.trainset.get_all_ratings()):

                predict = global_mean + bu[uid] + bi[iid]
                predict = max(predict, lowbound)
                predict = min(predict, highbound)
                err = rate - predict

                bu[uid] += l_rate * (err - self.t_regular * bu[uid])
                bi[iid] += l_rate * (err - self.t_regular * bi[iid])

        self.bu = bu
        self.bi = bi



    def predict_one(self, ruid, riid):

        score = self.trainset.global_mean
        know_user = self.trainset.known_user(ruid)
        know_item = self.trainset.known_item(riid)

        if know_user:
            uid = self.trainset.get_inner_userid(ruid)
            score += self.bu[uid]

        if know_item:
            iid = self.trainset.get_inner_itemid(riid)
            score += self.bi[iid]

        if know_user and know_item:
            uid = self.trainset.get_inner_userid(ruid)
            iid = self.trainset.get_inner_itemid(riid)
            score += np.dot(self.qi[iid], self.pu[uid])

        lowbound, highbound = self.trainset.rating_scale

        score = max(lowbound, score)
        score = min(highbound, score)
        return score


    def predict_all(self, testset):

        print('begin predicting...')
        scores = []
        for ruid, riid in tqdm(testset):
            score = self.predict_one(ruid, riid)
            scores.append((ruid,riid,score))
        return scores


def RMSE(predictions,targets):
    """

    :param predictions:
    :param targets:
    :return:  RMSE=(r'-r)**2/n
    """

    assert  len(predictions)==len(targets)
    length=len(predictions)
    sum=0
    for i in range(length):
        sum+=(predictions[i]-targets[i])**2
    sum/=length
    return math.sqrt(sum)

def plotRMSE(n_epochs,train_errors):
    plt.plot(range(n_epochs), train_errors, marker='o', label='Training Data')
    plt.title('SGD-WR Learning Curve')
    plt.xlabel('Number of Epochs')
    plt.ylabel('error')
    plt.legend()
    plt.grid()
    plt.show()
