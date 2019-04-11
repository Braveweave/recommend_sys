#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 10:42:34 2019

@author: chenjie
"""

import numpy as np
from tqdm import tqdm
from time import sleep
import matplotlib as plt

class SVDpp(object):
    """The *SVD++* algorithm, an extension of :class:`SVD` taking into account
    implicit ratings.
    The prediction :math:`\\hat{r}_{ui}` is set as:
    .. math::
        \hat{r}_{ui} = \mu + b_u + b_i + q_i^T\\left(p_u +
        |I_u|^{-\\frac{1}{2}} \sum_{j \\in I_u}y_j\\right)
    Where the :math:`y_j` terms are a new set of item factors that capture
    implicit ratings.
    Args:
        n_factors: The number of factors. Default is ``20``.
        n_epochs: The number of iteration of the SGD procedure. Default is
            ``20``.
        init_mean: The mean of the normal distribution for factor vectors
            initialization. Default is ``0``.
        init_std_dev: The standard deviation of the normal distribution for
            factor vectors initialization. Default is ``0.1``.


    Attributes:
        pu(numpy array of size (n_users, n_factors)): The user factors (only
            exists if ``fit()`` has been called)
        qi(numpy array of size (n_items, n_factors)): The item factors (only
            exists if ``fit()`` has been called)
        yj(numpy array of size (n_items, n_factors)): The (implicit) item
            factors (only exists if ``fit()`` has been called)
        bu(numpy array of size (n_users)): The user biases (only
            exists if ``fit()`` has been called)
        bi(numpy array of size (n_items)): The item biases (only
            exists if ``fit()`` has been called)
    """
    def __init__(self, n_factors=5, n_epochs=10, l_rate=0.01,t_regular=0.005,init_mean=0, init_std_dev=0.05):
        self.trainset = None
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.l_rate = l_rate
        self.init_mean = init_mean
        self.t_regular = t_regular
        self.init_std_dev = init_std_dev
        self.bu = None
        self.bi = None
        self.pu = None
        self.qi = None
        self.yj = None

    def fit(self, trainset):
        # initialize matrix
        self.trainset = trainset
        self.SGD(trainset)
        return

    def SGD(self,trainset):
        global_mean = self.trainset.global_mean # 0
        bu = np.zeros(self.trainset.n_users, np.double)
        bi = np.zeros(self.trainset.n_items, np.double)

        #random samples from a normal Gaussian distribution
        randstate = np.random.RandomState(0)
        pu = randstate.normal(self.init_mean, self.init_std_dev, (trainset.n_users, self.n_factors))
        qi = randstate.normal(self.init_mean, self.init_std_dev, (trainset.n_items, self.n_factors))
        yj = randstate.normal(self.init_mean, self.init_std_dev, (trainset.n_items, self.n_factors))

        lowbound, highbound = trainset.rating_scale

        plot_y = np.zeros(self.n_epochs)
        for cur_epoch in range(self.n_epochs):
            l_rate = self.l_rate / (2.1 ** cur_epoch - 0.1 * cur_epoch)
            print('%d round of fitting...' % (cur_epoch + 1))
            sleep(1)
            for uid, iid, rate in tqdm(self.trainset.get_all_ratings()):

                ru = 1 / np.sqrt(len(trainset.ur[uid]))

                user_implicit = np.zeros(self.n_factors, np.double)
                for j, _ in trainset.ur[uid]:
                    for f in range(self.n_factors):
                        user_implicit[f] += yj[j, f] * ru

                dot = np.dot(qi[iid], user_implicit + pu[uid])
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
                    qi[iid, f] += l_rate * (err * (puf + user_implicit[f]) - self.t_regular * qif)

                    for j, _ in trainset.ur[uid]:
                        yj[j, f] += l_rate * (err * qif * ru - self.t_regular * yj[j, f])
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
        self.yj = yj
        return

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
            ru = 1 / len(self.trainset.ur[uid])
            user_implicit = np.zeros(self.n_factors, np.double)
            for j, _ in self.trainset.ur[uid]:
                for f in range(self.n_factors):
                    user_implicit[f] += self.yj[j, f] * ru
            score += np.dot(self.qi[iid], user_implicit + self.pu[uid])
        lowbound, highbound = self.trainset.rating_scale

        score = max(lowbound, score)
        score = min(highbound, score)
        return score

    def predict_all(self, testset):

        print('begin predicting...')
        scores = []
        for ruid, riid in tqdm(testset):
            score = self.predict_one(ruid, riid)
            scores.append(ruid,riid,score)
        return scores
