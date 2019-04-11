#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 9:32:16 2019

@author: chenjie
"""

import numpy as np


class Trainset(object):

    """
        preprocess the trainset and calculate some useful data

        args & attributes:
            ratingList: userid itemid score
            ur: the users ratings
            ir: the items ratings
            n_users: total number of users
            n_items: total number of items
            n_ratings: total number of ratings
            global_mean: the mean of all ratings
            raw_users_id: raw user id -> inner user id
            raw_items_id: raw item id -> inner item id
            inner_users_id: inner user id -> raw user id
            inner_items_id: inner item id -> raw item id
    """

    def __init__(self, ur, ir, n_users, n_items, n_ratings, raw_users_id, raw_items_id, rating_scale):
        self.ur = ur
        self.ir = ir
        self.n_users = n_users
        self.n_items = n_items
        self.n_ratings = n_ratings
        self._global_mean = None
        self.raw_users_id = raw_users_id
        self.raw_items_id = raw_items_id
        self.inner_users_id = None
        self.inner_items_id = None
        self.rating_scale = rating_scale

    def get_all_ratings(self):
        for uid, u_rating in self.ur.items():
            for iid, rate in u_rating:
                yield uid, iid, rate

    def get_user_ratings(self, uid):
        for iid, rate in self.ur[uid]:
            yield iid, rate

    @property
    def global_mean(self):
        if self._global_mean is None:
            self._global_mean = np.mean([r for (_, _, r) in self.get_all_ratings()])
        return self._global_mean

    def get_raw_userid(self, userid):
        if self.inner_users_id is None:
            self.inner_users_id = { inner: raw for (raw, inner) in self.raw_items_id.items()}
        try:
            return self.inner_users_id[userid]
        except KeyError:
            raise Exception('Unknown inner user id!', userid)

    def get_raw_itemid(self, itemid):
        if self.inner_items_id is None:
            self.inner_items_id = { inner: raw for (raw, inner) in self.raw_items_id.items()}
        try:
            return self.inner_items_id[itemid]
        except KeyError:
            raise Exception('Unknown inner item id!', itemid)

    def get_inner_userid(self, userid):
        try:
            return self.raw_users_id[userid]
        except KeyError:
            raise Exception('Unknown raw user id!', userid)


    def get_inner_itemid(self, itemid):
        try:
            return self.raw_items_id[itemid]
        except KeyError:
            raise Exception('Unknown raw item id!', itemid)

    def known_user(self, userid):
        return userid in self.raw_users_id

    def known_item(self, itemid):
        return itemid in self.raw_items_id
