# -*- coding: utf-8 -*-
from __future__ import absolute_import
import logging

import os
import random
from abc import ABC
from collections import defaultdict

from torch.utils.data import Sampler


class StratifiedSampler(Sampler, ABC):
    def __init__(self, labels):
        self.idx_by_lb = defaultdict(list)
        for idx, lb in enumerate(labels):
            self.idx_by_lb[lb].append(idx)

        self.size = len(labels)

    def __len__(self):
        return self.size

    def __iter__(self):
        while True:
            songs_list = []
            artists_list = []
            for lb, v in self.idx_by_lb.items():
                for idx in v:
                    songs_list.append(idx)
                    artists_list.append(lb)

            shuffled = spotifyShuffle(songs_list, artists_list)
            for idx in shuffled:
                yield idx


def fisherYatesShuffle(arr):

    for i in range(len(arr)-1, 0, -1):
        j = random.randint(0, i)
        arr[i], arr[j] = arr[j], arr[i]
    return arr


def spotifyShuffle(songs_list, artists_list):
    artist2songs = defaultdict(list)
    for artist, song in zip(artists_list, songs_list):
        artist2songs[artist].append(song)
    songList = []
    songsLocs = []
    for artist, songs in artist2songs.items():
        songs = fisherYatesShuffle(songs)
        songList += songs
        songsLocs += get_locs(len(songs))
    return [songList[idx] for idx in argsort(songsLocs)]


def argsort(seq):
    return [i for i, j in sorted(enumerate(seq), key=lambda x:x[1])]


def get_locs(n):
    percent = 1. / n
    locs = [percent * random.random()]
    last = locs[0]
    for i in range(n - 1):
        value = last + percent * random.uniform(0.8, 1.2)  #
        locs.append(value)
        last = value
    return locs
