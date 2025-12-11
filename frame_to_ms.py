# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 10:46:38 2025

@author: conrad
"""
fps = 30

t1 = '21.04'
t2 = '28.00'

# split into seconds and frames
t1s, t1f = map(int, t1.split('.'))
t2s, t2f = map(int, t2.split('.'))

# convert to seconds
t1_seconds = t1s + t1f / fps
t2_seconds = t2s + t2f / fps

latency = t2_seconds - t1_seconds
print(latency)