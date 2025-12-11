# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 10:03:34 2025

call to remove top and right lines/ticks for cleaner looking grpah

@author: conrad
"""
import matplotlib.pyplot as plt

def boxoff():
    # Hide the top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # Set tick parameters to remove right and top ticks
    plt.gca().tick_params(axis='x', which='both', direction='out', bottom=True, top=False)
    plt.gca().tick_params(axis='y', which='both', direction='out', left=True, right=False)