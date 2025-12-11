# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 11:12:55 2025

@author: conrad
"""

import matplotlib.pyplot as plt
import numpy as np
    
def violinPlots(y1, y2, color1, color2, pointID, my_title, my_ylabel, yLimits, paired,
                my_yticks = None):

 
    fig, ax = plt.subplots(figsize=(10, 8))
    # idn = 1
    
    y = y1
    x = np.zeros(len(y)) + 1
    
    if y:
        # Violin plot for distribution
        parts = ax.violinplot(
                    y,
                    positions=[1],
                    showmeans=False,
                    showextrema=False,
                    showmedians=False
                )
    
        # Make the violin translucent
        for pc in parts['bodies']:
            pc.set_facecolor(color1)
            pc.set_alpha(0.3)
    
        # Scatter individual trials
        ax.scatter(
            x, y,
            color='black',
            alpha=0.5,
            zorder=3,
            label=pointID
        )
    
        # Mean and SEM
        mean_latency = np.mean(y)
        sem_latency = np.std(y, ddof=1) / np.sqrt(len(y))
    
        ax.errorbar(
            1, mean_latency,
            yerr=sem_latency,
            fmt='o',
            color='red',
            alpha = 0.5,
            capsize=5,
            markersize=8,
            label='Mean ± SEM',
            zorder=4
        )
    
    
    y = y2
    x = np.zeros(len(y)) + 2
    
    # Violin plot for distribution
    if y:
        parts = ax.violinplot(
                    y,
                    positions=[2],
                    showmeans=False,
                    showextrema=False,
                    showmedians=False
                )
    
        # Make the violin translucent
        for pc in parts['bodies']:
            pc.set_facecolor(color2)
            pc.set_alpha(0.3)
    
        # Scatter individual trials
        ax.scatter(
            x, y,
            color='black',
            alpha=0.5,
            zorder=3,
            label=pointID
        )
    
        # Mean and SEM
        mean_latency = np.mean(y)
        sem_latency = np.std(y, ddof=1) / np.sqrt(len(y))
    
        ax.errorbar(
            2, mean_latency,
            yerr=sem_latency,
            fmt='o',
            color='red',
            alpha = 0.5,
            capsize=5,
            markersize=8,
            label='Mean ± SEM',
            zorder=4
        )
    
    if paired:
        for a, b in zip(y1, y2):
            ax.plot([1, 2], [a, b], color='gray', alpha=0.5, zorder=2)
    
    ax.set_title(my_title)
    # axs[i].set_xlabel("Prey vs IR laser")
    ax.set_ylabel(my_ylabel)
    ax.set_xlim(0.5, 2.5)
    ax.set_ylim(yLimits[0], yLimits[1])  
    plt.xticks([1,2]),
    plt.yticks(my_yticks[0], my_yticks[1])
    # plt.yticks([0, 5, 10, 15, 20])
    # Hide the top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Set tick parameters to remove right and top ticks
    plt.gca().tick_params(axis='x', which='both', direction='out', bottom=True, top=False)
    plt.gca().tick_params(axis='y', which='both', direction='out', left=True, right=False)