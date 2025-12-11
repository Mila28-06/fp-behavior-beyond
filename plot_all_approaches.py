# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 14:24:26 2025

@author: conrad
"""

import numpy as np
import pandas as pd
import scipy.io
import glob
import matplotlib.pyplot as plt
# from scipy.interpolate import interp1d
# from scipy.interpolate import CubicSpline
from scipy.interpolate import PchipInterpolator as pchip
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle

dlc_savePath = 'W:\\Conrad\\Innate_approach\\Data_analysis\\24.35.01\\DLC'
