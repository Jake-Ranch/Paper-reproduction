import torch
import math
import numpy as np
import pygame
import torch.nn
import os
import pandas as pd
import random
import pickle
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt
import copy
from mpl_toolkits.mplot3d import Axes3D
from pylab import mpl
from scipy.stats import binom
from math import atan,sin,cos
import re
from IPython import display
import random




def match(text):
    matches = re.findall("'(.*?)'", text)
    mlist=[]
    for match in matches:
        mlist.append(match)
        # print(match)
    if mlist!=[]:
        return 1
    else:
        return 0

def file_exists(file_path):
    return os.path.exists(file_path)

def ball_dis(jingduA, weiduA, jingduB, weiduB):  # 输入AB点的经纬度，输出球面距离（输入是度数制，非弧度）
    a = (math.sin(Pi/180*(weiduA / 2 - weiduB / 2))) ** 2
    b = math.cos(weiduA * Pi / 180) * math.cos(weiduB * Pi / 180) * (
        math.sin((jingduA / 2 - jingduB / 2) * Pi / 180)) ** 2
    L = 2 * R * math.asin((a + b) ** 0.5)
    return L


def save_agent(agent, filename='demo.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(agent, f)

def load_agent(filename='demo.pkl'):
    with open(filename, 'rb') as f:
        agent = pickle.load(f)
    return agent



pre_load_RSU = []
random_gauss = []
RSUlocal = []
RSUloc_show = []
UAVloc_show = []
