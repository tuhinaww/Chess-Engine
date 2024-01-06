import math
import random
from copy import deepcopy
import torch
from numpy import isnan
from numpy.random import dirichlet
from einops import rearrange

from tools.toolbox import ToolBox

class MCTS:
    def __init__(
        self,
        backbone,
        value,
        policy,
        state,
        reward,
        Cca = None,
        action_space = 4096,
        c1 = 1.25,
        c2 = 19652,
        d_a = .3,
        e_f = .25,
        g_d = 1.,
        single_player = False,
        max_depth = float('inf')
    ):
        self.tree = {} 
        #self.tree = Manager().dict()
        self.l = 0 
        self.action_space = action_space 
        self.max_depth = max_depth 
        self.c1 = c1 
        self.c2 = c2 
        self.d_a = d_a 
        self.e_f = e_f 
        self.g_d = g_d 
        self.Q_max = 1 
        self.Q_min = -1 
        self.g = backbone 
        self.v = value 
        self.p = policy 
        self.s = state 
        self.r = reward 
        self.Cca = Cca 
        self.single_player = single_player 
