import os
import time
import math
import json
import torch
import random
import pandas as pd
from copy import deepcopy
from shutil import copyfile
import sys
sys.path.insert(0, os.getcwd())
from chess import Chess
from mcts import Agent, Plumbing
from model import TransformerModel

class train:
    def play_game(game_name, epoch, train=False, white='ai', black='ai', active_model='model-active.pth.tar', new_model='model-new.pth.tar', search_amount=50, max_depth=5, best_of=5):
        if str(white).lower() == 'ai' and str(black).lower() == 'ai':
            if (epoch+1) % best_of == 0:
                a_colour = random.choice(['w', 'b'])
            elif (epoch+1) % 2 == 0:
                a_colour = 'b'
            else:
                a_colour = 'w'
        elif str(white).lower() != 'ai' and str(black).lower() == 'ai':
            a_colour = 'b'
        elif str(white).lower() == 'ai' and str(black).lower() != 'ai':
            a_colour = 'w'
        else:
            a_colour = None
        if a_colour == 'w' and str(white).lower() == 'ai' and str(black).lower() == 'ai':
            w_bot = deepcopy(Agent(search_amount=search_amount, max_depth=max_depth, train=train, model=active_model))
            b_bot = deepcopy(Agent(search_amount=search_amount, max_depth=max_depth, train=train, model=new_model))
        elif a_colour == 'b' and str(white).lower() == 'ai' and str(black).lower() == 'ai':
            w_bot = deepcopy(Agent(search_amount=search_amount, max_depth=max_depth, train=train, model=new_model))
            b_bot = deepcopy(Agent(search_amount=search_amount, max_depth=max_depth, train=train, model=active_model))
        elif a_colour == 'w' and str(white).lower() == 'ai' and str(black).lower() != 'ai':
            w_bot = deepcopy(Agent(search_amount=search_amount, max_depth=max_depth, train=train, model=new_model))
            b_bot = None
        elif a_colour == 'b' and str(white).lower() != 'ai' and str(black).lower() == 'ai':
            w_bot = None
            b_bot = deepcopy(Agent(search_amount=search_amount, max_depth=max_depth, train=train, model=new_model))
        else:
            w_bot = None
            b_bot = None
        log = []
        plumbing = Plumbing()
        chess_game = deepcopy(Chess()) #'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -'