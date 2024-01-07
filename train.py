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

        while True:
            if str(white).lower() != 'ai' or str(black).lower() != 'ai':
                if chess_game.p_move == 1:
                    print('\nWhites Turn [UPPER CASE]\n')
                else:
                    print('\nBlacks Turn [LOWER CASE]\n')
                chess_game.display()
            if (chess_game.p_move == 1 and str(white).lower() != 'ai') or (chess_game.p_move == -1 and str(black).lower() != 'ai'):
                cur = input('What piece do you want to move?\n')
                next = input('Where do you want to move the piece to?\n')
            else:
                if chess_game.p_move == 1:
                    cur,next = w_bot.choose_action(chess_game)
                    w_log = pd.DataFrame(w_bot.log).drop_duplicates()
                    if len(w_log) > 0:
                        if 'imag_log' not in locals():
                            imag_log = pd.DataFrame(columns=list(w_log.columns.values))
                        for k,v in w_log.groupby(['value0','value1','value2']):
                            t_log = pd.DataFrame(log)
                            for i,x in enumerate(k):
                                t_log[f'value{i}'] = [x]*len(t_log)
                            t_log = t_log.append(v,ignore_index=True)
                            imag_log = imag_log.append(t_log,ignore_index=True)
                        imag_log = imag_log.drop_duplicates()
                else:
                    cur,next = b_bot.choose_action(chess_game)
                    b_log = pd.DataFrame(b_bot.log).drop_duplicates()
                    if len(b_log) > 0:
                        if 'imag_log' not in locals():
                            imag_log = pd.DataFrame(columns=list(b_log.columns.values))
                        for k,v in b_log.groupby(['value0','value1','value2']):
                            t_log = pd.DataFrame(log)
                            for i,x in enumerate(k):
                                t_log[f'value{i}'] = [x]*len(t_log)
                            t_log = t_log.append(v,ignore_index=True)
                            imag_log = imag_log.append(t_log,ignore_index=True)
                        imag_log = imag_log.drop_duplicates()
                print(f'w {cur.lower()}-->{next.lower()} | EPOCH:{epoch} BOARD:{game_name} MOVE:{len(log)} HASH:{chess_game.EPD_hash()}\n') if chess_game.p_move > 0 else print(f'b {cur.lower()}-->{next.lower()} | EPOCH:{epoch} BOARD:{game_name} MOVE:{len(log)} HASH:{chess_game.EPD_hash()}\n')
            enc_game = plumbing.encode_state(chess_game)
            valid = False
            if chess_game.move(cur, next) == False:
                print('Invalid move')
            else:
                valid = True
                cur_pos = chess_game.board_2_array(cur)
                next_pos = chess_game.board_2_array(next)
                log.append({**{f'state{i}':float(s) for i, s in enumerate(enc_game[0])},
                            **{f'action{x}':1 if x == ((cur_pos[0]+(cur_pos[1]*8))*64)+(next_pos[0]+(next_pos[1]*8)) else 0 for x in range(4096)}})
            if (str(white).lower() == 'ai' and chess_game.p_move == 1) or (str(black).lower() == 'ai' and chess_game.p_move == -1):
                state = chess_game.check_state(chess_game.EPD_hash())
                if state == '50M' or state == '3F':
                    state = [0, 1, 0] #Auto tie
                elif state == 'PP':
                    chess_game.pawn_promotion(n_part='Q') #Auto queen
                if state != [0, 1, 0]:
                    state = chess_game.is_end()
            else:
                state = chess_game.is_end()
                if state == [0, 0, 0]:
                    if chess_game.check_state(chess_game.EPD_hash()) == 'PP':
                        chess_game.pawn_promotion()
            if sum(state) > 0:
                print(f'FINISHED | EPOCH:{epoch} BOARD:{game_name} MOVE:{len(log)} STATE:{state}\n')
                game_train_data = pd.DataFrame(log)
                for i, x in enumerate(state):
                    game_train_data[f'value{i}'] = [x]*len(log)
                    game_train_data[f'value{i}'] = game_train_data[f'value{i}'].astype(float)
                if 'imag_log' in locals():
                    game_train_data = game_train_data.append(imag_log,ignore_index=True)
                game_train_data = game_train_data.astype(float)
                break
            if valid == True:
                chess_game.p_move = chess_game.p_move * (-1)
        return state, game_train_data, a_colour
