import os
import math
import json
import numpy
import torch
import random
import pandas as pd
from copy import deepcopy
from model.model import TransformerModel
from concurrent.futures import ProcessPoolExecutor, as_completed

class Agent:
    def __init__(self, max_depth=5, search_amount=50, train=False, model='model-active.pth.tar'):
        self.log = []
        self.train = train
        self.search_amount = search_amount
        self.MCTS = deepcopy(MCTS(max_depth=max_depth, train=train, filename=model))

    def choose_action(self, game):
        self.log = []
        self.MCTS.Player = game.p_move
        for n in self.MCTS.tree:
            self.MCTS.tree[n].max_depth = False
        parent_hash = game.EPD_hash()
        for x in range(self.search_amount):
            self.MCTS.depth = 0
            self.MCTS.search(game)
            if self.train == True:
                if sum(self.MCTS.state) > 0:
                    game_train_data = pd.DataFrame(self.MCTS.log)
                    for i, x in enumerate(self.MCTS.state):
                        game_train_data[f'value{i}'] = [x] * len(self.MCTS.log)
                        game_train_data[f'value{i}'] = game_train_data[f'value{i}'].astype(float)
                    game_train_data = game_train_data.to_dict('records')
                    if len(game_train_data) > 0:
                        self.log += game_train_data
                    self.MCTS.state = [0, 0, 0]
                self.MCTS.log = []
        u_bank = {}
        for c, moves in game.possible_board_moves(capture=True).items():
            if len(moves) > 0 and ((c[0].isupper() and game.p_move == 1) or (c[0].islower() and game.p_move == -1)):
                for n in moves:
                    imag_game = deepcopy(game)
                    if imag_game.move(c, f'{game.x[n[0]]}{game.y[n[1]]}') == True:
                        imag_game.p_move = imag_game.p_move * (-1)
                        hash = imag_game.EPD_hash()
                        if hash in self.MCTS.tree:
                            if self.MCTS.tree[hash].leaf == True and self.MCTS.tree[hash].Q == 6:
                                return c, f'{game.x[n[0]]}{game.y[n[1]]}'
                            else:
                                u_bank[f'{c}-{game.x[n[0]]}{game.y[n[1]]}'] = self.MCTS.tree[hash].Q + (
                                        self.MCTS.Cpuct * self.MCTS.tree[hash].P * (
                                            math.sqrt(math.log(self.MCTS.tree[parent_hash].N) / (
                                                    1 + self.MCTS.tree[hash].N))))
        m_bank = [k for k, v in u_bank.items() if v == max(u_bank.values())]
        if len(m_bank) > 0:
            cur, next = random.choice(m_bank).split('-')
        else:
            cur, next = ''
        return cur, next


class MCTS:
    def __init__(self, max_depth=5, train=False, folder='ai_ben/data', filename='model-active.pth.tar'):
        self.train = train
        self.tree = {}
        self.Cpuct = 0.77
        self.Player = None
        self.depth = 0
        self.max_depth = max_depth
        self.log = []
        self.state = [0, 0, 0]
        self.plumbing = Plumbing()

        with open(os.path.join(folder, 'model_param.json')) as f:
            m_param = json.load(f)

        self.Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Model = TransformerModel(
            m_param['input_size'],
            m_param['ntokens'],
            m_param['emsize'],
            m_param['nhead'],
            m_param['nhid'],
            m_param['nlayers'],
            m_param['dropout']
        ).to(self.Device)

        filepath = os.path.join(folder, filename)
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.Device)
            self.Model.load_state_dict(checkpoint['state_dict'])

    class Node:
        def __init__(self):
            self.Q = 0
            self.P = 0
            self.N = 0
            self.leaf = False
            self.max_depth = False

    def search(self, game):
        self.depth += 1
        parent_hash = game.EPD_hash()
        if parent_hash not in self.tree:
            self.tree[parent_hash] = self.Node()
        self.tree[parent_hash].N += 1
        state = game.check_state(parent_hash)
        if state == '50M':
            state = [0, 1, 0]
        elif state == '3F':
            state = [0, 1, 0]
        elif state == 'PP':
            game.pawn_promotion(n_part='Q')
        if state != [0, 1, 0]:
            state = game.is_end()
        if sum(state) > 0:
            if self.train == True:
                self.state = state
            self.tree[parent_hash].leaf = True
            if (state == [1, 0, 0] and self.Player == 1) or (state == [0, 0, 1] and self.Player == -1):
                self.tree[parent_hash].Q = 6
            elif state == [0, 1, 0]:
                self.tree[parent_hash].Q = 1
            else:
                self.tree[parent_hash].Q = -6
            return self.tree[parent_hash].Q, self.tree[parent_hash].P
        if self.tree[parent_hash].Q == 0:
            enc_state = self.plumbing.encode_state(game)
            v, p = self.Model(enc_state)
            state[torch.argmax(v).item()] = 1
            if (state == [1, 0, 0] and self.Player == 1) or (state == [0, 0, 1] and self.Player == -1):
                self.tree[parent_hash].Q = 3
            elif state == [0, 1, 0]:
                self.tree[parent_hash].Q = 1
            else:
                self.tree[parent_hash].Q = -3
            p = p.reshape(64, 8, 8)
            for cur, moves in game.possible_board_moves(capture=True).items():
                if len(moves) > 0 and ((cur[0].isupper() and game.p_move == 1) or (cur[0].islower() and game.p_move == -1)):
                    for next in moves:
                        imag_game = deepcopy(game)
                        if imag_game.move(cur, f'{game.x[next[0]]}{game.y[next[1]]}') == True:
                            imag_game.p_move = imag_game.p_move * (-1)
                            hash = imag_game.EPD_hash()
                            if hash not in self.tree:
                                self.tree[hash] = self.Node()
                            cur_pos = game.board_2_array(cur)
                            self.tree[hash].P = p[cur_pos[0] + (cur_pos[1] * 8)][next[1]][next[0]].item()
            return self.tree[parent_hash].Q, self.tree[parent_hash].P
        else:
            if self.depth == self.max_depth:
                return self.tree[parent_hash].Q, self.tree[parent_hash].P
            b_cur = None
            b_next = None
            b_action = None
            w_check = False
            b_upper = float('-inf')
            for cur, moves in game.possible_board_moves(capture=True).items():
                if len(moves) > 0 and ((cur[0].isupper() and game.p_move == 1) or (cur[0].islower() and game.p_move == -1)):
                    for next in moves:
                        imag_game = deepcopy(game)
                        if imag_game.move(cur, f'{game.x[next[0]]}{game.y[next[1]]}') == True:
                            state = imag_game.check_state(imag_game.EPD_hash())
                            if state == '50M' or state == '3F':
                                state = [0, 1, 0]
                            elif state == 'PP':
                                imag_game.pawn_promotion(n_part='Q')
                            if state != [0, 1, 0]:
                                state = imag_game.is_end()
                            if (state == [1, 0, 0] and imag_game.p_move == 1) or (state == [0, 0, 1] and imag_game.p_move == -1):
                                imag_game.p_move = imag_game.p_move * (-1)
                                b_action = deepcopy(imag_game)
                                b_cur = deepcopy(game.board_2_array(cur))
                                b_next = deepcopy(next)
                                w_check = True
                                break
                            imag_game.p_move = imag_game.p_move * (-1)
                            hash = imag_game.EPD_hash()
                            if hash in self.tree and self.tree[hash].max_depth == False:
                                u = self.tree[hash].Q + (self.Cpuct * self.tree[hash].P * (
                                            math.sqrt(math.log(self.tree[parent_hash].N) / (1 + self.tree[hash].N))))
                                if u > b_upper:
                                    b_action = deepcopy(imag_game)
                                    b_cur = deepcopy(game.board_2_array(cur))
                                    b_next = deepcopy(next)
                                    b_upper = u
                    if w_check == True:
                        break
            if b_action != None and b_cur != None and b_next != None:
                if self.train == True:
                    self.log.append({**{f'state{i}': float(s) for i, s in enumerate(self.plumbing.encode_state(b_action)[0])},
                                     **{f'action{x}': 1 if x == ((b_cur[0] + (b_cur[1] * 8)) * 64) + (b_next[0] + (b_next[1] * 8)) else 0 for x in range(4096)}})
                v, p = self.search(b_action)
                hash = b_action.EPD_hash()
                if hash in self.tree:
                    if self.depth == self.max_depth:
                        self.tree[hash].max_depth = True
                    self.tree[hash].Q = v
                    self.tree[hash].P = p
                    return self.tree[hash].Q, self.tree[hash].P
            return self.tree[parent_hash].Q, self.tree[parent_hash].P