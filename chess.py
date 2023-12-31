from copy import deepcopy
class Chess:
    def __init__(self, EPD='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -'):
        self.columns = 'abcdefgh'
        self.rows = '87654321'
        self.piece_notation = {'p': 1, 'n': 2, 'b': 3, 'r': 4, 'q': 5, 'k': 6}
        self.piece_names = {1: 'Pawn', 2: 'Knight', 3: 'Bishop', 4: 'Rook', 5: 'Queen', 6: 'King'}
        self.reset(EPD)

    def reset(self, EPD='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -'):
        self.log = []
        self.init_pos = EPD
        self.EPD_table = {}
        self.current_player = 1
        self.castling = [1] * 4
        self.en_passant = None
        self.previous_move = None
        self.board = [[0] * 8 for _ in range(8)]
        self.load_EPD(EPD)

    def display(self):
        result = '  a b c d e f g h  \n  ----------------\n'
        for i, row in enumerate(self.board):
            result += f'{8 - i}|{" ".join(self.get_piece_notation(piece) for piece in row)}|{8 - i}\n'
        result += '  ----------------\n  a b c d e f g h\n'
        print(result)

    def get_piece_notation(self, piece):
        if piece == 0:
            return '.'
        piece_name = self.piece_names[abs(piece)]
        notation = getattr(Chess, piece_name)().notation
        return notation.lower() if piece < 0 else notation.upper() or 'p' if notation == '' else notation

    def board_2_array(self, coordinate):
        x, y = coordinate[0], coordinate[1]
        if x in self.columns and y in self.rows:
            return self.columns.index(x), self.rows.index(y)
        return None

    def EPD_hash(self):
        def piece_to_str(piece):
            if piece == 0:
                return ''
            notation = getattr(Chess, self.piece_names[abs(piece)])().notation
            return notation.lower() if piece < 0 else notation.upper() or 'p' if notation == '' else notation

        result = ''
        for row in self.board:
            empty_count = 0
            for square in row:
                if square == 0:
                    empty_count += 1
                else:
                    if empty_count > 0:
                        result += str(empty_count)
                        empty_count = 0
                    result += piece_to_str(square)
            if empty_count > 0:
                result += str(empty_count)
            result += '/' if result.count('/') < 7 else ''

        result += ' w ' if self.p_move == -1 else ' b '
        result += ''.join(letter.upper() if self.castling[i] == 1 else '' for i, letter in enumerate('KQkq'))
        result += f' -' if sum(self.castling) == 0 else f' {self.columns[self.en_passant[0]]}{self.rows[self.en_passant[1]]}' if self.en_passant else f' -'
        return result

    def load_EPD(self, EPD):
        data = EPD.split(' ')
        if len(data) == 4:
            for x, rank in enumerate(data[0].split('/')):
                y = 0
                for p in rank:
                    if p.isdigit():
                        self.board[x][y:y + int(p)] = [0] * int(p)
                        y += int(p)
                    else:
                        self.board[x][y] = self.piece_notation[p.lower()] * (-1) if p.islower() else self.piece_notation[p.lower()]
                        y += 1
            self.p_move = 1 if data[1] == 'w' else -1
            self.castling = [1 if letter in data[2] else 0 for letter in 'KQkq']
            self.en_passant = None if data[3] == '-' else self.board_2_array(data[3])
            return True
        return False

    def log_move(self, part, cur_cord, next_cord, cur_pos, next_pos, n_part=None):
        def get_piece_notation():
            if part == 6 * self.p_move and next_pos[0] - cur_pos[0] == 2:
                return '0-0' if self.p_move == 1 else '0-0-0'
            elif part == 1 * self.p_move and n_part:
                return f'{str(next_cord).lower()}={str(n_part).upper()}'
            else:
                p_name = self.piece_names[abs(part)]
                move_notation = getattr(Chess, p_name)().notation.upper()
                if self.board[next_pos[1]][next_pos[0]] != 0 or (next_pos == self.en_passant and (part == 1 or part == -1)):
                    move_notation += 'x' if move_notation else str(cur_cord)[0] + 'x'
                return move_notation + str(next_cord).lower()

        self.log.append(get_piece_notation())

    def move(self, cur_pos, next_pos):
        cp, np = self.board_2_array(cur_pos), self.board_2_array(next_pos)
        
        if self.valid_move(cp, np):
            part = self.board[cp[1]][cp[0]]
            
            if np == self.en_passant and abs(part) == 1:
                self.board[self.en_passant[1] - self.p_move][-self.en_passant[0]] = 0
            
            self.log_move(part, cur_pos, next_pos, cp, np)
            self.prev_move = deepcopy(self.board)
            
            if part in {1, -1} and np == self.en_passant:
                self.en_passant = (np[0], np[1] + 1) if part == 1 else (np[0], np[1] - 1)
            elif part == 6 * self.p_move and np[0] - cp[0] in {2, -2}:
                d = (np[0] - cp[0]) // 2
                self.board[np[1]][np[0] - d], self.board[np[1]][np[0] + d] = 4 * self.p_move, 0
            else:
                self.en_passant = None
            
            self.castling[::2] = [0] * 2 if part == 6 * self.p_move else self.castling[::2]
            index = (0 if cp[0] == 0 else 2) if part == 4 * self.p_move else (1 if cp[0] == 0 else 3)
            self.castling[index] = 0 if cp[0] == 0 else self.castling[index]
            
            self.board[cp[1]][cp[0]], self.board[np[1]][np[0]] = 0, part
            self.EPD_table[self.EPD_hash()] = self.EPD_table.get(self.EPD_hash(), 0) + 1
            return True
        
        return False

    def valid_move(self, cur_pos, next_pos):
        if cur_pos and next_pos:
            part = self.board[cur_pos[1]][cur_pos[0]]
            
            if part * self.p_move > 0 and part:
                p_name = self.parts[abs(part)]
                v_moves = getattr(Chess, p_name).movement(self, self.p_move, cur_pos, capture=True)
                
                if len(self.log) > 0 and '+' in self.log[-1]:
                    v_moves = [m for m in v_moves if cur_pos in self.c_escape and m in self.c_escape[cur_pos]]
                
                return next_pos in v_moves
        
        return False

    def possible_board_moves(self, capture=True):
        moves = {}
        for y, row in enumerate(self.board):
            for x, part in enumerate(row):
                if part:
                    p_name = self.parts[abs(part)]
                    p_colour = 1 if part > 0 else -1
                    v_moves = getattr(Chess, p_name).movement(self, p_colour, [x, y], capture=capture)
                    
                    if len(self.log) > 0 and '+' in self.log[-1]:
                        v_moves = [m for m in v_moves if (x, y) in self.c_escape and m in self.c_escape[(x, y)]]
                    
                    pos = f'{str(self.x[x]).upper() if p_colour > 0 else str(self.x[x]).lower()}{self.y[y]}'
                    moves[pos] = v_moves
            
        return moves

    def is_checkmate(self, moves):
        self.c_escape = {}
        k_pos, p_blocks, u_moves = (), [], {}

        for p, a in moves.items():
            pos = self.board_2_array(p)
            if (str(p[0]).isupper() and self.p_move == -1) or (str(p[0]).islower() and self.p_move == 1):
                if self.board[pos[1]][pos[0]] == self.King().value * (-self.p_move):
                    k_pos = (pos, a)
                else:
                    p_blocks.extend([(pos, m) for m in a if (pos, m) not in p_blocks])
            else:
                u_moves.setdefault(pos, []).extend(a)

        p_moves = [m for a in u_moves.values() for m in a]
        if not k_pos:
            for y, row in enumerate(self.board):
                if self.King().value * (-self.p_move) in row:
                    k_pos = ((row.index(self.King().value*(-self.p_move)), y), [])
                    break

        if k_pos and k_pos[0] not in p_moves:
            return [0, 0, 0]
        if k_pos and k_pos[0] in p_moves:
            for m in p_blocks + k_pos[1]:
                i_game = deepcopy(self)
                i_game.p_move *= -1
                i_game.move(f'{self.x[m[0][0]]}{self.y[m[0][1]]}', f'{self.x[m[1][0]]}{self.y[m[1][1]]}') 
                i_game.p_move *= -1
                i_moves = i_game.possible_board_moves(capture=True)

                if not any(k_pos[0] in i_moves[k] for k in i_moves):
                    self.c_escape.setdefault(m[0], []).append(m[1])

            if self.c_escape:
                self.log[-1] += '+'
                return [0, 0, 0]
            winner = 1 if self.p_move == 1 else 0
            self.log[-1] += '#' 
            return [winner, 0, 1 - winner]
        return [1, 0, 0] if self.p_move == 1 else [0, 0, 1]
    
    def pawn_promotion(self, n_part=None):
        valid_parts = {'q': 'Queen', 'b': 'Bishop', 'n': 'Knight', 'r': 'Rook'}
        
        if n_part is None:
            while True:
                n_part = input(
                    '\nPawn Promotion - What piece would you like to switch to:\n\n*Queen[q]\n*Bishop[b]\n*Knight[n]\n*Rook[r]\n'
                ).lower()
                if n_part not in valid_parts:
                    print('\nInvalid Option')
                else:
                    break

        n_part = valid_parts[n_part]

        pos = self.board_2_array(self.log[-1].replace('+', '').split('x')[-1])
        
        if pos is not None:
            part = self.notation[n_part.lower()] * self.p_move
            self.board[pos[1]][pos[0]] = part
            self.log[-1] += f'={n_part.upper()}'
            return True
    
        return False
    
    def fifty_move_rule(self, moves, choice=None):
        if len(self.log) <= 100:
            return False
        
        recent_moves = self.log[-100:]
        if any('x' in move or move[0].islower() for move in recent_moves):
            return False
        
        if choice is None:
            while True:
                choice = input('Fifty move rule - do you want to claim a draw? [Y/N]').lower()
                if choice in {'y', 'yes', '1'}:
                    return True
                elif choice in {'n', 'no', '0'}:
                    return False
                print('Unsupported answer')
        
        return choice in {'y', 'yes', '1'}
    
    def seventy_five_move_rule(self, moves):
        if len(self.log) > 150:
            for m in self.log[-150:]:
                if 'x' in m or m[0].islower():
                    return False
        else:
            return False
        return True
    
    def five_fold_rule(self, hash):
        if hash in self.EPD_table:
            if self.EPD_table[hash] >= 5:
                return True
        return False
    
    def three_fold_rule(self, hash):
        if hash in self.EPD_table:
            if self.EPD_table[hash] == 3:
                while True:
                    choice = input('Three fold rule - do you want to claim a draw? [Y/N]')
                    if choice.lower() == 'y' or choice.lower() == 'yes' or choice.lower() == '1':
                        return True
                    elif choice.lower() == 'n' or choice.lower() == 'no' or choice.lower() == '0':
                        return False
                    print('Unsupported answer')
        return False
    
    def is_dead_position(self, moves):
        pieces = [x for y in self.board for x in y if x != 0]
        unique_pieces = set(pieces)
        
        if len(unique_pieces) > 4:
            return False
        
        if len(unique_pieces) == 2 and {-6, 6} in {unique_pieces}:
            return True
        
        if len(unique_pieces) == 3 and {-6, 3, 6} in {unique_pieces, {-6, -3, 6}, {-6, 2, 6}, {-6, -2, 6}}:
            return True
        
        return False
    
    def is_stalemate(self, moves):
        if False not in [False for p, a in moves.items() if len(a) > 0 and ((self.p_move == 1 and str(p[0]).isupper()) or (self.p_move == -1 and str(p[0]).islower()))]:
            return True
        return False
    
    def is_draw(self, moves, hash):
        draw_conditions = [
            self.is_stalemate(moves),
            self.is_dead_position(moves),
            self.seventy_five_move_rule(moves),
            self.five_fold_rule(hash),
            self.fifty_move_rule(moves),
            self.three_fold_rule(hash)
        ]
        
        return any(condition for condition in draw_conditions)
    
    def is_end(self):
        w_king = any(self.board[y][x] == self.King().value for y, row in enumerate(self.board) for x, piece in enumerate(row))
        b_king = any(self.board[y][x] == self.King().value * (-1) for y, row in enumerate(self.board) for x, piece in enumerate(row))

        if not w_king and not b_king:
            return [0, 1, 0]
        elif not w_king:
            return [0, 0, 1]
        elif not b_king:
            return [1, 0, 0]

        moves = self.possible_board_moves(capture=True)
        check_mate = self.is_checkmate(moves)
        hash_key = self.EPD_hash()

        if sum(check_mate) > 0:
            return check_mate
        elif self.is_draw(moves, hash_key):
            return [0, 1, 0]

        return [0, 0, 0]
    
    def check_state(self, hash):
        if len(self.log) > 0:
            last_move = self.log[-1]
            if (self.p_move == 1 and ('8' in last_move and not last_move[0].isupper() or last_move[0] == 'P')) \
                or (self.p_move == -1 and ('1' in last_move and not last_move[0].isupper() or last_move[0] == 'P')):
                return 'PP'

        if hash in self.EPD_table and self.EPD_table[hash] == 3:
            return '3F' 
        elif len(self.log) > 100 and not any('x' in m or m[0].islower() for m in self.log[-100:]):
            return '50M'  
        return None
    
    class King:
        def __init__(self):
            self.value = 6 
            self.notation = 'K'
        def movement(game, player, pos, capture=True):
            result = []
            offsets = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]

            for offset in offsets:
                new_pos = (pos[0] + offset[0], pos[1] + offset[1])
                if 0 <= new_pos[0] <= 7 and 0 <= new_pos[1] <= 7:
                    piece_at_new_pos = game.board[new_pos[1]][new_pos[0]]
                    if piece_at_new_pos * player <= 0 or piece_at_new_pos == 0:
                        result.append(new_pos)

            if (pos == (4, 7) or pos == (4, 0)) and game.board[pos[1]][pos[0] + 1] == 0 and game.board[pos[1]][pos[0] + 2] == 0 and ((game.castling[0] == 1 and game.p_move == 1) or (game.castling[2] == 1 and game.p_move == -1)):
                result.append((pos[0] + 2, pos[1]))

            if (pos == (4, 7) or pos == (4, 0)) and game.board[pos[1]][pos[0] - 1] == 0 and game.board[pos[1]][pos[0] - 2] == 0 and ((game.castling[1] == 1 and game.p_move == 1) or (game.castling[3] == 1 and game.p_move == -1)):
                result.append((pos[0] - 2, pos[1]))

            return result
        
    class Queen:
        def __init__(self):
            self.value = 5
            self.notation = 'Q'
        def movement(game, player, pos, capture=True):
            result = []
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]
            
            for direction in directions:
                for c in range(1, 8):
                    x, y = pos[0] + direction[0] * c, pos[1] + direction[1] * c
                    
                    if 0 <= x <= 7 and 0 <= y <= 7:
                        piece = game.board[y][x]
                        
                        if piece * player < 0 or piece == 0:
                            result.append((x, y))
                            
                            if capture and piece * player < 0:
                                break
                        else:
                            break
                    else:
                        break
                        
            return result
        
    class Rook:
        def __init__(self):
                    self.value = 3
                    self.notation = 'B'
        def movement(game, player, pos, capture=True):
            result = []
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            
            for direction in directions:
                for i in range(1, 8):
                    new_x, new_y = pos[0] + direction[0] * i, pos[1] + direction[1] * i
                    
                    if not (0 <= new_x <= 7 and 0 <= new_y <= 7):
                        break
                    
                    piece = game.board[new_y][new_x]
                    if piece * player >= 0:
                        if piece != 0:
                            break
                        result.append((new_x, new_y))
                    else:
                        if capture:
                            result.append((new_x, new_y))
                        break
                    
            return result

    class Bishop:
        def __init__(self):
            self.value = 3
            self.notation = 'B' 
            def movement(game, player, pos, capture=True):
                result = []

                directions = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
                for dx, dy in directions:
                    for c in range(1, 8):
                        new_x, new_y = pos[0] + c * dx, pos[1] + c * dy
                        if 0 <= new_x <= 7 and 0 <= new_y <= 7:
                            if game.board[new_y][new_x] * player < 0 or game.board[new_y][new_x] == 0:
                                result.append((new_x, new_y))
                                if game.board[new_y][new_x] * player < 0 and capture:
                                    break
                            else:
                                break
                        else:
                            break
                return result
            
    class Knight:
        def __init__(self):
            self.value = 2
            self.notation = 'N'

        def movement(self, game, player, pos, capture=True):
            moves = [
                (pos[0] + 1, pos[1] + 2), (pos[0] + 2, pos[1] + 1),
                (pos[0] + 2, pos[1] - 1), (pos[0] + 1, pos[1] - 2),
                (pos[0] - 1, pos[1] - 2), (pos[0] - 2, pos[1] - 1),
                (pos[0] - 2, pos[1] + 1), (pos[0] - 1, pos[1] + 2)
            ]
            
            result = [
                move for move in moves if 0 <= move[0] <= 7 and 0 <= move[1] <= 7
                and (game.board[move[1]][move[0]] * player <= 0 or game.board[move[1]][move[0]] == 0)
            ]
            
            return result
    class Pawn:
        def __init__(self):
            self.value = 1
            self.notation = ''

        def movement(self, game, player, pos, capture=True):
            result = []
            init = 1 if player < 0 else 6
            amt = 1 if pos[1] != init else 2

            for i in range(amt):
                new_pos = (pos[0], pos[1] - ((i + 1) * player))
                if 0 <= new_pos[1] <= 7 and game.board[new_pos[1]][new_pos[0]] == 0:
                    result.append(new_pos)
                else:
                    break

            directions = [(1, -1), (-1, -1)]
            for direction in directions:
                new_pos = (pos[0] + direction[0], pos[1] - player)
                if 0 <= new_pos[0] <= 7 and 0 <= new_pos[1] <= 7:
                    if game.board[new_pos[1]][new_pos[0]] * player < 0 or new_pos == game.en_passant:
                        result.append(new_pos)

            return result