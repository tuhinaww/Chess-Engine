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
                self.board[np[1]][np[0] - d] = 4 * self.p_move
                self.board[np[1]][np[0] + d] = 0
            else:
                self.en_passant = None
            
            self.castling[::2] = [0] * 2 if part == 6 * self.p_move else self.castling[::2]
            index = (0 if cp[0] == 0 else 2) if part == 4 * self.p_move else (1 if cp[0] == 0 else 3)
            self.castling[index] = 0 if cp[0] == 0 else self.castling[index]
            
            self.board[cp[1]][cp[0]], self.board[np[1]][np[0]] = 0, part
            hash_key = self.EPD_hash()
            self.EPD_table[hash_key] = self.EPD_table.get(hash_key, 0) + 1
            return True
        
        return False
    
    def valid_move(self, cur_pos, next_pos):
        if cur_pos is not None and next_pos is not None:
            part = self.board[cur_pos[1]][cur_pos[0]]
            if part * self.p_move > 0 and part:
                p_name = self.parts[abs(part)]
                v_moves = getattr(Chess, p_name).movement(self, self.p_move, cur_pos, capture=True)
                
                if len(self.log) > 0 and '+' in self.log[-1]:
                    v_moves = [m for m in v_moves if cur_pos in self.c_escape and m in self.c_escape[cur_pos]]
                
                return next_pos in v_moves
        return False

    
