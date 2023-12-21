from copy import deepcopy
class Chess:
    def __init__(self, EPD='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -'):
        self.columns = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        self.rows = ['8', '7', '6', '5', '4', '3', '2', '1']
        self.piece_notation = {'p': 1, 'n': 2, 'b': 3, 'r': 4, 'q': 5, 'k': 6}
        self.piece_names = {1: 'Pawn', 2: 'Knight', 3: 'Bishop', 4: 'Rook', 5: 'Queen', 6: 'King'}
        self.reset(EPD=EPD)

    def reset(self, EPD='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -'):
        self.log = []
        self.init_pos = EPD
        self.EPD_table = {}
        self.current_player = 1
        self.castling = [1, 1, 1, 1]
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
        if len(x) == 1 and x.lower() in self.columns and y in self.rows:
            return self.columns.index(x.lower()), self.rows.index(y)
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
        result += ''.join(['K' if self.castling[0] == 1 else '',
                           'Q' if self.castling[1] == 1 else '',
                           'k' if self.castling[2] == 1 else '',
                           'q' if self.castling[3] == 1 else ''])
        result += f' -' if sum(self.castling) == 0 else f' {self.columns[self.en_passant[0]]}{self.rows[self.en_passant[1]]}' if self.en_passant else f' -'
        return result
