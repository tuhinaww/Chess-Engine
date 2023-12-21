class Chess:
    def __init__(self, EPD='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -'):
        self.columns = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        self.rows = ['8', '7', '6', '5', '4', '3', '2', '1']
        self.piece_notation = {'p': 1, 'n': 2, 'b': 3, 'r': 4, 'q': 5, 'k': 6}
        self.piece_names = {1: 'Pawn', 2: 'Knight', 3: 'Bishop', 4: 'Rook', 5: 'Queen', 6: 'King'}
        self.reset(EPD=EPD)