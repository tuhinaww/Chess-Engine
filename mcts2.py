import random
import math
from copy import deepcopy
from abc import Chess  # assuming your Chess class is in a file named abc.py

PLAYER = 1
OPPONENT = -1

class TreeNode():

    def __init__(self, board):
        self.M = 0
        self.V = 0
        self.visitedMovesAndNodes = []
        self.nonVisitedLegalMoves = []
        self.board = board
        self.parent = None
        for m in self.board.possible_board_moves(capture=True):
            self.nonVisitedLegalMoves.append(m)

    def isMCTSLeafNode(self):
        return len(self.nonVisitedLegalMoves) != 0

    def isTerminalNode(self):
        return len(self.nonVisitedLegalMoves) == 0 and len(self.visitedMovesAndNodes) == 0

def uctValue(node, parent):
    val = (node.M / node.V) + 1.4142 * math.sqrt(math.log(parent.V) / node.V)
    return val

def select(node):
    if node.isMCTSLeafNode() or node.isTerminalNode():
        return node
    else:
        maxUctChild = None
        maxUctValue = -1000000.
        for move, child in node.visitedMovesAndNodes:
            uctValChild = uctValue(child, node)
            if uctValChild > maxUctValue:
                maxUctChild = child
                maxUctValue = uctValChild
        if maxUctChild is None:
            raise ValueError("Could not identify child with the best UCT value")
        else:
            return select(maxUctChild)

def expand(node):
    moveToExpand = random.choice(node.nonVisitedLegalMoves)
    new_board = deepcopy(node.board)
    new_board.move(moveToExpand[0], moveToExpand[1])
    childNode = TreeNode(new_board)
    childNode.parent = node
    node.visitedMovesAndNodes.append((moveToExpand, childNode))
    return childNode

def simulate(node):
    board = deepcopy(node.board)
    while board.is_end() == [0, 0, 0]:
        legal_moves = board.possible_board_moves(capture=True)
        move = random.choice(list(legal_moves.keys()))
        board.move(move[0], move[1])
    
    outcome = board.is_end()
    if outcome[0] == 1:
        return 1.0  # Player wins
    elif outcome[2] == 1:
        return 0.0  # Opponent wins
    else:
        return 0.5  # Draw

def backpropagate(node, payout):
    node.M += payout
    node.V += 1
    if node.parent is not None:
        return backpropagate(node.parent, payout)

# Example usage:

# Initialize your Chess game
game = Chess()

# Create the root node of the MCTS tree
root = TreeNode(game)

# Run MCTS for a certain number of iterations (e.g., 1000)
for _ in range(1000):
    # Selection
    selected_node = select(root)

    # Expansion
    if not selected_node.isMCTSLeafNode():
        selected_node = expand(selected_node)

    # Simulation
    simulation_result = simulate(selected_node)

    # Backpropagation
    backpropagate(selected_node, simulation_result)

# Choose the best move based on the MCTS results
best_move = max(root.visitedMovesAndNodes, key=lambda x: x[1].V)[0]

print("Best Move:", best_move)
