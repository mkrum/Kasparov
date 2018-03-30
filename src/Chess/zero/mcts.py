import copy
import numpy as np
from util import get_input
import chess

def softmax(vals):
    '''
    scales all values to [0, 1]
    '''
    return [ x / sum(vals) for x in vals]

class MCTS(object):

    def __init__(self, model, boards):
        self.model = model
        self.root = Node(self.model, boards)

    def search(self, iters=10):
        '''
        Perform the MCTS search algorithm from the root of the tree
        '''

        for _ in range(iters):
            self.root.search(self.model)
        
        return self.root.select_move()
    
    def get_probabilities(self, tau):
        '''
        returns the probabilites found in the MCTS for each possible move
        '''
        return softmax([1] * len(self.root.children))

class Edge(object):

    def __init__(self, parent, child, move, prior):
        self.parent = parent
        self.child = child
        self.move = move

        self.P = prior
        self.N = 0
        self.Q = child.value
        self.U = 0
        
    def backprop(self, value):
        '''
        Adjust the values of all the edges in the current path based on the new node
        '''

        if self.parent.leading_edge is not None:
            self.parent.leading_edge.backprop(-1* value)

        self.N += 1
        self.Q += 1.0 /self.N * (value - self.Q)
        self.U = self.P / (1.0 + self.N)


class Node(object):

    def __init__(self, model, boards):
        self.model = model
        self.boards = boards
        self.leading_edge = None
        self.edges = []
        self.value = model.evaluate(get_input(self.boards))

    def attach(self, edge):
        self.leading_edge = edge

    def is_leaf(self):
        return len(self.edges) == 0
    
    def is_terminal(self):
        possible_moves = self.game.get_possible()
        return len(possible_moves) == 0

    def search(self, model):
        '''
        Keep selecting nodes in the tree until you reach a leaf. Then, expand the node, select one of its children
        and backpropagate its value back to the root node.
        '''

        if self.is_leaf():
            self.expand()
            edge = self.select_edge()

            if edge:
                edge.backprop(edge.child.value)

            if len(self.edges) == 0:
                self.leading_edge.backprop(self.value)

        else:
            edge = self.select_edge()
            edge.child.search(model)
    
    def expand(self):
        '''
        Create children for all the possible moves from a node
        '''

        possible_moves = list(self.boards[-1].legal_moves)
        edge_probs = [1] * len(possible_moves)
        edge_probs = softmax(edge_probs)
        
        for move, prob in zip(possible_moves, edge_probs):
            new_boards = copy.deepcopy(self.boards)
            hyp_board = copy.copy(new_boards[-1])
            hyp_board.push(move)
            new_boards.append(hyp_board)


            child_node = Node(self.model, new_boards)
            new_edge = Edge(self, child_node, move, prob)
                
            child_node.attach(new_edge)

            self.edges.append(new_edge)
    
    def select_edge(self):
        '''
        Node selection algorithm, based off the "upper confidence bound" of the value estimate
        '''

        if len(self.edges) == 0:
            return None

        uct = list(map(lambda e: e.U + e.Q, self.edges))
        return self.edges[uct.index(max(uct))]

    def select_move(self):
        '''
        selects the best move from the root node, decided by total number of visits
        '''

        edge_counts = list(map(lambda e: e.N, self.edges))
        
        return self.edges[edge_counts.index(max(edge_counts))].move

def mcts_evaluate(model, boards, tau=1.0):
    '''
    Board evaluation function using MCTS
    '''

    if len(boards) == 0:
        boards = [chess.Board()]

    tree = MCTS(model, copy.deepcopy(boards))
    move = tree.search()

    return move

