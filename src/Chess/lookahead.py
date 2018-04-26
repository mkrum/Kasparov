
import heapq
import copy
import numpy as np
from util import get_simple_input

def get_value(model, t_boards, history):
    return model.evaluate(np.expand_dims(get_simple_input(t_boards, history), 0))

def min_select(boards, model, history):
    curr_board = boards[-1]
    possible = list(curr_board.legal_moves)
    
    min_val = float('inf')
    for move in possible:
        t_boards = copy.deepcopy(boards)
        t_curr = copy.copy(t_boards[-1])
        t_curr.push(move)
        t_boards.append(t_curr)

        val = get_value(model, t_boards, history)
        if val < min_val:
            min_val = val
            best_move = move

    return best_move


def lookahead_select(boards, model, history, depth, max_width=2, selection=min):
    curr_board = boards[-1]
    possible = list(curr_board.legal_moves)
    
    vals = []
    for move in possible:
        t_boards = copy.deepcopy(boards)
        t_curr = copy.copy(t_boards[-1])
        t_curr.push(move)
        t_boards.append(t_curr)

        vals.append(lookahead(t_boards, model, history, depth, max_width))

    return possible[vals.index(selection(vals))]


def lookahead(boards, model, history, depth, max_width=3, selection=min):

    curr_board = boards[-1]
    possible = list(curr_board.legal_moves)
    
    vals = []
    for move in possible:
        t_boards = copy.deepcopy(boards)
        t_curr = copy.copy(t_boards[-1])
        t_curr.push(move)
        t_boards.append(t_curr)

        vals.append(get_value(model, t_boards, history))
    
    if depth > 0:
        if selection is min:
            best_vals = heapq.nsmallest(max_width, vals)
        else:
            best_vals = heapq.nlargest(max_width, vals)

        best_moves = list(map(lambda v: possible[vals.index(v)], best_vals) )
        
        vals = []
        for move in best_moves:
            t_boards = copy.deepcopy(boards)
            t_curr = copy.copy(t_boards[-1])
            t_curr.push(move)
            t_boards.append(t_curr)

            vals.append(lookahead(t_boards, model, history, depth - 1, max_width=max_width, selection=selection))

    return selection(vals)

    
