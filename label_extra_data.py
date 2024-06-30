import pickle
from tqdm import tqdm, trange
import torch
from utils.chess_primitives import *

def convert_labeled_game(input_game):

    # Output of function
    states = []
    evals = []

    board = init_board()
    turn = "white"

    for move, stockfish_eval in zip(input_game["moves_pgn"], input_game["eval"]):
        # Retrieve the move
        move = move.replace('x','').replace('+','').replace('#','') # Drop capture/check/checkmate notation - unnecessary
        # Revert any passant pawns to regular pawns if they survived the enemy's turn.
        board[board==2] = 1
        # Calculate legal moves from this position
        candidates = candidate_moves(board)
        # Generate candidates and identify the board of the move that was made
        played_board = get_played(board, move, turn, candidates, False)
        # Maybe terminate
        if played_board is None:
            return states, evals

        # Add updated board to list - if not checkmate
        if stockfish_eval["type"] == "cp":
            states.append(played_board.copy())
            evals.append(stockfish_eval["value"] * (1 if turn == "white" else -1))

        # Record the new board state
        board = played_board
        # Conjugate board to other side's view
        board = conjugate_board(board)
        # Turn play over to the other side.
        if turn == 'white':
            turn = 'black'
        else:
            turn = 'white'

    return states, evals



new_dataset = []

for i in trange(15):
    with open(f"labeled_games/{i:08}.pkl", "rb") as fp:
        data = pickle.load(fp)

    for game in tqdm(data):
        game_states, game_vals = convert_labeled_game(game)
        new_dataset += list(zip(game_states, game_vals))

    torch.save(new_dataset, "general_dataset.pt")