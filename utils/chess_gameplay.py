import torch
import torch.nn as nn
import numpy as np
from random import choices
from itertools import accumulate
from utils.chess_primitives import init_board, conjugate_board, candidate_moves, in_check, is_draw, get_played, which_board, evaluate_position
from chess import Board
import chess.svg
import cairosvg
import time
import copy

def softmax_temp(x, temp=1):
    z = np.exp((x - x.max()) / temp)
    return z / z.sum()

def entropy(d):
    # Returns the entropy of a discrete distribution
    e = -(d * np.log2(d + 1e-10)).sum() # epsilon value added due to log2(0) == undefined.
    return e

def entropy_temperature(x, target_entropy, T=[1e-3, 1e0, 1e2], tol=1e-3, max_iter=10):
    # returns the temperature parameter (to within tol) required to transform the vector x into a 
    # probability distribution with a particular target entropy
    delta = np.inf
    for _ in range(max_iter):
        if delta > tol:
            E = [entropy(softmax_temp(x, temp=t)) for t in T]
            if E[0] > target_entropy:
                T = [T[0]/2, T[1], T[2]]
            elif E[2] < target_entropy:
                T = [T[0], T[1], T[2]*2]
            elif E[0] < target_entropy < E[1]:
                T = [T[0], (T[0]+T[1])/2, T[1]]
            elif E[1] < target_entropy < E[2]:
                T = [T[1], (T[1]+T[2])/2, T[2]]
            delta = T[2] - T[1]
        else:
            return (T[0]+T[2]) / 2
    return (T[0]+T[2]) / 2

def selector(scores, p=0.3, k=3):
    '''
    Squashes the options distribution to have a target (lower) entropy.
    Selects a token, based on log2(p * len(k)) degrees of freedom.
    '''

    # If there is no variance in the scores, then just chose randomly.
    if all([score == scores[0] for score in scores]): 
        return choices(range(len(scores)))[0]
    else:
        # Otherwise target entropy is either proportion p * max_possible_entropy (for small option sets) or 
        # as-if k-degree of freedom distribution (for num_scores >> k)
        target_entropy = min(p * np.log2(len(scores)), np.log2(k))
        # If we abandon the second term above, we allow the model more freedom when there are more options to 
        # chose from. Actually we could achieve the same thing by setting k ~ inf. Numpy handles this just fine 
        # so np.log2(float('inf')) = inf
        t = entropy_temperature(scores, target_entropy)
        dist = softmax_temp(scores, temp=t)
        return choices(range(len(scores)), cum_weights=list(accumulate(dist)))[0]

class Agent:
    def __init__(self, model=None, p=0.3, k=3):
        self.model, self.p, self.k = model, p, k

        if self.model:
            assert isinstance(model, nn.Module), "ERROR: model must be a torch nn.Module"
            self.model.eval()

    def select_move(self, options):
        # If there is no model passed, then just chose randomly.
        if self.model is None:
            return choices(range(len(options)))[0]
        with torch.no_grad():
            # Score the options with the model
            # scores = self.model(torch.tensor(options))

            # Split the options into individual boards for scoring.
            individual_boards = [torch.tensor(board).unsqueeze(0) for board in options]
            scores = np.array([self.model(board).item() for board in individual_boards])
            
        # Select end token
        selection = selector(scores, self.p, self.k)
        return selection

def play_game(table, agents, max_moves=float('inf'), min_seconds_per_move=2, verbose=False, poseval=False):

    # board, color_toplay = starting_state if starting_state is not None else (init_board(play_as='white'), 'white')
    board, color_toplay = init_board(play_as='white'), 'white'
    chs_board = Board()
    chs_boardsvg = chess.svg.board(chs_board, size=600, orientation=chess.WHITE, borders=False, coordinates=False)
    board_png_file_path = f"/mnt/chess/image{table}b.png"
    cairosvg.svg2png(bytestring=chs_boardsvg.encode('utf-8'), write_to="board_temp.png")

    game_result = {'white': {'moves': [], 'points': 0}, 'black': {'moves': [], 'points': 0}, 'all_moves': [(board, chs_board, None, chs_boardsvg)]}

    # Play a game until game over.
    while True:

        start = time.perf_counter()

        # Revert any passant pawns to regular pawns if they survived the last turn.
        board[board == 2] = 1
        # Options from each of the starting positions - init as empty dict
        options = candidate_moves(board)
        
        # Check if checkmate or draw.
        player_points, opponent_points, outcome = (None, None, None)
        if len(options) == 0:
            if in_check(board): # Checkmate
                player_points, opponent_points = (-1.0, 1.0)
                outcome = 'Checkmate'

            else: # Stalemate
                player_points, opponent_points = (0.0, 0.0)
                outcome = 'Stalemate'

        if is_draw(board) or len(game_result[color_toplay]['moves']) >= max_moves: # Known draw or max moves reached
            player_points, opponent_points = (0.0, 0.0)
            outcome = 'Draw or timeout'

            if poseval:
                if verbose:
                    print(f"{outcome} after {len(game_result[color_toplay]['moves'])} moves. Stockfish evaluating...")
                if color_toplay == 'white':
                    white_score = evaluate_position(chs_board.fen(), depth_limit=25)
                    if white_score > 0:
                        player_points, opponent_points = (1.0, -1.0)
                    else:
                        player_points, opponent_points = (-1.0, 1.0)
                else:
                    white_score = - evaluate_position(chs_board.fen(), depth_limit=25)
                    if white_score > 0:
                        player_points, opponent_points = (-1.0, 1.0)
                    else:
                        player_points, opponent_points = (1.0, -1.0)
            else:
                # No position evaluation for tiebreak, just a draw.
                player_points, opponent_points = (0.0, 0.0)

        if player_points is not None:
            player, opponent = ('white', 'black') if color_toplay == 'white' else ('black','white')
            game_result[player]['points'] = player_points
            game_result[opponent]['points'] = opponent_points
            return game_result

        move_not_selected = True
        while move_not_selected:

            # Select end_token
            move_selection = agents[color_toplay].select_move(options)
            selected_board = options[move_selection]

            ## GET THE SELECTED MOVE PGN, UPDATE CHS_BOARD FOR RENDERING
            chs_board_pgns = [None] * len(options)
            for cand_move in chs_board.legal_moves:
                # PGN token corresponding to this legal move
                cand_pgn = chs_board.san(cand_move)
                # Pre-process
                cand_pgn = cand_pgn.replace('x','').replace('+','').replace('#','')
                target_board = get_played(board, cand_pgn, color_toplay, options)
                cand_ind = which_board(target_board, options)
                chs_board_fwd = copy.deepcopy(chs_board)

                move = chs_board_fwd.parse_san(cand_pgn)
                uci_token = move.uci()

                chs_board_fwd.push_san(cand_pgn)
                chs_board_pgns[cand_ind] = (chs_board_fwd, cand_pgn, uci_token)

            chs_board, pgn_token, uci_token = chs_board_pgns[move_selection]
            move = chess.Move.from_uci(uci_token)
            chs_boardsvg = chess.svg.board(chs_board, size=600, orientation=chess.WHITE, lastmove=move, borders=False, coordinates=False)

            move_not_selected = False

        # Move is now selected, chs_board and chs_boardsvg now reflects the updated board state. Send to S3
        cairosvg.svg2png(bytestring=chs_boardsvg.encode('utf-8'), write_to="board_temp.png")

        # Add this move to the game_record
        game_result[color_toplay]['moves'].append((selected_board, chs_board, pgn_token, chs_boardsvg))
        game_result['all_moves'].append((selected_board, chs_board, pgn_token, chs_boardsvg))

        if verbose:
            print(f"{color_toplay}: {pgn_token}")

        # Swap to opponent's perspective
        color_toplay = 'white' if color_toplay == 'black' else 'black' # Swap to my turn
        board = conjugate_board(selected_board) # Conjugate selected_end_board to opponents perspective

        # Delay next move so that humans can watch!
        move_duration = time.perf_counter() - start
        time_remaining = min_seconds_per_move - move_duration
        if time_remaining > 0:
            time.sleep(time_remaining)

def play_tournament(table, agents, max_games=4, max_moves=float('inf'), min_seconds_per_move=5, verbose=False, poseval=False):
    # plays a number of paired games, one with agent0 as white, the other with agent0 as black.
    tournament_game_results = []
    is_draw = True

    tournament_results = dict()
    tournament_results['agent0'] = 0
    tournament_results['agent1'] = 0

    while is_draw:

        if verbose:
            print(f"\nPlaying Game {len(tournament_game_results) + 1}")
    
        # play game with FIRST model as white
        kwargs = {'table': table, 'agents': {'white': agents[0], 'black': agents[1]}, 'max_moves': max_moves, 'min_seconds_per_move': min_seconds_per_move, "verbose": verbose, "poseval": poseval}
        game_result = play_game(**kwargs)
        tournament_game_results.append(game_result)

        # game_results: {'white': {'moves': [(end_token, end_board), (end_token, end_board), ...], 'points': float}, 'black': {...}}
        tournament_results['agent0'] += game_result['white']['points']
        tournament_results['agent1'] += game_result['black']['points']

        if verbose:
            print(f"\nPlaying Game {len(tournament_game_results) + 1}")

        # play game with SECOND model as white
        kwargs = {'table': table, 'agents': {'white': agents[1], 'black': agents[0]}, 'max_moves': max_moves, 'min_seconds_per_move': min_seconds_per_move, "verbose": verbose, "poseval": poseval}
        game_result = play_game(**kwargs)
        tournament_game_results.append(game_result)
        # game_results: {'white': {'moves': [(end_token, end_board), (end_token, end_board), ...], 'points': float}, 'black': {...}}
        tournament_results['agent0'] += game_result['black']['points']
        tournament_results['agent1'] += game_result['white']['points']

        # Check if draw, if so, play again!
        is_draw = tournament_results['agent0'] == tournament_results['agent1']

        if is_draw:
            print("DRAW!")

        # If we've played our max_games, call it a day.
        if len(tournament_game_results) >= max_games:
            # End the loop
            is_draw = False

    return tournament_results, tournament_game_results
