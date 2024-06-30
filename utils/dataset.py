import torch
from torch.utils.data import Dataset
import os, pickle
import numpy as np
import zipfile
from hashlib import md5
from h5py import File as h5pyFile

class PGNDataset(Dataset):
    def __init__(self, pgn_path, dest_dir):

        pgns = {}
        zfile = zipfile.ZipFile(pgn_path)
        pgn_files = zfile.namelist()
        for file in pgn_files:
            print(file)
            pgn = zfile.open(file).readlines()
            pgns[file.split('.')[0]] = pgn
        zfile.close()

        pgn = pgns['WorldChamp2023']
        #pgn = []
        games = []
        chunk = []
        game = []

        for line in pgn:
            print(line)
            line = line.decode()
            if line.strip() == '':
                game.append(chunk)
                chunk = []
                if len(game) == 2:
                    games.append(game)
                    game = []
            else:
                chunk.append(line.strip())

        self.games = [('\n'.join(game[0]), ' '.join(game[1])) for game in games]

        # filter for games already processed
        done_games = os.listdir(dest_dir)
        self.games = [(meta,gamestring) for meta,gamestring in self.games if f"{md5(meta.encode()).hexdigest()}.chs" not in done_games]

    def __len__(self):
        return len(self.games)
    
    def __getitem__(self, idx):
        return self.games[idx]
    
class HDFDataset(Dataset):
    def __init__(self, source_path):
        super().__init__()

        self.source_path = source_path

        with open(os.path.join(self.source_path, "inventory.txt"), "r") as file:
            self.inventory = file.readlines()

        sizes, self.filenames = zip(*[i.split() for i in self.inventory[1:]])
        self.sizes = [int(s) for s in sizes]
        self.len = sum(self.sizes)
        self.examples_per_hdf = self.sizes[0]

    def __len__(self):
        return self.len
            
    def __getitem__(self, index):

        hdf_index = index // self.examples_per_hdf
        board_index = index - (hdf_index * self.examples_per_hdf)
        hdf_path = os.path.join(self.source_path, self.filenames[hdf_index])
        with h5pyFile(hdf_path, 'r') as hf:
            move = hf["moves"][board_index]
            turn = hf["turns"][board_index]
            board = hf["boards"][board_index]
            evaluation = hf["evaluations"][board_index]

        move = torch.tensor(move)
        turn = torch.tensor(turn)
        board = torch.from_numpy(board)
        evaluation = torch.tensor(evaluation)
            
        return move, turn, board, evaluation

    
