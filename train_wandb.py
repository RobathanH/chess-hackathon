from cycling_utils import TimestampedTimer

timer = TimestampedTimer("Imported TimestampedTimer")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import random_split
from pathlib import Path
import argparse
import os, sys
from utils.dataset import HDFDataset, SubsetDataset, ConcatDataset
from torch.utils.data import DataLoader
import socket
import yaml
import wandb
import pickle

from cycling_utils import (
    InterruptableDistributedSampler,
    MetricsTracker,
    atomic_torch_save,
)
from utils.LAMB import Lamb

'''
class WeightedDistributedSampler(InterruptableDistributedSampler):
    def __init__(
        self,
        dataset: Dataset,
        num_replicas: int | None = None,
        rank: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        weights: list | None = None
    ) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.weights = weights


    def state_dict(self):
        return {"progress": self.progress, "epoch": self.epoch, "weights": self.weights}

    def load_state_dict(self, state_dict):
        self.progress = state_dict["progress"]
        if not self.progress <= self.num_samples:
            raise AdvancedTooFarError(
                f"progress should be less than or equal to the number of samples. progress: {self.progress}, num_samples: {self.num_samples}"
            )
        self.epoch = state_dict["epoch"]
        self.weights = weights

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        # Weighted sample with replacement
        indices = torch.multinomial([self.weights[i] for i in indices], self.num_samples, True).tolist()

        # slice from progress to pick up where we left off
        for idx in indices[self.progress :]:
            yield idx
'''

timer.report("Completed imports")

import numpy as np
square_states = {
    'empty': 0,
    'my_pawn': 1, 'my_passant_pawn': 2,
    'my_virgin_rook': 3, 'my_moved_rook': 4,
    'my_knight': 5, 'my_ls_bishop': 6, 'my_ds_bishop': 7,
    'my_queen': 8, 'my_virgin_king': 9, 'my_moved_king': 10,
    'op_pawn': 11, 'op_passant_pawn': 12,
    'op_virgin_rook': 13, 'op_moved_rook': 14,
    'op_knight': 15, 'op_ls_bishop': 16, 'op_ds_bishop': 17,
    'op_queen': 18, 'op_virgin_king': 19, 'op_moved_king': 20
}

# Piece values based on standard chess piece values
piece_values = {
    'empty': 0,
    'my_pawn': 1, 'my_passant_pawn': 1,
    'my_virgin_rook': 5, 'my_moved_rook': 5,
    'my_knight': 3, 'my_ls_bishop': 3, 'my_ds_bishop': 3,
    'my_queen': 9, 'my_virgin_king': 0, 'my_moved_king': 0,
    'op_pawn': -1, 'op_passant_pawn': -1,
    'op_virgin_rook': -5, 'op_moved_rook': -5,
    'op_knight': -3, 'op_ls_bishop': -3, 'op_ds_bishop': -3,
    'op_queen': -9, 'op_virgin_king': 0, 'op_moved_king': 0
}

# Reverse dictionary for easier lookup
value_dict = {v: piece_values[k] for k, v in square_states.items()}

def evaluate_board(board):
    value = 0
    for row in board:
        for square in row:
            value += value_dict[int(square)]
    return value

def evaluate_moves(moves):
    """
    Evaluate a set of possible moves.
    
    Parameters:
    moves (numpy.ndarray): A tensor of shape (N, 8, 8) representing N possible board states.
    
    Returns:
    numpy.ndarray: An array of values of length N representing the evaluation of each board state.
    """
    values = np.zeros(len(moves))
    for i, board in enumerate(moves):
        values[i] = evaluate_board(board)
    return values


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--load-dir", type=Path, default=None)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--grad_accum", type=int, default=2)
    return parser

def logish_transform(data):
    reflector = -1 * (data < 0).to(torch.int8)
    return reflector * torch.log(torch.abs(data) + 1)

def main(args, timer):

    dist.init_process_group("nccl")  # Expects RANK set in environment variable
    rank = int(os.environ["RANK"])  # Rank of this GPU in cluster
    world_size = int(os.environ["WORLD_SIZE"]) # Total number of GPUs in the cluster
    args.device_id = int(os.environ["LOCAL_RANK"])  # Rank on local node
    args.is_master = rank == 0  # Master node for saving / reporting
    torch.cuda.set_device(args.device_id)  # Enables calling 'cuda'

    if args.device_id:
        hostname = socket.gethostname()
        print("Hostname:", hostname)
    timer.report("Setup for distributed training")

    args.save_chk_path = args.save_dir / "checkpoint.pt"
    if args.load_dir:
        # load from load path if one passed and save check path does not exist
        args.load_chk_path = args.load_dir / "checkpoint.pt"
    else:
        args.load_chk_path = ""
    args.save_chk_path.parent.mkdir(parents=True, exist_ok=True)
    timer.report("Validated checkpoint path")

    data_path = "/data"
    dataset = HDFDataset(data_path)
    timer.report("Loaded dataset to RAM")

    with open("valid_indices.pkl", "rb") as fp:
        valid_indices = set(pickle.load(fp))
    '''
    with open("valid_index_evals.pkl", "rb") as fp:
        valid_index_evals = pickle.load(fp)
    '''
    
    dataset = SubsetDataset(dataset, valid_indices)
    timer.report("Deduplicated dataset")

    # Add extra data
    if True:
        extra_dataset = torch.load("general_dataset.pt")
        '''
        extra_dataset = [
            (torch.tensor(0), torch.tensor(0), board, val)
            for _, _, board, val in extra_dataset
        ]
        torch.save(extra_dataset, "general_dataset.pt")
        '''
        dataset = ConcatDataset(dataset, extra_dataset)

    '''
    TEMP = 1000
    max_eval = max(max(valid_index_evals), -1 * min(valid_index_evals))
    weights = [np.exp((np.abs(e) - max_eval) / TEMP) for e in valid_index_evals]
    timer.report("Computed weights")

    random_generator = torch.Generator().manual_seed(42)
    train_indices, test_indices = random_split(range(len(dataset)), [0.8, 0.2], generator=random_generator)
    train_dataset = SubsetDataset(dataset, train_indices)
    test_dataset = SubsetDataset(dataset, test_indices)
    train_sampler = InterruptableDistributedSampler(train_dataset, weights=[weights[i] for i in train_indices])
    test_sampler = InterruptableDistributedSampler(test_dataset, weights=[weights[i] for i in test_indices])
    train_dataloader = DataLoader(train_dataset, batch_size=64, sampler=train_sampler)
    test_dataloader = DataLoader(test_dataset, batch_size=64, sampler=test_sampler)
    '''

    random_generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(dataset, [0.8, 0.2], generator=random_generator)

    train_sampler = InterruptableDistributedSampler(train_dataset)
    test_sampler = InterruptableDistributedSampler(test_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler)
    timer.report("Prepared dataloaders")

    #from models.convolutional import Model
    #model_config = yaml.safe_load(open("models/convolutional.yaml"))
    #model = Model(**model_config)

    #model_config = yaml.safe_load(open("models/convolutional.yaml"))
    #from models.masked_convolutional import Model
    #model = Model(**model_config)
    model_config = yaml.safe_load(open("models/transformer.yaml"))
    from models.rel2d_transformer import Model
    #from models.transformer import Model
    model = Model(**model_config)
    model = model.to(args.device_id)
    model = DDP(model, device_ids=[args.device_id])
    timer.report("Prepared model for distributed training")

    loss_fn = nn.MSELoss()
    if False:
        optimizer = Lamb(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    metrics = {"train": MetricsTracker(), "test": MetricsTracker()}

    if os.path.isfile(args.load_chk_path):
        if args.is_master:
            print(f"Loading checkpoint from {args.load_chk_path}")
        checkpoint = torch.load(args.load_chk_path, map_location=f"cuda:{args.device_id}")

        model.module.load_state_dict(checkpoint["model"], strict=False)
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
        except:
            optimizer = optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        train_dataloader.sampler.load_state_dict(checkpoint["train_sampler"])
        test_dataloader.sampler.load_state_dict(checkpoint["test_sampler"])
        metrics = checkpoint["metrics"]
        timer.report("Retrieved saved checkpoint")

    grad_accum_steps = args.grad_accum
    save_steps = 5

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="chess-bot",

        # track hyperparameters and run metadata
        config={
        "epochs": 10,
        }
    )

    for epoch in range(train_dataloader.sampler.epoch, 10000):

        with train_dataloader.sampler.in_epoch(epoch):

            timer.report(f"Training epoch {epoch}")
            train_steps_per_epoch = len(train_dataloader)
            optimizer.zero_grad()
            model.train()

            for _moves, _turns, boards, evals in train_dataloader:

                # Determine the current step
                step = train_dataloader.sampler.progress // train_dataloader.batch_size
                is_last_step = (step + 1) == train_steps_per_epoch

                if True:
                    evals = logish_transform(evals) # suspect this might help

                # Horizontal flip augment
                if torch.randint(2, (1,))[0] == 1:
                    boards = torch.flip(boards, dims=(2,))

                #print(evals)

                #evals[evals > 0] *= 10
                boards, evals = boards.to(args.device_id), evals.to(args.device_id)
                scores = model(boards)
                #fake_scores = torch.tensor(evaluate_moves(boards)).cuda()


                #scores = fake_scores

                loss_version = 1
                if loss_version == 1:
                    loss = loss_fn(scores, evals)
                elif loss_version == 2:
                    eval_threshold = torch.topk(evals, 10, sorted=True).values[-1]
                    top_evals = (evals >= eval_threshold).type(torch.float)
                    loss = F.binary_cross_entropy(F.softmax(scores), top_evals)
                elif loss_version == 3:
                    # Weight samples by their position relative to batch
                    # Push above-average samples to high scores and vice-versa
                    sample_weights = (evals - torch.mean(evals)) / torch.std(evals)
                    loss = (-sample_weights * (scores - scores.mean())).mean()
                elif loss_version == 4:
                    # Encourage difference betweeen scores to scale with dif between evals
                    eval_diffs = evals[:, None] - evals[None, :]
                    score_diffs = scores[:, None] - scores[None, :]
                    loss = -1 * (eval_diffs * score_diffs).mean()
                elif loss_version == 5:
                    # mse loss + std bonus
                    loss = loss_fn(scores, evals) - 0.1 * torch.std(scores)

                loss = loss / grad_accum_steps
                loss.backward()
                train_dataloader.sampler.advance(len(evals))

                top_eval_index = evals.argmax()
                top1_score_indices = torch.topk(scores, 1).indices
                top5_score_indices = torch.topk(scores, 5).indices

                metrics["train"].update({
                    "examples_seen": len(evals),
                    "accum_loss": loss.item(), 
                    "top1_accuracy": 1 if top_eval_index in top1_score_indices else 0, 
                    "top5_accuracy": 1 if top_eval_index in top5_score_indices else 0
                })

                #print(_moves.shape)
                #print(_turns.shape)
                #print(boards.shape)
                #print(evals.shape)

                if (step + 1) % grad_accum_steps == 0 or is_last_step:

                    optimizer.step()
                    optimizer.zero_grad()
                    metrics["train"].reduce()

                    if args.is_master:
                        rpt = metrics["train"].local
                        rpt_loss =rpt["accum_loss"]
                        rpt_top1 = rpt["top1_accuracy"] / rpt["examples_seen"]
                        rpt_top5 = rpt["top5_accuracy"] / rpt["examples_seen"]

                        wandb.log({
                                "examples_seen": len(evals),
                                "loss": rpt_loss, 
                                "top1_accuracy": rpt_top1, 
                                "top5_accuracy": rpt_top5
                                })

                        print(f"Step {train_dataloader.sampler.progress}, Loss {rpt_loss:,.3f}, Top1 {rpt_top1:,.3f}, Top5 {rpt_top5:,.3f}, Examples: {rpt['examples_seen']:,.0f}")

                    metrics["train"].reset_local()

                # Saving
                if ((step + 1) % save_steps == 0 or is_last_step) and args.is_master:
                    # Save checkpoint
                    atomic_torch_save(
                        {
                            "model": model.module.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "train_sampler": train_dataloader.sampler.state_dict(),
                            "test_sampler": test_dataloader.sampler.state_dict(),
                            "metrics": metrics
                        },
                        args.save_chk_path,
                    )

            with test_dataloader.sampler.in_epoch(epoch):

                timer.report(f"Testing epoch {epoch}")
                test_steps_per_epoch = len(test_dataloader)
                model.eval()

                with torch.no_grad():
                    for _moves, _turns, boards, evals in test_dataloader:



                        # Determine the current step
                        step = test_dataloader.sampler.progress // test_dataloader.batch_size
                        is_last_step = (step + 1) == test_steps_per_epoch

                        evals = logish_transform(evals) # suspect this might help
                        boards, evals = boards.to(args.device_id), evals.to(args.device_id)
                        scores = model(boards)
                        loss = loss_fn(scores, evals) 
                        test_dataloader.sampler.advance(len(evals))

                        top_eval_index = evals.argmax()
                        top1_score_indices = torch.topk(scores, 1).indices
                        top5_score_indices = torch.topk(scores, 5).indices

                        metrics["test"].update({
                            "examples_seen": len(evals),
                            "accum_loss": loss.item(), 
                            "top1_accuracy": 1 if top_eval_index in top1_score_indices else 0, 
                            "top5_accuracy": 1 if top_eval_index in top5_score_indices else 0
                            })
                        
                        # Reporting
                        if is_last_step:
                            metrics["test"].reduce()

                            if args.is_master:
                                rpt = metrics["test"].local
                                rpt_loss =rpt["accum_loss"] / rpt["examples_seen"]
                                rpt_top1 = rpt["top1_accuracy"] / rpt["examples_seen"]
                                rpt_top5 = rpt["top5_accuracy"] / rpt["examples_seen"]



                                print(f"Epoch {epoch}, Loss {rpt_loss:,.3f}, Top1 {rpt_top1:,.3f}, Top5 {rpt_top5:,.3f}")




                            metrics["test"].reset_local()
                        
                        # Saving
                        if ((step + 1) % save_steps == 0 or is_last_step) and args.is_master:
                            # Save checkpoint
                            atomic_torch_save(
                                {
                                    "model": model.module.state_dict(),
                                    "optimizer": optimizer.state_dict(),
                                    "train_sampler": train_dataloader.sampler.state_dict(),
                                    "test_sampler": test_dataloader.sampler.state_dict(),
                                    "metrics": metrics
                                },
                                args.save_chk_path,
                            )

    wandb.finish()


timer.report("Defined functions")
if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args, timer)
