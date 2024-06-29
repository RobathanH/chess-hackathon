from cycling_utils import TimestampedTimer

timer = TimestampedTimer("Imported TimestampedTimer")

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import random_split
from pathlib import Path
import argparse
import os
from models.transformer2 import Model
from utils.dataset import HDFDataset
from torch.utils.data import DataLoader
import socket
import yaml
import wandb

from cycling_utils import (
    InterruptableDistributedSampler,
    MetricsTracker,
    atomic_torch_save,
)
from utils.LAMB import Lamb

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
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--load-dir", type=Path, default=None)
    parser.add_argument("--lr", type=float, default=None)
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
    if args.load_dir and not os.path.isfile(args.save_chk_path):
        # load from load path if one passed and save check path does not exist
        args.load_chk_path = args.load_dir
    else:
        # otherwise presume to save and load from the same place
        args.load_chk_path = args.save_chk_path
    args.save_chk_path.parent.mkdir(parents=True, exist_ok=True)
    timer.report("Validated checkpoint path")

    data_path = "/data"
    dataset = HDFDataset(data_path)
    timer.report("Loaded dataset to RAM")

    random_generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(dataset, [0.8, 0.2], generator=random_generator)

    train_sampler = InterruptableDistributedSampler(train_dataset)
    test_sampler = InterruptableDistributedSampler(test_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=64, sampler=train_sampler)
    test_dataloader = DataLoader(test_dataset, batch_size=64, sampler=test_sampler)
    timer.report("Prepared dataloaders")

    model_config = yaml.safe_load(open(args.model_config))
    model = Model(**model_config)
    model = model.to(args.device_id)
    model = DDP(model, device_ids=[args.device_id])
    timer.report("Prepared model for distributed training")

    loss_fn = nn.MSELoss()
    optimizer = Lamb(model.parameters(), lr=args.lr)
    metrics = {"train": MetricsTracker(), "test": MetricsTracker()}

    if os.path.isfile(args.load_chk_path):
        if args.is_master:
            print(f"Loading checkpoint from {args.load_chk_path}")
        checkpoint = torch.load(args.load_chk_path, map_location=f"cuda:{args.device_id}")

        model.module.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        train_dataloader.sampler.load_state_dict(checkpoint["train_sampler"])
        test_dataloader.sampler.load_state_dict(checkpoint["test_sampler"])
        metrics = checkpoint["metrics"]
        timer.report("Retrieved saved checkpoint")

    grad_accum_steps = 1
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

                evals = logish_transform(evals) # suspect this might help
                #print(evals)

                #evals[evals > 0] *= 10
                boards, evals = boards.to(args.device_id), evals.to(args.device_id)
                scores = model(boards)
                #fake_scores = torch.tensor(evaluate_moves(boards)).cuda()


                #scores = fake_scores

                loss = loss_fn(scores, evals)
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
