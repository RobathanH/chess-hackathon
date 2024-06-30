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
from utils.dataset import HDFDataset, SubsetDataset
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler
import socket
import yaml
import pickle
import numpy as np

from utils.LAMB import Lamb

timer.report("Completed imports")

def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", type=Path, default="model_config.yaml")
    parser.add_argument("--save-dir", type=Path, default="saves")
    parser.add_argument("--load-dir", type=Path, default=None)
    parser.add_argument("--lr", type=float, default=0.002)
    return parser

def logish_transform(data):
    reflector = -1 * (data < 0).to(torch.int8)
    return reflector * torch.log(torch.abs(data) + 1)

def main(args, timer):

    args.save_chk_path = args.save_dir / "checkpoint.pt"
    if args.load_dir and not os.path.isfile(args.save_chk_path):
        # load from load path if one passed and save check path does not exist
        args.load_chk_path = args.load_dir
    else:
        # otherwise presume to save and load from the same place
        args.load_chk_path = ""
    args.save_chk_path.parent.mkdir(parents=True, exist_ok=True)
    timer.report("Validated checkpoint path")

    data_path = "/data"
    dataset = HDFDataset(data_path)
    timer.report("Loaded dataset to RAM")

    with open("valid_indices.pkl", "rb") as fp:
        valid_indices = pickle.load(fp)
    with open("valid_index_evals.pkl", "rb") as fp:
        valid_index_evals = pickle.load(fp)
    
    dataset = SubsetDataset(dataset, valid_indices)

    TEMP = 1000
    max_eval = max(max(valid_index_evals), -1 * min(valid_index_evals))
    weights = [np.exp((np.abs(e) - max_eval) / TEMP) for e in valid_index_evals]
    print(f"sample weight bounds: {min(weights)}, {max(weights)}, mean={np.mean(weights)}")

    random_generator = torch.Generator().manual_seed(42)
    train_indices, test_indices = random_split(range(len(dataset)), [0.8, 0.2], generator=random_generator)
    train_dataset = SubsetDataset(dataset, train_indices)
    test_dataset = SubsetDataset(dataset, test_indices)
    train_sampler = WeightedRandomSampler([weights[i] for i in train_indices], 64)
    test_sampler = WeightedRandomSampler([weights[i] for i in test_indices], 64)
    train_dataloader = DataLoader(dataset, batch_size=64, sampler=train_sampler)
    test_dataloader = DataLoader(dataset, batch_size=64, sampler=test_sampler)
    timer.report("Prepared dataloaders")

    # MANUALLY SET MODEL
    model_config = yaml.safe_load(open("models/transformer.yaml"))
    from models.rel2d_transformer import Model
    model = Model(**model_config)
    model.to("cpu")
    timer.report("Prepared model")

    loss_fn = nn.MSELoss()
    optimizer = Lamb(model.parameters(), lr=args.lr)
    metrics = {"train": [], "test": []}

    if os.path.isfile(args.load_chk_path):
        print(f"Loading checkpoint from {args.load_chk_path}")
        checkpoint = torch.load(args.load_chk_path)

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        metrics = checkpoint["metrics"]
        timer.report("Retrieved saved checkpoint")

    grad_accum_steps = 10
    save_steps = 10

    for epoch in range(0, 10000):


        timer.report(f"Training epoch {epoch}")
        train_steps_per_epoch = len(train_dataloader)
        optimizer.zero_grad()
        model.train()

        for step, (_moves, _turns, boards, evals) in enumerate(train_dataloader):

            # Determine the current step
            is_last_step = (step + 1) == train_steps_per_epoch

            evals = logish_transform(evals) # suspect this might help
            scores = model(boards)
            loss = loss_fn(scores, evals)
            loss = loss / grad_accum_steps
            loss.backward()

            top_eval_index = evals.argmax()
            top1_score_indices = torch.topk(scores, 1).indices
            top5_score_indices = torch.topk(scores, 5).indices

            metrics["train"].append({})
            metrics["train"][-1].update({
                "examples_seen": len(evals),
                "accum_loss": loss.item(), 
                "top1_accuracy": 1 if top_eval_index in top1_score_indices else 0, 
                "top5_accuracy": 1 if top_eval_index in top5_score_indices else 0
            })

            if (step + 1) % grad_accum_steps == 0 or is_last_step:

                optimizer.step()
                optimizer.zero_grad()

                rpt = metrics["train"][-1]
                rpt_loss =rpt["accum_loss"]
                rpt_top1 = rpt["top1_accuracy"] / rpt["examples_seen"]
                rpt_top5 = rpt["top5_accuracy"] / rpt["examples_seen"]
                print(f"Step {step * 64}, Loss {rpt_loss:,.3f}, Top1 {rpt_top1:,.3f}, Top5 {rpt_top5:,.3f}, Examples: {rpt['examples_seen']:,.0f}")

            # Saving
            if ((step + 1) % save_steps == 0 or is_last_step):
                # Save checkpoint
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "metrics": metrics
                    },
                    args.save_chk_path,
                )


        timer.report(f"Testing epoch {epoch}")
        test_steps_per_epoch = len(test_dataloader)
        model.eval()

        with torch.no_grad():
            for step, (_moves, _turns, boards, evals) in enumerate(test_dataloader):

                # Determine the current step
                is_last_step = (step + 1) == test_steps_per_epoch

                evals = logish_transform(evals) # suspect this might help
                scores = model(boards)
                loss = loss_fn(scores, evals)

                top_eval_index = evals.argmax()
                top1_score_indices = torch.topk(scores, 1).indices
                top5_score_indices = torch.topk(scores, 5).indices

                metrics["test"].append({})
                metrics["test"][-1].update({
                    "examples_seen": len(evals),
                    "accum_loss": loss.item(), 
                    "top1_accuracy": 1 if top_eval_index in top1_score_indices else 0, 
                    "top5_accuracy": 1 if top_eval_index in top5_score_indices else 0
                    })
                
                # Reporting
                if is_last_step:

                    rpt = metrics["test"][-1]
                    rpt_loss = rpt["accum_loss"] / rpt["examples_seen"]
                    rpt_top1 = rpt["top1_accuracy"] / rpt["examples_seen"]
                    rpt_top5 = rpt["top5_accuracy"] / rpt["examples_seen"]
                    print(f"Epoch {epoch}, Loss {rpt_loss:,.3f}, Top1 {rpt_top1:,.3f}, Top5 {rpt_top5:,.3f}")

                
                # Saving
                if ((step + 1) % save_steps == 0 or is_last_step):
                    # Save checkpoint
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "metrics": metrics
                        },
                        args.save_chk_path,
                    )


timer.report("Defined functions")
if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args, timer)
