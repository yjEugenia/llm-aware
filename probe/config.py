# -*- coding: utf-8 -*-
import argparse
import json
import os
import random
from dataclasses import dataclass, asdict

import numpy as np
import torch


@dataclass
class Config:
    # --------------------
    # Experiment
    # --------------------
    experiment_name: str = "alignment_probe"
    save_dir: str = "./checkpoints"
    seed: int = 42

    # --------------------
    # Training
    # --------------------
    num_epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 5e-3 #5e-3 #5e-3 #3e-4
    weight_decay: float = 1e-5 #1e-5 #1e-4

    # --------------------
    # Scheduler
    # --------------------
    lr_factor: float = 0.75 #0.5 #0.75
    lr_patience: int = 5 #3 #2

    # --------------------
    # Evaluation
    # --------------------
    boundary_threshold: float = 0.5

    # --------------------
    # Data
    # --------------------
    num_workers: int = 4

    @classmethod
    def from_args(cls):
        parser = argparse.ArgumentParser(description="Probe Training Config")

        # Experiment
        parser.add_argument("--experiment_name", type=str, default=cls.experiment_name)
        parser.add_argument("--save_dir", type=str, default=cls.save_dir)
        parser.add_argument("--seed", type=int, default=cls.seed)

        # Training
        parser.add_argument("--num_epochs", type=int, default=cls.num_epochs)
        parser.add_argument("--batch_size", type=int, default=cls.batch_size)
        parser.add_argument("--learning_rate", type=float, default=cls.learning_rate)
        parser.add_argument("--weight_decay", type=float, default=cls.weight_decay)

        # Scheduler
        parser.add_argument("--lr_factor", type=float, default=cls.lr_factor)
        parser.add_argument("--lr_patience", type=int, default=cls.lr_patience)

        # Evaluation
        parser.add_argument("--boundary_threshold", type=float, default=cls.boundary_threshold)

        # Data
        parser.add_argument("--num_workers", type=int, default=cls.num_workers)

        args = parser.parse_args()
        return cls(**vars(args))

    def setup_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

    def make_save_dir(self):
        path = os.path.join(self.save_dir, self.experiment_name)
        os.makedirs(path, exist_ok=True)
        self.save_dir = path

    def save(self):
        with open(os.path.join(self.save_dir, "config.json"), "w") as f:
            json.dump(asdict(self), f, indent=4)
