import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import itertools


import neural_lyapunov_training.output_train_utils as output_train_utils
import neural_lyapunov_training.lyapunov as lyapunov
import neural_lyapunov_training.controllers as controllers
import neural_lyapunov_training.quadrotor2d as quadrotor2d
import neural_lyapunov_training.train_utils as train_utils

import neural_lyapunov_training.lyapunov_roa_visualization as lrv
import neural_lyapunov_training.roa_metrics as rmet
import neural_lyapunov_training.symbolic_dynamics as sd
import neural_lyapunov_training.symbolic_systems as ss

import wandb
import os

device = torch.device("cuda")
dtype = torch.float


@hydra.main(config_path="./config", config_name="quadrotor2d_output_training")
def main(cfg: DictConfig):
    OmegaConf.save(cfg, os.path.join(os.getcwd(), "config.yaml"))
    train_utils.set_seed(cfg.seed)

    quadrotor_continuous = ss.SymbolicQuadrotor2D()
    dt = cfg.model.dt
    dynamics = sd.GenericDiscreteTimeSystem(
        quadrotor_continuous, 
        dt, 
        integration_method = sd.IntegrationMethod["RK4"],
        position_integration = sd.IntegrationMethod["RK4"],
    )
    dynamics.to(device)

    nx = quadrotor_continuous.nx
    x_max = torch.tensor([1, np.pi / 2, 2, 2 * np.pi], device=device)
    e_max = x_max / 2
    limit_xe = torch.concat((x_max, e_max))
    limit_scale = cfg.model.limit_scale
    limit = limit_scale * limit_xe
    grid_size = torch.tensor([4, 6, 4, 8, 2, 3, 2, 3], device=device)
    lower_limit = -limit
    upper_limit = limit

    h = lambda x: quadrotor_continuous.h(x)

    x0 = (dynamics.x_equilibrium).to(device)
    controller = controllers.NeuralNetworkController(
        nlayer=2,
        in_dim=nx + quadrotor_continuous.ny,
        out_dim=2,
        hidden_dim=8,
        clip_output="clamp",
        u_lo=torch.tensor([0, 0.0], device=device),
        u_up=(dynamics.u_equilibrium * 3).to(device),
        x_equilibrium=torch.cat(
            (x0, torch.zeros(quadrotor_continuous.ny, device=device))
        ),
        u_equilibrium=(dynamics.u_equilibrium).to(device),
    )
    controller.to(device)
    controller.load_state_dict(
        torch.load(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "data/quadrotor2d/output_feedback/controller_[8, 8].pth",
            )
        )
    )
    controller.eval()