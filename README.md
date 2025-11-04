# Fork of "Lyapunov-stable Neural Control for State and Output Feedback: A Novel Formulation"  For User-Defined Dynamics
## CS 7268 Group Project, Fall 2025

### Introduction
**The paper this repository fork is based off of:**

*Lujie Yang\*, Hongkai Dai\*, Zhouxing Shi, Cho-Jui Hsieh, Russ Tedrake, and Huan Zhang*
"[Lyapunov-stable Neural Control for State and Output Feedback: A Novel Formulation](https://arxiv.org/pdf/2404.07956.pdf)" (\*Equal contribution)

The goal of this project (for Northeastern University's CS 7268: Verifiable Machine Learning Class, Fall 2025) is to extend the code developed for the above paper to allow user-specified nonlinear dynamical systems as opposed to the 4 systems hard-coded in [path_tracking.py](https://github.com/gbenezer/Lyapunov_Stable_NN_Controllers_Custom_Dynamics/blob/main/neural_lyapunov_training/path_tracking.py), [pendulum.py](https://github.com/gbenezer/Lyapunov_Stable_NN_Controllers_Custom_Dynamics/blob/main/neural_lyapunov_training/pendulum.py), [pvtol.py](https://github.com/gbenezer/Lyapunov_Stable_NN_Controllers_Custom_Dynamics/blob/main/neural_lyapunov_training/pvtol.py), and [quadrotor_2d.py](https://github.com/gbenezer/Lyapunov_Stable_NN_Controllers_Custom_Dynamics/blob/main/neural_lyapunov_training/quadrotor_2d.py). The initial evaluation of this approach will be comparing regions of attraction and controller performance for these four systems trained and verified with the hard-coded dynamics to those trained and verified with user specified dynamics equations. The second part of evaluation is to assess how this approach scales in terms of performance as a function of state space dimensionality and/or controller and Lyapunov function network size and depth.

### Modifications

TODO: Describe modifications to repository

### Example Usage

TODO: Go through examples of how to use the interface

### Performance Evaluation

#### Reproduction of Results with Hard-Coded Dynamics

TODO: Add results of testing

#### Assessment of Approach Scalability/Drawbacks

TODO: Add results of varying state space dimensionality, controller size, observer size, etc.
