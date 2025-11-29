# Fork of "Lyapunov-stable Neural Control for State and Output Feedback: A Novel Formulation"  For User-Defined Dynamics
# CS 7268 Group Project, Fall 2025

## Introduction
**The paper this repository fork is based off of:**

*Lujie Yang\*, Hongkai Dai\*, Zhouxing Shi, Cho-Jui Hsieh, Russ Tedrake, and Huan Zhang*
"[Lyapunov-stable Neural Control for State and Output Feedback: A Novel Formulation](https://arxiv.org/pdf/2404.07956.pdf)" (\*Equal contribution)

The goal of this project (for Northeastern University's CS 7268: Verifiable Machine Learning Class, Fall 2025) is to extend the code developed for the above paper to allow user-specified nonlinear dynamical systems as opposed to the 4 systems hard-coded in [path_tracking.py](https://github.com/gbenezer/Lyapunov_Stable_NN_Controllers_Custom_Dynamics/blob/main/neural_lyapunov_training/path_tracking.py), [pendulum.py](https://github.com/gbenezer/Lyapunov_Stable_NN_Controllers_Custom_Dynamics/blob/main/neural_lyapunov_training/pendulum.py), [pvtol.py](https://github.com/gbenezer/Lyapunov_Stable_NN_Controllers_Custom_Dynamics/blob/main/neural_lyapunov_training/pvtol.py), and [quadrotor2d.py](https://github.com/gbenezer/Lyapunov_Stable_NN_Controllers_Custom_Dynamics/blob/main/neural_lyapunov_training/quadrotor2d.py). The initial evaluation of this approach will be comparing regions of attraction and controller performance for these four systems trained and verified with the hard-coded dynamics to those trained and verified with user specified dynamics equations. The second part of evaluation is to assess how this approach scales in terms of performance as a function of state space dimensionality and/or controller and Lyapunov function network size and depth.

## Installation

Create a conda environment and install the dependencies except those for verification:
```bash
conda create --name lyapunov_neural_control python=3.11
conda activate lyapunov_neural_control
pip install -r original_requirements.txt
```

The original repository uses [auto_LiRPA](https://github.com/Verified-Intelligence/auto_LiRPA.git) and [alpha-beta-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN.git) for verification. To install both of them, run:
```bash
git clone --recursive https://github.com/Verified-Intelligence/alpha-beta-CROWN.git
(cd alpha-beta-CROWN/auto_LiRPA && pip install -e .)
(cd alpha-beta-CROWN/complete_verifier && pip install -r requirements.txt)
```

To install the required files for this modified directory, execute the following code
after the above two code blocks ```

```bash
pip install -r requirements.txt
```

To set up the path:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/alpha-beta-CROWN:$(pwd)/alpha-beta-CROWN/complete_verifier"
```

This path setup line needs to be executed every time a new command prompt is opened, along with the activation of the virtual environment, for the code to work properly.

## Training

### Original Documentation

```
python examples/pendulum_state_training.py
python examples/pendulum_output_training.py
python examples/path_tracking_state_training.py
python examples/quadrotor2d_state_training.py
python examples/quadrotor2d_output_training.py
```

All the training files provide an estimate of the sublevel set value $\hat \rho_{\text{max}}$, figures of ROA slices and $V(x_t)$ along simulated trajectories after the training ends.

We use `hydra` to manage all the configurations. Take pendulum state feedback training as an example, it loads the configuration file in `examples/config/pendulum_state_training.yaml` file for all the parameters. To set your specific parameters, we recommend adding the config file in examples/config/user/USERNAME.yaml, and then run the command
```
python examples/pendulum_state_training.py user=USERNAME
```
You can pattern match `examples/config/user/pendulum_state_training_default.yaml` file to set your own `USERNAME.yaml`.

To reproduce the pendulum state traning result in our paper, please use the following command with a non-default config file

```
python examples/pendulum_state_training.py --config-name pendulum_state_training_reproduce
```

Note that each run will create a directory, specified in `user.run_dir` in the configuration yaml file. The directory will contain the learned model, wandb data, and the configuration file `config.yaml` used for that run, so that you can easily reproduce the result with the saved configuration file.

You can change `cfg.model.limit_scale` to increase the region for training. We recommend starting from 0.1 and gradually grow the limit scale to 1.0, using models from smaller regions as initializations for larger regions. You can also change `cfg.model.rho_multiplier` to encourage the growth of the sublevel set.

Before the end of the training procedure, the program will first test the trained models using projected gradient descent (PGD). Usually, PGD won't be able to find any counterexamples at the end of training, so the models are ready for formal verification. In addition, `rho` used during training will be printed out, which can be used as the `--init_rho` for the bisection in the verification step.

```
[2024-04-11 00:42:36,757][__main__][INFO] - PGD verifier finds counter examples? False
rho =  0.12872669100761414
```

If PGD attacks can still find counterexamples after training, you should set `cfg.train.train_lyaloss = False` and decrease the last entry of `cfg.model.rho_multiplier` until PGD attack can no longer find counterexamples to obtain a more accurate $\hat \rho_{\text{max}}$.

### Our Modified Files

Our modified files are
```
python examples/pendulum_state_training_symbolic.py
python examples/pendulum_output_training_symbolic.py
python examples/path_tracking_state_training_symbolic.py
python examples/quadrotor2d_state_training_symbolic.py
python examples/quadrotor2d_output_training_symbolic.py
```

These files use dynamical systems defined using our framework in `neural_lyapunov_training/symbolic_dynamics.py`, all of which are found in `neural_lyapunov_training/symbolic_systems.py`. The original files and these files also have been modified to generate 2D and 3D interactive HTML plots of the Lyapunov functions along with the Lyapunov derivative under the controller that has been trained. The files also generate a set of metrics from numerical integration of the ROA inner approximation that was found along with the region where the Lyapunov derivative is non-positive.

## Modifications

The main modifications in this fork are contained within the following files:

1. `neural_lyapunov_training/symbolic_dynamics.py`
    - `SymbolicDynamicalSystem`
        - Main superclass for user definition of arbitrary continous-time dynamical systems
            - `__init__` must construct a field defining the order of the dynamical system `self.order` and a field that calls and stores `self.define_system`
            - `define_system` is a method that should take `self` along with the constant numerical parameters of the system (e.g. gravitational constant, damping terms, etc.) and
                - Define a SymPy symbol for each state, control, and parameter variable of the system
                - Construct a Dictionary `self.parameters` with SymPy Symbol keys associating parameters to the values passed into the function
                - Construct lists `self.state_vars`, `self.control_vars`, `self.output_vars` containing the defined symbolic variables (`self.output_vars` is optional for full state feedback)
                - Define the functional form of the open-loop dynamical system $\frac{d}{dt}[x] = f(x, u)$ as a symbolic column vector in the private field `self._f_sym`
                - (Optional) Define the equilibrium state and equilibrium control action for the dynamical system using `self.x_equilibrium` and `self.u_equilibrium` (defaults to zero vectors of appropriate length). Framework is currently limited to storing only one equilibrium, but certain methods can be passed equilibria separately. 
                - (Optional) Define the observation function $y = h(x)$ as a symbolic column vector in the private field `self._h_sym` (defaults to full state observability)
        - After definition and instantiation, the following public methods are available
            - `forward` evaluates the nonlinear dynamics given a state and control, or a batch of them, and `h` does the same for observations
            - Generate NumPy or PyTorch-compatible numerical functions with `generate_numpy_function` or `generate_torch_function`
            - `linearized_dynamics_symbolic`, `linearized_observation_symbolic`, `linearized_dynamics`, `linearized_observation` symbolically and numerically evaluate linearizations of dynamics and observations
            - Check if the system's state and control equilibria are actually equilibria, evaluate eigenvalues of the linearized systems at the equilibrium, and check if the equilibrium is stable with `check_equilibrium`, `eigenvalues_at_equilibrium`, and `is_stable_equilibrium` respectively
            - Calculate Linear-Quadratic Regulator control gain or Kalman filter gain for the continuous-time system with `lqr_control` and `kalman_gain`
    - `GenericDiscreteTimeSystem`
        - Wrapper class for the discretization of a continuous time nonlinear dynamical system
        - User passes a `SymbolicDynamicalSystem` object along with a discretization time scale and an integration method from "ExplicitEuler", "Midpoint", and "RK4" (and optionally a different integration method for position for systems higher than first-order) to the `GenericDiscreteTimeSystem` constructor method
        - This object then allows a user to
            - Compute the next state from a given initial state and control input with `forward`, or fully simulate one (or multiple) trajectories using `simulate`
            - Calculate Linear-Quadratic Regulator control gain or Kalman filter gain for the discrete-time system with `dlqr_control` and `discrete_kalman_gain`
            - Plot trajectories and phase portraits using 