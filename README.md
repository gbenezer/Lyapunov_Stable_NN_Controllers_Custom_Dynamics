# Dynamics Specification for Neural Control Lyapunov Function and Neural Controller Training

**CS 7268 Group Project, Fall 2025**  
*Northeastern University - Verifiable Machine Learning*

This repository fork extends the work from the paper:

*Lujie Yang\*, Hongkai Dai\*, Zhouxing Shi, Cho-Jui Hsieh, Russ Tedrake, and Huan Zhang*  
"[Lyapunov-stable Neural Control for State and Output Feedback: A Novel Formulation](https://arxiv.org/pdf/2404.07956.pdf)" (\*Equal contribution)

## Project Overview

The original repository demonstrated neural network controllers and observers with provable Lyapunov stability guarantees for four hard-coded dynamical systems (pendulum, path tracking, PVTOL, and quadrotor). Our extension introduces a **symbolic dynamics framework** that allows users to define arbitrary nonlinear dynamical systems without modifying the core training and verification pipeline.

### Key Contributions

* **Symbolic dynamics framework** enabling user-defined continuous-time nonlinear systems through SymPy
* **Automatic discretization** with multiple integration methods (Explicit Euler, Midpoint, RK4)
* **Generalized training pipeline** that works with arbitrary system definitions
* **Enhanced visualization** with interactive 2D and 3D HTML plots of Lyapunov functions and derivatives
* **Performance benchmarking** comparing symbolic vs. hard-coded implementations across system complexity

### Features Inherited from Original Work

* Novel formulation defining larger verifiable region-of-attraction (ROA) than prior work
* Training framework for neural network controllers/observers with Lyapunov certificates
* No reliance on expensive solvers (MIP, SMT, SDP) during training or verification
* Support for both state-feedback and output-feedback control
* Formal verification using auto_LiRPA and alpha-beta-CROWN

## Installation

### 1. Create Conda Environment and Install Verification Tools

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

### 2. Set Up Python Path

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/alpha-beta-CROWN:$(pwd)/alpha-beta-CROWN/complete_verifier"
```

**Note:** This path setup must be executed each time you open a new terminal session.

## Quick Start

### Training with Symbolic Dynamics

Train controllers using our symbolic dynamics framework:

```bash
# Pendulum state feedback
python examples/pendulum_state_training_symbolic.py

# Pendulum output feedback  
python examples/pendulum_output_training_symbolic.py

# Path tracking
python examples/path_tracking_state_training_symbolic.py

# Quadrotor 2D state feedback
python examples/quadrotor2d_state_training_symbolic.py

# Quadrotor 2D output feedback
python examples/quadrotor2d_output_training_symbolic.py
```

### Training with Original Hard-coded Dynamics

For comparison with the original implementation:

```bash
python examples/pendulum_state_training.py
python examples/pendulum_output_training.py
python examples/path_tracking_state_training.py
python examples/quadrotor2d_state_training.py
python examples/quadrotor2d_output_training.py
```

## Symbolic Dynamics Framework

### Core Components

The symbolic dynamics framework is implemented in several key files:

* `neural_lyapunov_training/symbolic_dynamics.py` - Core framework classes
* `neural_lyapunov_training/symbolic_systems.py` - Example system definitions (15+ systems)
* `neural_lyapunov_training/roa_metrics.py` - Region of Attraction quantification
* `neural_lyapunov_training/lyapunov_roa_visualization.py` - Interactive visualization tools

### Defining a Custom System

To define your own nonlinear dynamical system, subclass `SymbolicDynamicalSystem`:

```python
from sympy import symbols, Matrix, sin, cos, pi
import torch

class InvertedPendulumWithOffset(SymbolicDynamicalSystem):
    """
    Second order inverted pendulum with non-zero equilibrium and partial observation
    
    Equilibrium at θ = π/4 (45 degrees from vertical) with gravity compensation
    Observes only angular position, not velocity
    """
    def __init__(self, m=1.0, l=0.5, g=9.81, beta=0.1):
        super().__init__()
        # Store parameter values for equilibrium computation
        self.m_val = m
        self.l_val = l
        self.g_val = g
        self.beta_val = beta

        # Essential portion for initialization method
        self.order = 2
        self.define_system(m, l, g, beta)
    
    def define_system(self, m_val, l_val, g_val, beta_val):
        # State variables
        theta, theta_dot = symbols('theta theta_dot', real=True)
        tau = symbols('tau', real=True)
        
        # Parameters
        m, l, g, beta = symbols('m l g beta', real=True, positive=True)
        
        self.parameters = {m: m_val, l: l_val, g: g_val, beta: beta_val}
        self.state_vars = [theta, theta_dot]
        self.control_vars = [tau]
        self.output_vars = [theta]  # Only angle is measured
        
        # Dynamics
        I = m * l * l
        theta_ddot = -(beta / I) * theta_dot + (g / l) * sin(theta) + tau / I
        
        self._f_sym = Matrix([theta_ddot])  # Second-order: return only acceleration
        self._h_sym = Matrix([theta])  # Observe angle only (output feedback)
    
    @property
    def x_equilibrium(self) -> torch.Tensor:
        """Override equilibrium: balance at 45 degrees from vertical"""
        theta_eq = torch.tensor(torch.pi / 4)  # 45 degrees
        theta_dot_eq = torch.tensor(0.0)
        return torch.tensor([theta_eq, theta_dot_eq])
    
    @property
    def u_equilibrium(self) -> torch.Tensor:
        """Torque needed to maintain 45-degree equilibrium (gravity compensation)"""
        # At equilibrium: τ = -(g/l)*sin(θ_eq) * I
        import numpy as np
        theta_eq = np.pi / 4
        I = self.m_val * self.l_val * self.l_val
        tau_eq = -(self.g_val / self.l_val) * np.sin(theta_eq) * I
        return torch.tensor([tau_eq])
```

### Using Your Custom System

```python
# Create system instance
system = InvertedPendulumWithOffset(m=0.15, l=0.5, g=9.81, beta=0.1)

# Verify custom equilibrium
print(f"Equilibrium state: {system.x_equilibrium}")
print(f"Equilibrium control: {system.u_equilibrium}")

is_eq, max_deriv = system.check_equilibrium(
    system.x_equilibrium,
    system.u_equilibrium
)
print(f"Is equilibrium valid? {is_eq}, max derivative: {max_deriv:.6f}")

# Check if we need an observer (partial observability)
print(f"State dimension (nx): {system.nx}")
print(f"Output dimension (ny): {system.ny}")
print(f"Need observer: {system.ny < system.nx}")

# Create discrete-time version
from neural_lyapunov_training.symbolic_dynamics import GenericDiscreteTimeSystem

dt = 0.01  # Time step
discrete_system = GenericDiscreteTimeSystem(
    system, 
    dt, 
    integration_method="RK4"
)

# Simulate trajectory
import torch
x0 = torch.tensor([[0.5, 0.0]])  # Initial state
u_trajectory = torch.zeros(100, 1)  # Control sequence
trajectory = discrete_system.simulate(x0, u_trajectory)

# Compute LQR controller
Q = torch.eye(2)  # State cost
R = torch.eye(1)  # Control cost
K = discrete_system.dlqr_control(Q, R)
```

### Available Methods

**SymbolicDynamicalSystem Methods:**
* `forward(x, u)` - Evaluate continuous-time dynamics dx/dt = f(x, u)
* `h(x)` - Evaluate observation function y = h(x)
* `generate_numpy_function()` - Generate NumPy-compatible numerical function
* `generate_torch_function()` - Generate PyTorch-compatible numerical function (automatic differentiation compatible)
* `linearized_dynamics(x_eq, u_eq)` - Compute linearization A = ∂f/∂x, B = ∂f/∂u at equilibrium
* `linearized_observation(x_eq)` - Compute observation Jacobian C = ∂h/∂x
* `check_equilibrium()` - Verify equilibrium correctness
* `eigenvalues_at_equilibrium()` - Compute eigenvalues at linearized equilibrium
* `is_stable_equilibrium()` - Check equilibrium stability (continuous or discrete)
* `lqr_control(Q, R)` - Compute continuous-time LQR gain
* `kalman_gain(Q_process, R_measurement)` - Compute continuous-time Kalman filter gain
* `lqg_control(Q_lqr, R_lqr, Q_process, R_measurement)` - Design combined LQR controller + Kalman observer
* `lqg_closed_loop_matrix(K, L)` - Compute closed-loop system matrix for stability analysis
* `verify_jacobians(x, u)` - Verify symbolic Jacobians against numerical autodiff (debugging tool)
* `check_numerical_stability(x, u)` - Check for NaN, Inf, extreme values
* `print_equations()` - Display symbolic equations in human-readable format
* `get_performance_stats()` - Get timing and call count statistics
* `save_config(filename)` - Save system configuration to file

**GenericDiscreteTimeSystem Methods:**
* `forward(x, u)` - Compute next state x[k+1] = discrete_dynamics(x[k], u[k])
* `simulate(x0, controller, horizon)` - Simulate trajectory from initial state with various controller types:
  - Pre-computed control sequence: `simulate(x0, controller=u_seq)`
  - Controller function: `simulate(x0, controller=π, horizon=100)`
  - Neural network controller: `simulate(x0, controller=nn_controller, horizon=100)`
  - Zero control: `simulate(x0, controller=None, horizon=100)`
  - Output feedback with observer: `simulate(x0, controller=π, observer=obs, horizon=100)`
* `linearized_dynamics(x, u)` - Compute discrete linearization Ad, Bd
* `linearized_observation(x)` - Compute observation Jacobian C = ∂h/∂x
* `h(x)` - Evaluate discrete-time observation
* `dlqr_control(Q, R)` - Compute discrete-time LQR gain
* `discrete_kalman_gain(Q_process, R_measurement)` - Compute discrete-time Kalman gain
* `dlqg_control(Q_lqr, R_lqr, Q_process, R_measurement)` - Design discrete LQG controller
* `dlqg_closed_loop_matrix(K, L)` - Discrete closed-loop system matrix
* `output_feedback_lyapunov(K, L)` - Solve discrete Lyapunov equation for stability analysis
* `print_info()` - Display comprehensive system information including linearization
* `summary()` - Get brief system summary string
* `plot_trajectory(trajectory, controls)` - Interactive Plotly visualization with adaptive sizing:
  - Automatic subplot layout optimization (1-12+ variables)
  - Adaptive figure dimensions based on content
  - Compact mode for many variables
  - Batch trajectory support with color coding
  - Control sequence overlay
  - Customizable color schemes (Plotly, D3, Vivid, etc.)
  - Interactive HTML export
* `plot_trajectory_3d(trajectory, state_indices)` - 3D trajectory (time-colored paths for single trajectory, otherwise solid)
* `plot_phase_portrait_2d(trajectory, state_indices)` - 2D phase space visualization
* `plot_phase_portrait_3d(trajectory, state_indices)` - 3D phase space with solid colors per trajectory

### Supported Integration Methods

* **ExplicitEuler** - First-order explicit Euler (fast, less accurate)
* **Midpoint** - Second-order midpoint method (moderate speed and accuracy)
* **RK4** - Fourth-order Runge-Kutta (slower, more accurate)

Higher-order systems can use different integration methods for position vs. velocity states.

### Advanced Features

**State Estimation and Control:**
* `ExtendedKalmanFilter` - Nonlinear state estimation with re-linearization at each timestep
  - Handles nonlinear dynamics and observations
  - Automatic covariance propagation
  - Supports both continuous and discrete systems
* `LinearObserver` - Constant-gain Luenberger observer for output feedback
  - Lower computational cost than EKF
  - Suitable for locally linear behavior
* `LinearController` - State feedback controller with equilibrium offset
  - Supports both state and output feedback (with observer)
  - Handles batched inputs
  - GPU-compatible

**Helper Classes:**
All controller and observer classes support:
- `.to(device)` - Move to CPU/GPU
- `.reset()` - Reset to initial conditions
- Batched operations for efficient computation

## Configuration Management

The framework uses [Hydra](https://hydra.cc/) for configuration management. Each training script loads parameters from YAML files in `examples/config/`.

### Custom Configurations

Create a user-specific configuration:

```bash
# Create your config file
cp examples/config/user/pendulum_state_training_default.yaml examples/config/user/USERNAME.yaml

# Edit USERNAME.yaml with your parameters

# Run with your config
python examples/pendulum_state_training_symbolic.py user=USERNAME
```

### Key Configuration Parameters

* `cfg.model.limit_scale` - Training region size (start at 0.1, grow to 1.0)
* `cfg.model.rho_multiplier` - Encourages sublevel set growth
* `cfg.train.train_lyaloss` - Enable/disable Lyapunov loss during training
* `user.run_dir` - Output directory for models and results

### Output Structure

Each training run creates a directory containing:
* Trained neural network models
* WandB logging data
* Configuration file (`config.yaml`) for reproducibility
* Interactive HTML visualizations
* ROA metrics and statistics

## Training Details

### Training Process

During training, the framework:
1. Learns neural network controller and Lyapunov function jointly
2. Uses fast empirical falsification with PGD attacks
3. Applies strategic regularization for stability
4. Generates interactive visualizations of ROA and Lyapunov surfaces
5. Computes numerical integration metrics for ROA characterization

### Enhanced Training Scripts

The symbolic training scripts provide comprehensive output beyond the original implementation:

**Automatic Visualization Generation:**
- Interactive 2D plots: Lyapunov function and derivative contours
- Interactive 3D plots: Surface visualizations of V(x) and ΔV(x)
- Saved as standalone HTML files for easy sharing

**Quantitative ROA Metrics:**
- Multiple sampling methods possible (Sobol QMC used by default)
- Area/volume estimates
- Lyapunov difference statistics (ΔV)
- Verification percentage within ROA

### Training Output

* `PGD finds counter examples? False` - Model ready for formal verification
* `rho` value - Use as `--init_rho` for verification bisection

If PGD still finds counterexamples:
1. Set `cfg.train.train_lyaloss = False`
2. Decrease last entry of `cfg.model.rho_multiplier`
3. Retrain until PGD finds no counterexamples

### Progressive Training Strategy

For large regions of attraction:
1. Start with `cfg.model.limit_scale = 0.1`
2. Train until convergence
3. Increase `limit_scale` gradually (e.g., 0.3, 0.5, 1.0)
4. Use previous model as initialization for larger region
5. Verify at each stage

## Verification

### Using Pre-trained Models

Verify the provided pre-trained models:

```bash
cd verification
export CONFIG_PATH=$(pwd)
cd complete_verifier

# Verify each system
python abcrown.py --config $CONFIG_PATH/pendulum_state_feedback_lyapunov_in_levelset.yaml
python abcrown.py --config $CONFIG_PATH/pendulum_output_feedback_lyapunov_in_levelset.yaml
python abcrown.py --config $CONFIG_PATH/path_tracking_state_feedback_lyapunov_in_levelset.yaml
python abcrown.py --config $CONFIG_PATH/quadrotor2d_state_feedback_lyapunov_in_levelset.yaml
python abcrown.py --config $CONFIG_PATH/quadrotor2d_output_feedback_lyapunov_in_levelset.yaml
```

### Verification Output

The verification will output a summary of results. For example, here are the
results the original authors obtained on Pendulum Output Feedback using `pendulum_output_feedback_lyapunov_in_levelset.yaml`:

```
############# Summary #############
Final verified acc: 100.0% (total 8 examples)
Problem instances count: 8 , total verified (safe/unsat): 8 , total falsified (unsafe/sat): 0 , timeout: 0
mean time for ALL instances (total 8):12.023795893354652, max time: 23.111693859100342
mean time for verified SAFE instances(total 8): 12.023810923099518, max time: 23.111693859100342
safe (total 8), index: [0, 1, 2, 3, 4, 5, 6, 7]
```

### Memory Requirements

Original author verification configurations were tested on a GPU with 48GB memory.
If you are using a GPU with less memory, you may decrease the batch size
of verification by modifying the `batch_size` item in the configuration files
or passing an argument `--batch_size BATCH_SIZE`,
until it fits into the GPU memory.

```bash
# Reduce batch size in config file or via argument
python abcrown.py --config CONFIG_FILE.yaml --batch_size 32
```

## Verifying New Models

### Step 1: Bisection for Optimal ρ

You can use bisection to find the largest $\rho$ that satisfies the verification objective.
You can use the script `neural_lyapunov_training/rho_bisect.py` for automatic bisection.

You need to specify the region for verification using `--lower_limit`, `--upper_limit`, and `--hole_size`.
The `--lower_limit` and `--upper_limit` define the region of interests $\mathcal{B}$ (which is problem specific; see Table 3 in the Appendix of original author paper). The `--hole_size` excludes a very small region (default 0.1%) around the origin, which the verifier may have numerical issues with since the Lyapunov function values are very close to 0.

You provide an initial $\rho$ value by `--init_rho`
and specify the precision for the bisection by `--rho_eps`.
`--init_rho` is the initial guess of sublevel set value, and you can use the `rho` reported after training finishes.

For verification during the bisection, a configuration file needs to be specified
by `--config` and a timeout value is needed by `--timeout`.

Optionally, an `--output_folder` argument may be used to specify an output folder the bisection.
Additional arguments for the verifier may also be provided.

For models which may take a long time to verify, you may add `--check_x_adv_only`
to only check $\xi_{t+1}\in\mathcal{B}$ but not $-F(\xi_t)>0$
(Theorem 3.3 in original paper) during the bisection.

In case that the $\rho$ you obtain from the bisection does not lead to a
successful verification when you generate full specifications using the $\rho$
(see the next sections), you may further reduce $\rho$ manually.

The below is an example for the author hard-coded pendulum full state feedback system:

```bash
cd verification
python -m ROOT/neural_lyapunov_training/rho_bisect.py \ 
    --lower_limit -12 -12 \
    --upper_limit 12 12 \
    --hole_size 0.001 \
    --init_rho 603.5202 \
    --rho_eps 0.1 \
    --config pendulum_state_feedback_lyapunov_in_levelset.yaml \
    --timeout 100
```

The bisection will output the result of each bisection iteration. For example:

```txt
Generating specs with rho=603.5202
Start verification
Output path: ./output/rho_603.52020.txt
Result: defaultdict(<class 'list'>, {'safe': [0, 1, 2, 3]})
safe
```

In the end, it will output the lower bound and upper bound for $\rho$ as:
```txt
rho_l=708.0606252685546
rho_u=708.1342971679687
```

You can take the lower bound of $\rho$ denoted as `rho_l` above. Note that if you are _not_ using `--check_x_adv_only`, then the model is verified with the sublevel set value `rho_l`; in this case, the next step is not necessary, although it can still be helpful to save the specifications with this specific `rho_l` for reproducing the verification results without bisection.


### Step 2: Generate VNNLIB Specifications

After obtaining a suitable $\rho$ value,
you can generate the specifications for verification with a fixed $\rho$.
You should use VNNLIB format files to describe the specification.

The command below will generate several VNNLIB files and a CSV file with a list of
all VNNLIB filenames. Each VNNLIB file describes a subproblem to verify, and
the number of subproblems is determined by the state dimension.

Similar to the aforementioned bisection, `--lower_limit`, `--upper_limit`, and `--hole_size` need to be specified. And the sublevel set value $\rho$ is provided by `--value_levelset`.
All specification files will be saved in the `specs` folder.

Again, this is an example for the pendulum full state feedback system.

```bash
cd verification

python -m ROOT/neural_lyapunov_training/generate_vnnlib \
    --lower_limit -12 -12 \
    --upper_limit 12 12 \
    --hole_size 0.001 \
    --value_levelset 672 \
    specs/pendulum_state_feedback
```

This generates VNNLIB files in `specs/` directory, which can then be verified using alpha-beta-CROWN.

## Region of Attraction (ROA) Analysis

The framework provides comprehensive tools for quantifying and visualizing Regions of Attraction.

### ROA Metrics (`roa_metrics.py`)

**Sampling Methods:**

1. **Monte Carlo (Random Sampling)**
   ```python
   from neural_lyapunov_training.roa_metrics import compute_roa_area_monte_carlo
   
   metrics = compute_roa_area_monte_carlo(
       lyapunov_nn, state_limits, rho, 
       num_samples=100000, device='cuda'
   )
   ```
   - Standard random sampling
   - Convergence: O(1/√N)
   - Best for quick estimates

2. **Quasi-Monte Carlo - Sobol Sequence**
   ```python
   from neural_lyapunov_training.roa_metrics import compute_roa_area_qmc_sobol
   
   metrics = compute_roa_area_qmc_sobol(
       lyapunov_nn, state_limits, rho,
       num_samples=10000,  # 10-100x fewer samples for same accuracy
       compute_discrepancy_metric=True,
       round_to_pow2=True  # Optimal Sobol properties
   )
   ```
   - Low-discrepancy sequence for better coverage
   - Convergence: O(1/N) - much faster than random
   - Typically requires 10-100x fewer samples
   - Optimal when N is power of 2
   - Best for dimensions ≤ 20

3. **Quasi-Monte Carlo - Halton Sequence**
   ```python
   from neural_lyapunov_training.roa_metrics import compute_roa_area_qmc_halton
   
   metrics = compute_roa_area_qmc_halton(
       lyapunov_nn, state_limits, rho,
       num_samples=10000,
       compute_discrepancy_metric=True
   )
   ```
   - Alternative QMC sequence
   - Best for lower dimensions (d ≤ 10)
   - Doesn't require power-of-2 sample count

4. **Grid-Based Method (2D Only)**
   ```python
   from neural_lyapunov_training.roa_metrics import compute_roa_area_grid
   
   metrics = compute_roa_area_grid(
       lyapunov_nn, state_limits, rho,
       grid_resolution=200,
       state_indices=(0, 1)
   )
   ```
   - Most accurate for 2D visualization
   - Deterministic results
   - Not practical for d > 3

**Discrete-Time Lyapunov Difference Analysis:**

For discrete-time systems, compute both ROA (V ≤ ρ) AND stability region (ΔV ≤ 0):

```python
from neural_lyapunov_training.roa_metrics import (
    compute_lyapunov_difference_metrics_monte_carlo,
    compute_lyapunov_difference_metrics_qmc_sobol,
    print_lyapunov_difference_metrics
)

# Compute with actual closed-loop dynamics
metrics = compute_lyapunov_difference_metrics_qmc_sobol(
    lyapunov_nn, controller_nn, dynamics_system,
    state_limits, rho,
    num_samples=100000,
    observer_nn=observer_nn,  # Optional for output feedback
    state_indices=(0, 1),
    stability_threshold=0.0
)

# Comprehensive output includes:
# - ROA volume (V ≤ ρ)
# - Decreasing region volume (ΔV ≤ 0) 
# - Verified ROA volume (both conditions)
# - ΔV statistics (mean, max, std)
# - Violation analysis
print_lyapunov_difference_metrics(metrics, title="System Performance")
```

**Output Feedback Support:**

For systems with observers, automatically handles augmented state [x, e]:
- Samples over PHYSICAL state space only
- Assumes converged observer (e=0) for ideal behavior analysis
- Correctly augments states internally for Lyapunov evaluation
- Computes closed-loop ΔV using controller and observer

**Metrics Objects:**

`ROAMetrics` contains:
- `area_roa` - Estimated ROA volume/area
- `area_domain` - Total domain volume
- `coverage_ratio` - Fraction of domain in ROA
- `num_samples_in_roa` - Sample count in ROA
- `method` - Sampling method used
- `discrepancy` - Uniformity metric (QMC only)

`LyapunovDifferenceMetrics` contains:
- All `ROAMetrics` fields plus:
- `area_decreasing` - Volume where ΔV ≤ 0
- `area_verified_roa` - Volume satisfying both conditions
- `mean_delta_V_in_roa` - Average Lyapunov decrease
- `max_violation_in_roa` - Worst ΔV violation
- `percent_verified` - % of ROA that's stable

## Visualization (`lyapunov_roa_visualization.py`)

The framework provides comprehensive interactive visualization tools using Plotly.

### 2D Visualizations

**Lyapunov Function and Derivative:**
```python
from neural_lyapunov_training.lyapunov_roa_visualization import plot_lyapunov_2d

fig = plot_lyapunov_2d(
    lyapunov_nn, controller_nn, dynamics_system,
    state_limits=((-2, 2), (-2, 2)),
    state_indices=(0, 1),
    state_names=('θ', 'θ̇'),
    rho=0.5,
    grid_resolution=100,
    observer_nn=observer_nn,  # Optional for output feedback
    trajectories=[traj1, traj2],  # Overlay trajectories
    save_html='lyapunov_analysis.html',
    colorscale='Viridis',
    trajectory_colorscale='Plotly'  # Or 'D3', 'Vivid', 'Dark24', etc.
)
```

Features:
- Side-by-side V(x) and ΔV(x) contour plots
- ROA boundary (V = ρ) highlighted in red
- Stability boundary (ΔV = 0) shown as dashed line
- Equilibrium point marked
- Trajectory overlays with customizable colors
- **Automatic output feedback handling** - correctly augments states for observer-based systems
- Interactive hover information
- Export to HTML for sharing

**Output Feedback - Estimation Error Analysis (in beta, not fully tested or evaluated):**
```python
from neural_lyapunov_training.lyapunov_roa_visualization import plot_lyapunov_2d_error_slices

fig = plot_lyapunov_2d_error_slices(
    lyapunov_nn, controller_nn, dynamics_system, observer_nn,
    state_limits=((-2, 2), (-2, 2)),
    error_values=[0.0, 0.1, 0.2, 0.5],  # Different error magnitudes
    error_dim=0,  # Which error dimension to vary
    state_indices=(0, 1),
    rho=0.5,
    save_html='error_sensitivity.html'
)
```

Shows impact of estimation error on:
- V([x, e]) contours
- ΔV([x, e]) with actual closed-loop dynamics
- Verified ROA (where both V ≤ ρ AND ΔV ≤ 0)
- Multiple slices for different error values

### 3D Surface Visualizations

**Single Lyapunov Surface:**
```python
from neural_lyapunov_training.lyapunov_roa_visualization import plot_lyapunov_3d_surface

fig = plot_lyapunov_3d_surface(
    lyapunov_nn,
    state_limits=((-2, 2), (-2, 2)),
    state_indices=(0, 1),
    rho=0.5,
    grid_resolution=80,
    colorscale='Viridis',
    trajectories=[traj1, traj2],
    trajectory_colorscale='Plotly',
    save_html='lyapunov_3d.html'
)
```

**Dual Surface (V and ΔV):**
```python
fig = plot_lyapunov_3d_surface(
    lyapunov_nn,
    state_limits=((-2, 2), (-2, 2)),
    controller_nn=controller_nn,  # Required for ΔV
    dynamics_system=dynamics_system,  # Required for ΔV
    observer_nn=observer_nn,  # Optional
    show_derivative=True,  # Enable side-by-side V and ΔV
    trajectories=[traj1, traj2],
    save_html='lyapunov_dual_3d.html'
)
```

Features:
- Interactive 3D rotation and zoom
- ROA threshold plane (V = ρ)
- Zero plane for ΔV
- Trajectories projected onto surfaces
- V(trajectory) and ΔV(trajectory) computed and overlaid
- Equilibrium point marked
- Customizable camera angles

**ROA vs Estimation Error (Output Feedback, in beta, not fully tested or evaluated):**
```python
from neural_lyapunov_training.lyapunov_roa_visualization import plot_roa_vs_error

fig = plot_roa_vs_error(
    lyapunov_nn, controller_nn, dynamics_system, observer_nn,
    state_limits=((-2, 2), (-2, 2)),
    error_range=(-0.5, 0.5),
    rho=0.5,
    state_dim=0,  # Which physical state to plot
    error_dim=0,  # Which error dimension to vary
    grid_resolution=80,
    save_html='roa_error_3d.html'
)
```

Shows:
- V([x, e]) surface over (physical_state, error) space
- ROA threshold plane
- Highlighted e=0 slice (ideal behavior)
- ΔV > 0 violations marked as scatter points

### Animations

**ROA Evolution with Estimation Error (Output Feedback, in beta, not fully tested or evaluated):**
```python
from neural_lyapunov_training.lyapunov_roa_visualization import create_roa_error_animation

fig = create_roa_error_animation(
    lyapunov_nn, controller_nn, dynamics_system, observer_nn,
    state_limits=((-2, 2), (-2, 2)),
    error_range=(-0.5, 0.5),
    error_dim=0,
    n_frames=20,
    state_indices=(0, 1),
    rho=0.5,
    grid_resolution=100,
    save_html='roa_animation.html'
)
```

Creates animated visualization showing:
- How verified ROA changes with estimation error
- V([x, e]) contours for each error value
- ROA boundary (V = ρ)
- Stability boundary (ΔV = 0)
- Verified ROA region (shaded green)
- Interactive slider and play/pause controls
- Uses actual closed-loop dynamics to compute ΔV

### Trajectory Plotting

All visualization functions support trajectory overlays with:
- Multiple trajectories with distinct colors
- Color sequence customization (Plotly, D3, Vivid, Dark24, Set1, Pastel, etc.)
- Start/end markers
- Interactive hover information showing V(x) or ΔV(x) along trajectory
- Automatic handling of batched trajectories

### Output Feedback Support

All visualization functions automatically handle output feedback systems:
- Detect observer presence automatically
- Augment physical states with estimation error internally
- Default to e=0 (ideal behavior) for visualization
- Allow custom estimation error specification (assumed constant)
- Compute closed-loop ΔV using controller and observer
- Display appropriate labels and legends

### Customization Options

All plotting functions support:
- Custom colormaps (Viridis, Plasma, RdBu_r, etc.)
- Adjustable grid resolution
- Custom titles and labels
- HTML export for interactive sharing
- Show/hide control
- Multiple trajectory color schemes
- Camera angle adjustment (3D plots)

## Project Structure

```
.
├── examples/
│   ├── config/                          # Hydra configuration files
│   ├── pendulum_state_training.py       # Original hard-coded training
│   ├── pendulum_state_training_symbolic.py  # Symbolic framework training
│   └── ...
├── neural_lyapunov_training/
│   ├── symbolic_dynamics.py             # Core symbolic framework
│   ├── symbolic_systems.py              # Example system definitions
│   ├── pendulum.py                      # Original hard-coded pendulum
│   ├── path_tracking.py                 # Original hard-coded path tracking
│   ├── quadrotor2d.py                   # Original hard-coded quadrotor
│   └── ...
├── verification/
│   ├── bisect.py                        # ρ bisection script
│   ├── generate_vnnlib.py               # Specification generator
│   └── specs/                           # VNNLIB specifications
├── alpha-beta-CROWN/                    # Verification tools (cloned)
├── requirements.txt                     # Python dependencies
└── README.md
```

## Examples

### Pre-Defined Systems (15+ Systems Available)

The framework includes comprehensive implementations of diverse dynamical systems:

**Pendulum Systems:**
* `SymbolicPendulum` - Inverted pendulum with partial observation (angle only)
* `SymbolicPendulum2ndOrder` - Second-order formulation (returns only acceleration)

**Aerial Vehicles:**
* `SymbolicQuadrotor2D` - Planar quadrotor (3 DOF, 2 inputs)
* `SymbolicQuadrotor2DLidar` - Lidar-based partial observations (4 ray distances)
* `PVTOL` - Planar Vertical Take-Off and Landing aircraft with body-frame velocities

**Mechanical Systems:**
* `CartPole` - Classic underactuated system (inverted pendulum on cart)
* `Manipulator2Link` - Two-link planar robot arm with coupled dynamics
* `CoupledOscillatorSystem` - Two masses with spring-damper and rotational coupling
* `FifthOrderMechanicalSystem` - High-order test system for integration schemes

**Nonlinear Dynamics:**
* `VanDerPolOscillator` - Self-excited oscillator with limit cycle
* `DuffingOscillator` - Nonlinear oscillator (bistable, hardening/softening spring)
* `Lorenz` - Famous chaotic system from atmospheric convection
* `NonlinearChainSystem` - Chain of five coupled nonlinear oscillators

**Vehicle Dynamics:**
* `DubinsVehicle` - Kinematic car model (unicycle dynamics)
* `PathTracking` - Error dynamics for circular path following

Each system includes:
- Detailed physical parameter descriptions
- Equilibrium point computation
- Full documentation of dynamics equations
- Parameter tuning guidelines

## Common Issues and Solutions

### Issue: Import errors when running scripts
**Solution:** Ensure PYTHONPATH is set correctly:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/alpha-beta-CROWN:$(pwd)/alpha-beta-CROWN/complete_verifier"
```

### Issue: PGD finds counterexamples after training
**Solution:** 
1. Set `cfg.train.train_lyaloss = False`
2. Reduce `cfg.model.rho_multiplier[-1]`
3. Retrain until stable

### Issue: Verification timeout
**Solution:**
1. Reduce batch size
2. Increase timeout
3. Use `--check_x_adv_only` during bisection

### Issue: Out of GPU memory during verification
**Solution:** Reduce batch size in config file or command line

### Issue: ROA metrics differ across sampling methods
**Solution:** This is expected for Monte Carlo (random). Use QMC methods (Sobol/Halton) for more consistent results with fewer samples. Grid method provides deterministic ground truth for 2D systems.

### Issue: Symbolic Jacobian verification fails
**Solution:** Use `verify_jacobians()` method to debug (with example debug below):
```python
system = SymbolicPendulum(m=1.0, l=0.5)
x = torch.tensor([0.1, 0.0])
u = torch.tensor([0.0])
result = system.verify_jacobians(x, u, tol=1e-3)
print(f"A matches: {result['A_match']}, error: {result['A_error']}")
print(f"B matches: {result['B_match']}, error: {result['B_error']}")
```

### Issue: Integration diverges for large time steps
**Solution:**
1. Reduce `dt` in `GenericDiscreteTimeSystem`
2. Use higher-order integration (RK4 instead of Euler)
3. Check system stiffness with `eigenvalues_at_equilibrium()`

### Issue: Visualization shows unexpected ROA shape
**Solution:**
1. Verify ρ threshold is appropriate
2. Check ΔV ≤ 0 region with `plot_lyapunov_2d(..., show_derivative=True)`
3. Use `compute_lyapunov_difference_metrics_qmc_sobol()` to quantify verified ROA
4. Inspect `max_violation_in_roa` and `percent_verified` metrics

## Citation

If you use this code, please cite the original paper and the course report:

```bibtex
@article{yang2024lyapunov,
  title={Lyapunov-stable Neural Control for State and Output Feedback: A Novel Formulation},
  author={Yang, Lujie and Dai, Hongkai and Shi, Zhouxing and Hsieh, Cho-Jui and Tedrake, Russ and Zhang, Huan},
  journal={arXiv preprint arXiv:2404.07956},
  year={2024}
}
```

```bibtex
@thesis{benezer2025lyapunov_extension,
  title={Dynamics Specification for Neural Control Lyapunov Function and Neural Controller Training},
  author={Benezer, Gil and Li, Fang-Hsin},
  school={Northeastern University},
  type={Final Project Report},
  year={2025},
  url={https://github.com/gbenezer/Lyapunov_Stable_NN_Controllers_Custom_Dynamics}
}
```

## License

MIT License

## Acknowledgments

* Original paper authors for their foundational work
* Northeastern University CS 7268 professor Dr. Michael Everett
* auto_LiRPA and alpha-beta-CROWN developers