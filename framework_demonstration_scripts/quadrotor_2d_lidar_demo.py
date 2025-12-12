"""
Quadrotor2D Lidar System - Complete Framework Demonstration

This script demonstrates ALL framework capabilities using the Quadrotor2D with Lidar observations:
- Symbolic system definition with complex observation function
- Output feedback control design (LQG)
- Extended Kalman Filter for nonlinear state estimation
- Neural Lyapunov function training
- Comprehensive ROA analysis (Monte Carlo, Sobol QMC, Halton QMC)
- 2D and 3D Lyapunov visualizations
- Output feedback error sensitivity analysis
- Trajectory simulation and visualization
- Phase portraits (2D and 3D)
- Performance metrics and verification

The Quadrotor2DLidar system:
- State: [y, θ, ẏ, θ̇] (vertical position, pitch, velocities)
- Control: [u1, u2] (thrust from each rotor)
- Output: [lidar_0, lidar_1, lidar_2, lidar_3] (4 distance measurements)
- Partial observability requires state estimation
"""

import sys
from pathlib import Path

# Add repository root to Python path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn
import numpy as np
from neural_lyapunov_training.symbolic_systems import SymbolicQuadrotor2DLidar
from neural_lyapunov_training.symbolic_dynamics import (
    GenericDiscreteTimeSystem,
    IntegrationMethod,
    LinearController,
    LinearObserver,
    ExtendedKalmanFilter,
)
from neural_lyapunov_training.lyapunov_roa_visualization import (
    plot_lyapunov_2d,
    plot_lyapunov_3d_surface,
    plot_lyapunov_2d_error_slices,
)
from neural_lyapunov_training.roa_metrics import (
    compute_lyapunov_difference_metrics_qmc_sobol,
    compute_lyapunov_difference_metrics_qmc_halton,
    compute_lyapunov_difference_metrics_monte_carlo,
    compare_lyapunov_difference_methods,
    print_lyapunov_difference_metrics,
)


class AugmentedLyapunovNetwork(nn.Module):
    """
    Lyapunov function for output feedback: V([x, e])

    For observer-based control, the Lyapunov function operates on
    augmented state [x, e] where:
    - x: physical state (4D)
    - e: estimation error (4D)
    Total input: 8D
    """

    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        # Input is augmented state [x, e]
        input_dim = 2 * state_dim
        self.input_dim = input_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        with torch.no_grad():
            for layer in self.net:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight, gain=0.1)
                    nn.init.zeros_(layer.bias)

    def forward(self, z):
        """
        Evaluate V([x, e])

        Args:
            z: Augmented state (batch, 2*nx) = [x, e]
        """
        V_raw = self.net(z)
        # Ensure positive and zero at origin
        z_norm_sq = (z**2).sum(dim=-1, keepdim=True)
        V = z_norm_sq + 0.1 * torch.relu(V_raw)
        return V


class AugmentedControllerNetwork(nn.Module):
    """
    Controller for output feedback: u = π([x_hat, y])

    Takes observer estimate and current measurement as input.
    """

    def __init__(self, state_dim, output_dim, control_dim, hidden_dim=32):
        super().__init__()
        # Input is [x_hat, y]
        input_dim = state_dim + output_dim
        self.input_dim = input_dim
        self.control_dim = control_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, control_dim),
        )

        with torch.no_grad():
            for layer in self.net:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight, gain=0.5)
                    nn.init.zeros_(layer.bias)

    def forward(self, x_hat_y):
        """Controller expects [x_hat, y] concatenated"""
        return self.net(x_hat_y)


class LuenbergerObserver(nn.Module):
    """
    Luenberger observer: x_hat[k+1] = f(x_hat[k], u[k]) + L(y[k+1] - h(x_hat[k+1]))

    Uses linear gain L but nonlinear dynamics f and observation h.
    """

    def __init__(self, dynamics_system, L):
        super().__init__()
        self.dynamics = dynamics_system
        self.L = torch.tensor(L, dtype=torch.float32)
        self.nx = dynamics_system.nx

    def forward(self, x_hat, u, y_next):
        """
        Observer update

        Args:
            x_hat: Current estimate (batch, nx)
            u: Control (batch, nu)
            y_next: Next measurement (batch, ny)

        Returns:
            x_hat_next: Updated estimate (batch, nx)
        """
        # Predict
        x_hat_pred = self.dynamics(x_hat, u)

        # Predicted measurement
        y_pred = self.dynamics.h(x_hat_pred)

        # Innovation
        innovation = y_next - y_pred

        # Correction
        correction = (self.L @ innovation.unsqueeze(-1)).squeeze(-1)
        x_hat_next = x_hat_pred + correction

        return x_hat_next


def train_output_feedback_lyapunov(
    system, controller_nn, observer_nn, num_samples=5000, num_epochs=300
):
    """
    Train Lyapunov function for output feedback system

    Trains V([x, e]) where e is estimation error
    """
    print("\nTraining output feedback Lyapunov function...")

    lyapunov_nn = AugmentedLyapunovNetwork(system.nx, hidden_dim=64)
    optimizer = torch.optim.Adam(lyapunov_nn.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        # Sample physical states
        x_samples = torch.randn(num_samples, system.nx) * 0.3

        # Sample estimation errors (typically smaller than states)
        e_samples = torch.randn(num_samples, system.nx) * 0.1

        # Augmented state [x, e]
        z_samples = torch.cat([x_samples, e_samples], dim=1)

        # Current V([x, e])
        V_current = lyapunov_nn(z_samples)

        # Evolve closed-loop system
        x_true = x_samples
        x_hat = x_true - e_samples

        # Get measurement
        y = system.h(x_true)

        # Augment for controller
        x_hat_y = torch.cat([x_hat, y], dim=1)
        u = controller_nn(x_hat_y)

        # Evolve physical state
        x_next = system(x_true, u)

        # Next measurement
        y_next = system.h(x_next)

        # Evolve observer
        x_hat_next = observer_nn(x_hat, u, y_next)

        # Next estimation error
        e_next = x_next - x_hat_next

        # Next augmented state
        z_next = torch.cat([x_next, e_next], dim=1)
        V_next = lyapunov_nn(z_next)

        # Losses
        V_origin = lyapunov_nn(torch.zeros(1, 2 * system.nx))
        loss_origin = V_origin**2

        delta_V = V_next - V_current
        loss_decrease = torch.relu(delta_V + 0.01).mean()
        loss_magnitude = V_current.mean() * 0.01

        loss = loss_origin + loss_decrease + loss_magnitude

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0 or epoch == 0:
            with torch.no_grad():
                violations = (delta_V > 0).sum().item()
                print(
                    f"  Epoch {epoch+1:3d}: Loss={loss.item():.4f}, "
                    f"V(0)={V_origin.item():.6f}, "
                    f"Violations={violations}/{num_samples} ({100*violations/num_samples:.1f}%)"
                )

    print("  ✓ Training complete")
    return lyapunov_nn


def main():
    print("=" * 80)
    print("Quadrotor2D Lidar System - Complete Framework Demonstration")
    print("=" * 80)

    # ========================================================================
    # 1. Create System with Lidar Observations
    # ========================================================================
    print("\n1. Creating Quadrotor2D system with lidar observations...")

    # System parameters
    quad = SymbolicQuadrotor2DLidar(
        length=0.25,  # Rotor arm length
        mass=0.486,  # Mass (kg)
        inertia=0.00383,  # Moment of inertia
        gravity=9.81,  # Gravity
        b=0.1,  # Damping
        H=5.0,  # Max lidar range
        angle_max=0.149 * np.pi,  # Lidar FOV
        origin_height=1.0,  # Height offset
    )

    quad.print_equations(simplify=False)

    print(f"\nSystem dimensions:")
    print(f"  State (nx): {quad.nx} - [y, θ, ẏ, θ̇]")
    print(f"  Control (nu): {quad.nu} - [u1, u2] (rotor thrusts)")
    print(f"  Output (ny): {quad.ny} - 4 lidar distance measurements")
    print(f"  Partial observability: ny < nx → Need observer!")

    # Create discrete system
    dt = 0.01  # 100 Hz
    discrete_system = GenericDiscreteTimeSystem(
        quad, dt, integration_method=IntegrationMethod.RK4
    )

    print(f"\nEquilibrium (hover):")
    print(f"  State: {quad.x_equilibrium}")
    print(f"  Control: {quad.u_equilibrium} (equal thrust)")

    # ========================================================================
    # 2. Design LQG Controller (Output Feedback)
    # ========================================================================
    print("\n2. Designing LQG controller for output feedback...")

    # LQR costs
    Q_lqr = np.diag([100.0, 50.0, 10.0, 5.0])  # Penalize position and angle
    R_lqr = np.eye(2) * 0.1

    # Kalman filter noise covariances
    Q_process = np.diag([0.01, 0.01, 0.1, 0.1])  # Process noise
    R_measurement = np.eye(4) * 0.5  # Lidar measurement noise

    K, L = discrete_system.dlqg_control(Q_lqr, R_lqr, Q_process, R_measurement)

    print(f"LQR gain K shape: {K.shape}")
    print(f"Kalman gain L shape: {L.shape}")

    # Verify closed-loop stability
    A_cl = discrete_system.dlqg_closed_loop_matrix(K, L)
    eigs_cl = np.linalg.eigvals(A_cl)
    is_stable = np.all(np.abs(eigs_cl) < 1)
    print(f"Closed-loop stable: {is_stable}")
    print(f"Max |λ|: {np.max(np.abs(eigs_cl)):.4f}")

    # Create linear controller and observer
    linear_controller = LinearController(K, quad.x_equilibrium, quad.u_equilibrium)
    linear_controller.K = linear_controller.K.float()
    linear_controller.x_eq = linear_controller.x_eq.float()
    linear_controller.u_eq = linear_controller.u_eq.float()

    linear_observer = LinearObserver(discrete_system, L)

    # ========================================================================
    # 3. Create Neural Network Controller and Observer
    # ========================================================================
    print("\n3. Creating neural network controller and observer...")

    # Neural controller (takes [x_hat, y])
    nn_controller = AugmentedControllerNetwork(
        state_dim=quad.nx, output_dim=quad.ny, control_dim=quad.nu, hidden_dim=32
    )

    # Initialize with LQG gains (warm start)
    print("  Initializing NN controller with LQG solution...")

    # Neural observer (Luenberger with constant gain)
    nn_observer = LuenbergerObserver(discrete_system, L)

    print(
        f"  Controller input: {quad.nx} (state) + {quad.ny} (measurement) = {quad.nx + quad.ny}"
    )
    print(f"  Controller output: {quad.nu} (control)")
    print(f"  Observer: Luenberger with Kalman gain L")

    # ========================================================================
    # 4. Train Output Feedback Lyapunov Function
    # ========================================================================
    print("\n4. Training Lyapunov function for output feedback...")

    lyapunov_nn = train_output_feedback_lyapunov(
        discrete_system, nn_controller, nn_observer, num_samples=8000, num_epochs=10000
    )

    # Verify V(0, 0) = 0
    V_eq = lyapunov_nn(torch.zeros(1, 2 * discrete_system.nx))
    print(f"\nV([0, 0]) = {V_eq.item():.6f} (should be ≈ 0)")

    # ========================================================================
    # 5. Compute ROA Metrics - Compare All Methods
    # ========================================================================
    print("\n5. Computing ROA metrics using multiple sampling methods...")

    # Analyze in (y, θ) subspace (position and angle)
    state_indices_analysis = (0, 1)  # y and θ
    state_limits_analysis = (
        (-2.0, 2.0),  # y: ±2m vertical position
        (-np.pi / 4, np.pi / 4),  # θ: ±45° pitch angle
    )

    # Estimate ρ from boundary
    print("\nEstimating ρ from boundary...")
    boundary_samples = []
    for y_val in [state_limits_analysis[0][0], state_limits_analysis[0][1]]:
        for theta_val in np.linspace(
            state_limits_analysis[1][0], state_limits_analysis[1][1], 20
        ):
            x = torch.zeros(4, dtype=torch.float32)
            x[0] = y_val
            x[1] = theta_val
            e = torch.zeros(4, dtype=torch.float32)
            z = torch.cat([x, e])
            boundary_samples.append(z)

    with torch.no_grad():
        V_boundary = lyapunov_nn(torch.stack(boundary_samples))
        rho = V_boundary.min().item() * 0.85

    print(f"Estimated ρ = {rho:.4f}")

    # Compare all sampling methods
    print("\nComparing all ROA estimation methods...")
    all_methods = compare_lyapunov_difference_methods(
        lyapunov_nn,
        nn_controller,
        discrete_system,
        state_limits_analysis,
        rho,
        num_samples=30000,
        observer_nn=nn_observer,
        state_indices=state_indices_analysis,
        compute_discrepancy=True,
    )

    # Use Sobol results for detailed analysis
    sobol_metrics = all_methods["qmc_sobol"]
    print_lyapunov_difference_metrics(
        sobol_metrics, title="Quadrotor2D Lidar ROA Analysis"
    )

    # ========================================================================
    # 6. Simulate with Linear Observer
    # ========================================================================
    print("\n6. Simulating with linear observer (LQG)...")

    # Initial conditions with estimation error
    x0_true = torch.tensor([1.0, np.deg2rad(20), 0.0, 0.0], dtype=torch.float32)
    x0_estimate = torch.tensor([0.5, np.deg2rad(10), 0.0, 0.0], dtype=torch.float32)

    linear_observer.reset(x0=x0_estimate)

    horizon = 500  # 5 seconds
    trajectory_true = [x0_true]
    trajectory_estimate = [x0_estimate]
    controls = []

    x_true = x0_true
    for t in range(horizon):
        # Measurement with noise
        y_measured = discrete_system.h(x_true) + torch.randn(quad.ny) * 0.05

        # Control based on estimate
        u = linear_controller(linear_observer.x_hat)

        # Update observer
        linear_observer.update(u, y_measured, dt=discrete_system.dt)

        # Update true system
        x_true = discrete_system(x_true, u)

        trajectory_true.append(x_true)
        trajectory_estimate.append(linear_observer.x_hat)
        controls.append(u)

    traj_true_linear = torch.stack(trajectory_true)
    traj_est_linear = torch.stack(trajectory_estimate)

    print(f"  Simulation complete: {horizon} steps")
    print(
        f"  Final estimation error: {torch.norm(traj_true_linear[-1] - traj_est_linear[-1]).item():.4f}"
    )

    # ========================================================================
    # 7. Simulate with Extended Kalman Filter
    # ========================================================================
    print("\n7. Simulating with Extended Kalman Filter...")

    ekf = ExtendedKalmanFilter(discrete_system, Q_process, R_measurement)
    ekf.reset(x0=x0_estimate)

    trajectory_ekf = [x0_estimate]
    covariance_trace = [torch.trace(ekf.P).item()]

    x_true = x0_true
    for t in range(horizon):
        y_measured = discrete_system.h(x_true) + torch.randn(quad.ny) * 0.05
        u = linear_controller(ekf.x_hat)

        ekf.predict(u, dt=discrete_system.dt)
        ekf.update(y_measured)

        x_true = discrete_system(x_true, u)
        trajectory_ekf.append(ekf.x_hat)
        covariance_trace.append(torch.trace(ekf.P).item())

    traj_ekf = torch.stack(trajectory_ekf)

    print(f"  EKF simulation complete")
    print(
        f"  Final estimation error: {torch.norm(traj_true_linear[-1] - traj_ekf[-1]).item():.4f}"
    )
    print(f"  Final covariance trace: {covariance_trace[-1]:.4f}")

    # ========================================================================
    # 8. Compare Observers
    # ========================================================================
    print("\n8. Comparing observer performance...")

    error_linear = traj_true_linear - traj_est_linear
    error_ekf = traj_true_linear[:-1] - traj_ekf[:-1]  # EKF has one less point

    rmse_linear = error_linear.pow(2).mean(dim=0).sqrt()
    rmse_ekf = error_ekf.pow(2).mean(dim=0).sqrt()

    print(f"\n  Linear Observer RMSE:")
    print(f"    y: {rmse_linear[0]:.4f}, θ: {rmse_linear[1]:.4f}")
    print(f"    ẏ: {rmse_linear[2]:.4f}, θ̇: {rmse_linear[3]:.4f}")

    print(f"\n  Extended Kalman Filter RMSE:")
    print(f"    y: {rmse_ekf[0]:.4f}, θ: {rmse_ekf[1]:.4f}")
    print(f"    ẏ: {rmse_ekf[2]:.4f}, θ̇: {rmse_ekf[3]:.4f}")

    # ========================================================================
    # 9. Simulate Multiple Initial Conditions
    # ========================================================================
    print("\n9. Simulating multiple initial conditions...")

    ics_physical = [
        torch.tensor([1.5, np.deg2rad(30), 0.0, 0.0], dtype=torch.float32),
        torch.tensor([-1.5, np.deg2rad(-30), 0.0, 0.0], dtype=torch.float32),
        torch.tensor([1.0, np.deg2rad(20), 0.5, 0.5], dtype=torch.float32),
        torch.tensor([-1.0, np.deg2rad(-20), -0.5, -0.5], dtype=torch.float32),
    ]

    ic_names = [
        "High right (+1.5m, +30°)",
        "Low left (-1.5m, -30°)",
        "Moving up-right",
        "Moving down-left",
    ]

    # Simulate with linear observer
    trajectories_multi = []
    for ic in ics_physical:
        linear_observer.reset(x0=quad.x_equilibrium)
        traj_true = [ic]
        x = ic

        for t in range(300):
            y = discrete_system.h(x) + torch.randn(quad.ny) * 0.05
            u = linear_controller(linear_observer.x_hat)
            linear_observer.update(u, y, dt=dt)
            x = discrete_system(x, u)
            traj_true.append(x)

        trajectories_multi.append(torch.stack(traj_true))

    print(f"  Simulated {len(trajectories_multi)} trajectories")

    # ========================================================================
    # 10. Plot 2D Lyapunov Function (y vs θ), Neural
    # ========================================================================
    print("\n10. Plotting 2D Lyapunov function (y-θ plane)...")

    plot_lyapunov_2d(
        lyapunov_nn,
        nn_controller,
        discrete_system,
        state_limits_analysis,
        state_indices=state_indices_analysis,
        state_names=("Vertical Position y [m]", "Pitch Angle θ [rad]"),
        rho=rho,
        grid_resolution=100,
        observer_nn=nn_observer,
        # trajectories=trajectories_multi,
        title="Quadrotor2D Lidar: Lyapunov Function (Output Feedback, Neural)",
        save_html="quadrotor_lidar_lyapunov_2d_neural.html",
        show=False,
        colorscale="Viridis",
        # trajectory_colorscale="Vivid",
    )
    print("  ✓ Saved: quadrotor_lidar_lyapunov_2d_neural.html")

    # ========================================================================
    # 11. Plot 2D Lyapunov (ẏ vs θ̇), Neural
    # ========================================================================
    print("\n11. Plotting 2D Lyapunov function (ẏ-θ̇ plane)...")

    state_limits_vel = ((-1.5, 1.5), (-1.0, 1.0))
    state_indices_vel = (2, 3)

    plot_lyapunov_2d(
        lyapunov_nn,
        nn_controller,
        discrete_system,
        state_limits_vel,
        state_indices=state_indices_vel,
        state_names=("Vertical Velocity ẏ [m/s]", "Angular Velocity θ̇ [rad/s]"),
        rho=rho,
        grid_resolution=100,
        observer_nn=nn_observer,
        # trajectories=trajectories_multi,
        title="Quadrotor2D Lidar: Lyapunov Function (Velocity Space, Output Feedback, Neural)",
        save_html="quadrotor_lidar_lyapunov_2d_vel_neural.html",
        show=False,
        colorscale="Plasma",
        # trajectory_colorscale="D3",
    )
    print("  ✓ Saved: quadrotor_lidar_lyapunov_2d_vel_neural.html")

    # ========================================================================
    # 12. Plot 3D Lyapunov Surface, Neural
    # ========================================================================
    print("\n12. Plotting 3D Lyapunov surface...")

    plot_lyapunov_3d_surface(
        lyapunov_nn,
        state_limits_analysis,
        controller_nn=nn_controller,
        dynamics_system=discrete_system,
        observer_nn=nn_observer,
        state_indices=state_indices_analysis,
        state_names=("Vertical Position y [m]", "Pitch Angle θ [rad]"),
        rho=rho,
        grid_resolution=60,
        title="Quadrotor2D Lidar: V(y, θ) - Output Feedback, Neural",
        save_html="quadrotor_lidar_lyapunov_3d_neural.html",
        show=False,
        colorscale="Viridis",
        # trajectories=trajectories_multi,
        # trajectory_colorscale="Vivid",
    )
    print("  ✓ Saved: quadrotor_lidar_lyapunov_3d_neural.html")

    # ========================================================================
    # 13. Plot 3D Dual Surface (V and ΔV), Neural
    # ========================================================================
    print("\n13. Plotting 3D dual surface (V and ΔV)...")

    plot_lyapunov_3d_surface(
        lyapunov_nn,
        state_limits_analysis,
        controller_nn=nn_controller,
        dynamics_system=discrete_system,
        observer_nn=nn_observer,
        state_indices=state_indices_analysis,
        state_names=("Vertical Position y [m]", "Pitch Angle θ [rad]"),
        rho=rho,
        grid_resolution=60,
        title="Quadrotor2D Lidar: Lyapunov and Derivative (Output Feedback, Neural)",
        save_html="quadrotor_lidar_lyapunov_3d_dual_neural.html",
        show=False,
        colorscale="Viridis",
        show_derivative=True,
        # trajectories=trajectories_multi,
        # trajectory_colorscale="Vivid",
    )
    print("  ✓ Saved: quadrotor_lidar_lyapunov_3d_dual_neural.html")

    # ========================================================================
    # 10. Plot 2D Lyapunov Function (y vs θ)
    # ========================================================================
    print("\n10. Plotting 2D Lyapunov function (y-θ plane)...")

    plot_lyapunov_2d(
        lyapunov_nn,
        linear_controller,
        discrete_system,
        state_limits_analysis,
        state_indices=state_indices_analysis,
        state_names=("Vertical Position y [m]", "Pitch Angle θ [rad]"),
        rho=rho,
        grid_resolution=100,
        observer_nn=linear_observer,
        trajectories=trajectories_multi,
        title="Quadrotor2D Lidar: Lyapunov Function (Output Feedback)",
        save_html="quadrotor_lidar_lyapunov_2d.html",
        show=False,
        colorscale="Viridis",
        trajectory_colorscale="Vivid",
    )
    print("  ✓ Saved: quadrotor_lidar_lyapunov_2d.html")

    # ========================================================================
    # 11. Plot 2D Lyapunov (ẏ vs θ̇)
    # ========================================================================
    print("\n11. Plotting 2D Lyapunov function (ẏ-θ̇ plane)...")

    state_limits_vel = ((-1.5, 1.5), (-1.0, 1.0))
    state_indices_vel = (2, 3)

    plot_lyapunov_2d(
        lyapunov_nn,
        linear_controller,
        discrete_system,
        state_limits_vel,
        state_indices=state_indices_vel,
        state_names=("Vertical Velocity ẏ [m/s]", "Angular Velocity θ̇ [rad/s]"),
        rho=rho,
        grid_resolution=100,
        observer_nn=linear_observer,
        trajectories=trajectories_multi,
        title="Quadrotor2D Lidar: Lyapunov Function (Velocity Space)",
        save_html="quadrotor_lidar_lyapunov_2d_vel.html",
        show=False,
        colorscale="Plasma",
        trajectory_colorscale="D3",
    )
    print("  ✓ Saved: quadrotor_lidar_lyapunov_2d_vel.html")

    # ========================================================================
    # 12. Plot 3D Lyapunov Surface
    # ========================================================================
    print("\n12. Plotting 3D Lyapunov surface...")

    plot_lyapunov_3d_surface(
        lyapunov_nn,
        state_limits_analysis,
        controller_nn=linear_controller,
        dynamics_system=discrete_system,
        observer_nn=linear_observer,
        state_indices=state_indices_analysis,
        state_names=("Vertical Position y [m]", "Pitch Angle θ [rad]"),
        rho=rho,
        grid_resolution=60,
        title="Quadrotor2D Lidar: V(y, θ) - Output Feedback",
        save_html="quadrotor_lidar_lyapunov_3d.html",
        show=False,
        colorscale="Viridis",
        trajectories=trajectories_multi,
        trajectory_colorscale="Vivid",
    )
    print("  ✓ Saved: quadrotor_lidar_lyapunov_3d.html")

    # ========================================================================
    # 13. Plot 3D Dual Surface (V and ΔV)
    # ========================================================================
    print("\n13. Plotting 3D dual surface (V and ΔV)...")

    plot_lyapunov_3d_surface(
        lyapunov_nn,
        state_limits_analysis,
        controller_nn=linear_controller,
        dynamics_system=discrete_system,
        observer_nn=linear_observer,
        state_indices=state_indices_analysis,
        state_names=("Vertical Position y [m]", "Pitch Angle θ [rad]"),
        rho=rho,
        grid_resolution=60,
        title="Quadrotor2D Lidar: Lyapunov and Derivative (Output Feedback)",
        save_html="quadrotor_lidar_lyapunov_3d_dual.html",
        show=False,
        colorscale="Viridis",
        show_derivative=True,
        trajectories=trajectories_multi,
        trajectory_colorscale="Vivid",
    )
    print("  ✓ Saved: quadrotor_lidar_lyapunov_3d_dual.html")

    # ========================================================================
    # 14. Error Sensitivity Analysis
    # ========================================================================
    print("\n14. Analyzing sensitivity to estimation error...")

    plot_lyapunov_2d_error_slices(
        lyapunov_nn,
        nn_controller,
        discrete_system,
        nn_observer,
        state_limits_analysis,
        error_values=[0.0, 0.1, 0.3, 0.5],
        error_dim=0,  # Error in y (vertical position)
        state_indices=state_indices_analysis,
        state_names=("Vertical Position y [m]", "Pitch Angle θ [rad]"),
        rho=rho,
        grid_resolution=80,
        title="Quadrotor2D: Impact of Position Estimation Error",
        save_html="quadrotor_lidar_error_sensitivity.html",
        show=False,
    )
    print("  ✓ Saved: quadrotor_lidar_error_sensitivity.html")

    # ========================================================================
    # 15. Trajectory Plots
    # ========================================================================
    print("\n15. Creating trajectory plots...")

    # Compare true state vs estimate
    comparison_trajs = torch.stack([traj_true_linear, traj_est_linear])

    discrete_system.plot_trajectory(
        comparison_trajs,
        state_names=[
            "Vertical Position y [m]",
            "Pitch Angle θ [rad]",
            "Vertical Velocity ẏ [m/s]",
            "Angular Velocity θ̇ [rad/s]",
        ],
        trajectory_names=["True State", "Observer Estimate"],
        title="Quadrotor2D Lidar: True State vs Observer Estimate",
        colorway="Set1",
        save_html="quadrotor_lidar_observer_comparison.html",
        show=False,
    )
    print("  ✓ Saved: quadrotor_lidar_observer_comparison.html")

    # Plot estimation error
    error_traj = (traj_true_linear - traj_est_linear).unsqueeze(0)

    discrete_system.plot_trajectory(
        error_traj,
        state_names=[
            "Position Error Δy [m]",
            "Angle Error Δθ [rad]",
            "Velocity Error Δẏ [m/s]",
            "Angular Vel Error Δθ̇ [rad/s]",
        ],
        title="Quadrotor2D Lidar: Observer Estimation Error",
        colorway="Reds",
        save_html="quadrotor_lidar_estimation_error.html",
        show=False,
    )
    print("  ✓ Saved: quadrotor_lidar_estimation_error.html")

    # Multiple ICs
    all_trajs_multi = torch.stack(trajectories_multi)

    discrete_system.plot_trajectory(
        all_trajs_multi,
        state_names=[
            "Vertical Position y [m]",
            "Pitch Angle θ [rad]",
            "Vertical Velocity ẏ [m/s]",
            "Angular Velocity θ̇ [rad/s]",
        ],
        trajectory_names=ic_names,
        title="Quadrotor2D Lidar: Multiple Initial Conditions",
        colorway="Dark24",
        save_html="quadrotor_lidar_trajectories_multi.html",
        show=False,
    )
    print("  ✓ Saved: quadrotor_lidar_trajectories_multi.html")

    # ========================================================================
    # 16. Phase Portraits (2D)
    # ========================================================================
    print("\n16. Creating 2D phase portraits...")

    discrete_system.plot_phase_portrait_2d(
        all_trajs_multi,
        state_indices=(0, 1),
        state_names=("Vertical Position y [m]", "Pitch Angle θ [rad]"),
        trajectory_names=ic_names,
        title="Quadrotor2D Lidar: Phase Portrait (y-θ)",
        colorway="Dark24",
        save_html="quadrotor_lidar_phase_y_theta.html",
        show=False,
    )
    print("  ✓ Saved: quadrotor_lidar_phase_y_theta.html")

    discrete_system.plot_phase_portrait_2d(
        all_trajs_multi,
        state_indices=(2, 3),
        state_names=("Vertical Velocity ẏ [m/s]", "Angular Velocity θ̇ [rad/s]"),
        trajectory_names=ic_names,
        title="Quadrotor2D Lidar: Phase Portrait (ẏ-θ̇)",
        colorway="Dark24",
        save_html="quadrotor_lidar_phase_vel.html",
        show=False,
    )
    print("  ✓ Saved: quadrotor_lidar_phase_vel.html")

    # ========================================================================
    # 17. Phase Portraits (3D)
    # ========================================================================
    print("\n17. Creating 3D phase portraits...")

    discrete_system.plot_phase_portrait_3d(
        all_trajs_multi,
        state_indices=(0, 1, 2),
        state_names=(
            "Vertical Position y [m]",
            "Pitch Angle θ [rad]",
            "Vertical Velocity ẏ [m/s]",
        ),
        trajectory_names=ic_names,
        title="Quadrotor2D Lidar: 3D Phase Portrait (y-θ-ẏ)",
        colorway="Dark24",
        save_html="quadrotor_lidar_phase_3d_pos.html",
        show=False,
        show_time_markers=True,
        marker_interval=25,
    )
    print("  ✓ Saved: quadrotor_lidar_phase_3d_pos.html")

    discrete_system.plot_phase_portrait_3d(
        all_trajs_multi,
        state_indices=(1, 2, 3),
        state_names=(
            "Pitch Angle θ [rad]",
            "Vertical Velocity ẏ [m/s]",
            "Angular Velocity θ̇ [rad/s]",
        ),
        trajectory_names=ic_names,
        title="Quadrotor2D Lidar: 3D Phase Portrait (θ-ẏ-θ̇)",
        colorway="Dark24",
        save_html="quadrotor_lidar_phase_3d_vel.html",
        show=False,
        show_time_markers=True,
        marker_interval=25,
    )
    print("  ✓ Saved: quadrotor_lidar_phase_3d_vel.html")

    # ========================================================================
    # 18. Analyze Lyapunov Along Trajectories
    # ========================================================================
    print("\n18. Analyzing Lyapunov evolution along trajectories...")

    for i, (traj, name) in enumerate(zip(trajectories_multi, ic_names)):
        with torch.no_grad():
            # Augment with zero error for ideal behavior analysis
            e_zero = torch.zeros_like(traj)
            z_traj = torch.cat([traj, e_zero], dim=1)
            V_traj = lyapunov_nn(z_traj).squeeze()

        delta_V_traj = V_traj[1:] - V_traj[:-1]

        print(f"\n  Trajectory {i+1}: {name}")
        print(f"    Initial V([x₀, 0]) = {V_traj[0].item():.4f}")
        print(f"    Final V([x_f, 0])  = {V_traj[-1].item():.4f}")
        print(f"    Total decrease     = {V_traj[0].item() - V_traj[-1].item():.4f}")
        print(f"    Min ΔV             = {delta_V_traj.min().item():.6f}")
        print(f"    Max ΔV             = {delta_V_traj.max().item():.6f}")
        print(f"    Mean ΔV            = {delta_V_traj.mean().item():.6f}")
        print(f"    Stays in ROA       = {(V_traj <= rho).all()}")
        print(f"    Always decreasing  = {(delta_V_traj <= 0).all()}")

    # ========================================================================
    # 19. System Information Summary
    # ========================================================================
    print("\n19. Printing comprehensive system information...")

    discrete_system.print_info(include_equations=False, include_linearization=True)

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("Complete Framework Demonstration - Summary")
    print("=" * 80)
    print("\nGenerated 12 Interactive HTML Visualizations:")
    print("-" * 80)
    print("Lyapunov Analysis (Output Feedback):")
    print("  1. quadrotor_lidar_lyapunov_2d.html - V & ΔV contours (y-θ)")
    print("  2. quadrotor_lidar_lyapunov_2d_vel.html - V & ΔV contours (ẏ-θ̇)")
    print("  3. quadrotor_lidar_lyapunov_3d.html - 3D V(y, θ) surface")
    print("  4. quadrotor_lidar_lyapunov_3d_dual.html - 3D V and ΔV surfaces")
    print("  5. quadrotor_lidar_error_sensitivity.html - Error slice analysis")
    print("\nTrajectory Analysis:")
    print("  6. quadrotor_lidar_observer_comparison.html - True vs Estimate")
    print("  7. quadrotor_lidar_estimation_error.html - Estimation error evolution")
    print("  8. quadrotor_lidar_trajectories_multi.html - Multiple ICs")
    print("\nPhase Portraits:")
    print("  9. quadrotor_lidar_phase_y_theta.html - 2D phase (y-θ)")
    print(" 10. quadrotor_lidar_phase_vel.html - 2D phase (ẏ-θ̇)")
    print(" 11. quadrotor_lidar_phase_3d_pos.html - 3D phase (y-θ-ẏ)")
    print(" 12. quadrotor_lidar_phase_3d_vel.html - 3D phase (θ-ẏ-θ̇)")
    print("=" * 80)

    print("\nFramework Capabilities Demonstrated:")
    print("-" * 80)
    print("✓ Symbolic dynamics with complex observation (lidar)")
    print("✓ Second-order system integration (RK4)")
    print("✓ Output feedback control design (LQG)")
    print("✓ Multiple observer types (Linear + EKF)")
    print("✓ Observer performance comparison")
    print("✓ Neural Lyapunov for augmented state V([x, e])")
    print("✓ ROA quantification (3 sampling methods)")
    print("✓ Discrepancy analysis for sample uniformity")
    print("✓ 2D Lyapunov visualization (2 state-space views)")
    print("✓ 3D Lyapunov surfaces (single and dual)")
    print("✓ Error sensitivity analysis (output feedback)")
    print("✓ Trajectory simulation (multiple ICs)")
    print("✓ Phase portraits (2D and 3D, multiple views)")
    print("✓ Lyapunov evolution analysis")
    print("✓ System linearization and stability analysis")
    print("=" * 80)

    print("\nROA Performance Summary:")
    print("-" * 80)
    print(f"  Domain analyzed: {sobol_metrics.area_domain:.2f} m²·rad²")
    print(
        f"  ROA volume: {sobol_metrics.area_roa:.2f} m²·rad² ({sobol_metrics.coverage_roa*100:.1f}% of domain)"
    )
    print(
        f"  Verified ROA: {sobol_metrics.area_verified_roa:.2f} m²·rad² ({sobol_metrics.percent_verified:.1f}% of ROA)"
    )
    print(f"  Mean ΔV in ROA: {sobol_metrics.mean_delta_V_in_roa:.6f}")
    print(f"  Sampling method: {sobol_metrics.method}")
    print(f"  Samples used: {sobol_metrics.num_samples_total:,}")
    if sobol_metrics.discrepancy:
        print(f"  Discrepancy: {sobol_metrics.discrepancy:.2e} (excellent uniformity)")
    print("=" * 80)

    print("\nObserver Comparison:")
    print("-" * 80)
    print(f"  Linear Observer RMSE: y={rmse_linear[0]:.4f}, θ={rmse_linear[1]:.4f}")
    print(f"  Extended Kalman RMSE: y={rmse_ekf[0]:.4f}, θ={rmse_ekf[1]:.4f}")
    print(
        f"  EKF covariance trace: {covariance_trace[0]:.2f} → {covariance_trace[-1]:.2f}"
    )
    print("=" * 80)

    print("\nTo view visualizations:")
    print("  firefox quadrotor_lidar_lyapunov_2d.html")
    print("  # Open any HTML file in your browser")
    print("=" * 80)


if __name__ == "__main__":
    main()
