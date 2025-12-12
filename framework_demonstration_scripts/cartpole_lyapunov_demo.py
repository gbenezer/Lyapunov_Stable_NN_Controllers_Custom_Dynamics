"""
CartPole System - Lyapunov Function Visualization Demo

This script demonstrates Lyapunov function visualization capabilities for the CartPole system:
- Training a simple neural Lyapunov function
- 2D contour plots of V(x) and ΔV(x)
- 3D surface plots of Lyapunov function
- ROA analysis and metrics
- Trajectory overlays showing V(x) evolution

The CartPole is a classic underactuated system: inverted pendulum on a moving cart.
State: [cart_position, pole_angle, cart_velocity, pole_angular_velocity]
Control: [horizontal_force]
"""

import sys
import os
from pathlib import Path

# Add repository root to Python path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn
import numpy as np
from neural_lyapunov_training.symbolic_systems import CartPole
from neural_lyapunov_training.symbolic_dynamics import (
    GenericDiscreteTimeSystem,
    IntegrationMethod,
    LinearController,
)
from neural_lyapunov_training.lyapunov_roa_visualization import (
    plot_lyapunov_2d,
    plot_lyapunov_3d_surface,
)
from neural_lyapunov_training.roa_metrics import (
    compute_lyapunov_difference_metrics_qmc_sobol,
    print_lyapunov_difference_metrics,
)


class SimpleLyapunovNetwork(nn.Module):
    """Simple quadratic-like Lyapunov function for demonstration"""

    def __init__(self, state_dim, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        # Initialize to approximate quadratic form
        with torch.no_grad():
            for layer in self.net:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight, gain=0.1)
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        """Evaluate V(x), ensuring V(0) ≈ 0 and V(x) > 0 for x ≠ 0"""
        V_raw = self.net(x)
        x_norm_sq = (x**2).sum(dim=-1, keepdim=True)
        V = x_norm_sq + 0.1 * torch.relu(V_raw)
        return V


def train_simple_lyapunov(system, controller, num_samples=5000, num_epochs=100):
    """Train a simple Lyapunov function by sampling and enforcing conditions"""
    print("\nTraining simple Lyapunov function...")

    lyapunov_nn = SimpleLyapunovNetwork(system.nx, hidden_dim=32)
    optimizer = torch.optim.Adam(lyapunov_nn.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        x_samples = torch.randn(num_samples, system.nx) * 0.5
        V_current = lyapunov_nn(x_samples)
        u_samples = controller(x_samples)
        x_next = system(x_samples, u_samples)
        V_next = lyapunov_nn(x_next)

        V_origin = lyapunov_nn(torch.zeros(1, system.nx))
        loss_origin = V_origin**2
        delta_V = V_next - V_current
        loss_decrease = torch.relu(delta_V + 0.01).mean()
        loss_magnitude = V_current.mean() * 0.01
        loss = loss_origin + loss_decrease + loss_magnitude

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0 or epoch == 0:
            with torch.no_grad():
                violations = (delta_V > 0).sum().item()
                print(
                    f"  Epoch {epoch+1:3d}: Loss={loss.item():.4f}, "
                    f"V(0)={V_origin.item():.6f}, "
                    f"Violations={violations}/{num_samples} "
                    f"({100*violations/num_samples:.1f}%)"
                )

    print("  ✓ Training complete")
    return lyapunov_nn


def main():
    print("=" * 70)
    print("CartPole System - Lyapunov Function Visualization Demo")
    print("=" * 70)

    # ========================================================================
    # 1. Create CartPole System
    # ========================================================================
    print("\n1. Creating CartPole system...")

    cartpole = CartPole(m_cart=1.0, m_pole=0.1, length=0.5, gravity=9.81, friction=0.1)
    cartpole.print_equations(simplify=True)

    dt = 0.02
    discrete_system = GenericDiscreteTimeSystem(
        cartpole, dt, integration_method=IntegrationMethod.RK4
    )

    print(f"\nSystem parameters:")
    print(f"  Cart mass: 1.0 kg, Pole mass: 0.1 kg")
    print(f"  Pole length: 0.5 m, Sample time: {dt} s ({1/dt} Hz)")
    print(f"\nEquilibrium (upright): {cartpole.x_equilibrium}")

    is_eq, max_deriv = cartpole.check_equilibrium(
        cartpole.x_equilibrium, cartpole.u_equilibrium
    )
    print(f"Equilibrium valid: {is_eq}")
    print(f"Open-loop stable: {cartpole.is_stable_equilibrium()}")

    # ========================================================================
    # 2. Design LQR Controller
    # ========================================================================
    print("\n2. Designing LQR controller...")

    Q = np.diag([10.0, 100.0, 1.0, 1.0])
    R = np.array([[0.1]])
    K, S = discrete_system.dlqr_control(Q, R)
    print(f"LQR gain K: {K}")

    Ad, Bd = discrete_system.linearized_dynamics(
        cartpole.x_equilibrium.unsqueeze(0), cartpole.u_equilibrium.unsqueeze(0)
    )
    if Bd.squeeze().numpy().ndim == 1:
        Bd_np = Bd.squeeze().numpy().reshape(-1, 1)
    else:
        Bd_np = Bd.squeeze().numpy()

    A_cl = Ad.squeeze().numpy() + Bd_np @ K
    eigs_cl = np.linalg.eigvals(A_cl)
    print(f"Closed-loop stable: {np.all(np.abs(eigs_cl) < 1)}")

    controller = LinearController(K, cartpole.x_equilibrium, cartpole.u_equilibrium)
    controller.K = controller.K.float()
    controller.x_eq = controller.x_eq.float()
    controller.u_eq = controller.u_eq.float()

    # ========================================================================
    # 3. Train Lyapunov Function
    # ========================================================================
    print("\n3. Training Lyapunov function...")
    lyapunov_nn = train_simple_lyapunov(
        discrete_system, controller, num_samples=10000, num_epochs=200
    )

    V_eq = lyapunov_nn(torch.zeros(1, discrete_system.nx))
    print(f"\nV(equilibrium) = {V_eq.item():.6f} (should be ≈ 0)")

    # ========================================================================
    # 4. Compute ROA Metrics
    # ========================================================================
    print("\n4. Computing ROA metrics...")

    # Analyze in (θ, θ̇) subspace - state indices (1, 3)
    state_indices_analysis = (1, 3)
    state_limits_analysis = (
        (-np.pi / 3, np.pi / 3),  # θ bounds
        (-2.0, 2.0),  # θ̇ bounds
    )

    # Estimate ρ
    boundary_samples = []
    for theta in [state_limits_analysis[0][0], state_limits_analysis[0][1]]:
        for theta_dot in np.linspace(
            state_limits_analysis[1][0], state_limits_analysis[1][1], 20
        ):
            state = torch.zeros(4, dtype=torch.float32)
            state[1] = theta
            state[3] = theta_dot
            boundary_samples.append(state)

    with torch.no_grad():
        V_boundary = lyapunov_nn(torch.stack(boundary_samples))
        rho = V_boundary.min().item() * 0.9

    print(f"Estimated ρ = {rho:.4f}")

    metrics = compute_lyapunov_difference_metrics_qmc_sobol(
        lyapunov_nn,
        controller,
        discrete_system,
        state_limits_analysis,
        rho,
        num_samples=50000,
        state_indices=state_indices_analysis,
        compute_discrepancy_metric=True,
    )

    print_lyapunov_difference_metrics(metrics, title="CartPole ROA Analysis")

    # ========================================================================
    # 5. Simulate Trajectories
    # ========================================================================
    print("\n5. Simulating sample trajectories...")

    initial_conditions = [
        torch.tensor([0.0, np.deg2rad(30), 0.0, 0.0], dtype=torch.float32),
        torch.tensor([0.0, np.deg2rad(-30), 0.0, 0.0], dtype=torch.float32),
        torch.tensor([0.0, np.deg2rad(20), 0.0, 1.0], dtype=torch.float32),
        torch.tensor([0.0, np.deg2rad(-20), 0.0, -1.0], dtype=torch.float32),
    ]

    trajectory_names = ["30° right", "30° left", "20° right, θ̇=+1", "20° left, θ̇=-1"]

    trajectories = [
        discrete_system.simulate(ic, controller=controller, horizon=300)
        for ic in initial_conditions
    ]
    print(f"Simulated {len(trajectories)} trajectories")

    # ========================================================================
    # 6. Plot 2D Lyapunov (θ vs θ̇)
    # ========================================================================
    print("\n6. Plotting 2D Lyapunov function (θ-θ̇ plane)...")

    plot_lyapunov_2d(
        lyapunov_nn,
        controller,
        discrete_system,
        state_limits_analysis,
        state_indices=state_indices_analysis,
        state_names=("Pole Angle θ [rad]", "Angular Velocity θ̇ [rad/s]"),
        rho=rho,
        grid_resolution=120,
        trajectories=trajectories,
        title="CartPole: Lyapunov Function and ROA (θ-θ̇ Plane)",
        save_html="cartpole_lyapunov_2d.html",
        show=False,
        colorscale="Viridis",
        trajectory_colorscale="Vivid",
    )
    print("  ✓ Saved: cartpole_lyapunov_2d.html")

    # ========================================================================
    # 7. Plot 2D Lyapunov (x vs θ)
    # ========================================================================
    print("\n7. Plotting 2D Lyapunov function (x-θ plane)...")

    state_limits_x_theta = ((-1.0, 1.0), (-np.pi / 3, np.pi / 3))
    state_indices_x_theta = (0, 1)

    plot_lyapunov_2d(
        lyapunov_nn,
        controller,
        discrete_system,
        state_limits_x_theta,
        state_indices=state_indices_x_theta,
        state_names=("Cart Position x [m]", "Pole Angle θ [rad]"),
        rho=rho,
        grid_resolution=120,
        trajectories=trajectories,
        title="CartPole: Lyapunov Function (x-θ Plane)",
        save_html="cartpole_lyapunov_2d_x_theta.html",
        show=False,
        colorscale="Plasma",
        trajectory_colorscale="D3",
    )
    print("  ✓ Saved: cartpole_lyapunov_2d_x_theta.html")

    # ========================================================================
    # 8. Plot 3D Lyapunov Surface (θ vs θ̇)
    # ========================================================================
    print("\n8. Plotting 3D Lyapunov surface...")

    plot_lyapunov_3d_surface(
        lyapunov_nn,
        state_limits_analysis,
        controller_nn=controller,
        dynamics_system=discrete_system,
        state_indices=state_indices_analysis,
        state_names=("Pole Angle θ [rad]", "Angular Velocity θ̇ [rad/s]"),
        rho=rho,
        grid_resolution=80,
        title="CartPole: Lyapunov Function V(θ, θ̇)",
        save_html="cartpole_lyapunov_3d_single.html",
        show=False,
        colorscale="Viridis",
        show_derivative=False,
        trajectories=trajectories,
        trajectory_colorscale="Vivid",
    )
    print("  ✓ Saved: cartpole_lyapunov_3d_single.html")

    # ========================================================================
    # 9. Plot 3D Dual Surface (V and ΔV)
    # ========================================================================
    print("\n9. Plotting 3D dual surface...")

    plot_lyapunov_3d_surface(
        lyapunov_nn,
        state_limits_analysis,
        controller_nn=controller,
        dynamics_system=discrete_system,
        state_indices=state_indices_analysis,
        state_names=("Pole Angle θ [rad]", "Angular Velocity θ̇ [rad/s]"),
        rho=rho,
        grid_resolution=80,
        title="CartPole: Lyapunov Function and Derivative",
        save_html="cartpole_lyapunov_3d_dual.html",
        show=False,
        colorscale="Viridis",
        show_derivative=True,
        trajectories=trajectories,
        trajectory_colorscale="Vivid",
    )
    print("  ✓ Saved: cartpole_lyapunov_3d_dual.html")

    # ========================================================================
    # 10. Analyze Trajectories
    # ========================================================================
    print("\n10. Analyzing Lyapunov evolution along trajectories...")

    for i, (traj, name) in enumerate(zip(trajectories, trajectory_names)):
        with torch.no_grad():
            V_traj = lyapunov_nn(traj).squeeze()
        delta_V_traj = V_traj[1:] - V_traj[:-1]

        print(f"\n  Trajectory {i+1}: {name}")
        print(f"    Initial V(x₀) = {V_traj[0].item():.4f}")
        print(f"    Final V(x_f)  = {V_traj[-1].item():.4f}")
        print(f"    Decrease      = {V_traj[0].item() - V_traj[-1].item():.4f}")
        print(f"    Min ΔV        = {delta_V_traj.min().item():.6f}")
        print(f"    Max ΔV        = {delta_V_traj.max().item():.6f}")
        print(f"    Mean ΔV       = {delta_V_traj.mean().item():.6f}")
        print(f"    Stays in ROA  = {(V_traj <= rho).all()}")

    # ========================================================================
    # 11. Trajectory Summary Plot
    # ========================================================================
    print("\n11. Creating trajectory summary plot...")

    all_trajs = torch.stack(trajectories)
    discrete_system.plot_trajectory(
        all_trajs,
        state_names=[
            "Cart Position x [m]",
            "Pole Angle θ [rad]",
            "Cart Velocity ẋ [m/s]",
            "Angular Velocity θ̇ [rad/s]",
        ],
        trajectory_names=trajectory_names,
        title="CartPole: State Evolution",
        colorway="Vivid",
        save_html="cartpole_trajectories.html",
        show=False,
    )
    print("  ✓ Saved: cartpole_trajectories.html")

    # ========================================================================
    # 12. Phase Portraits
    # ========================================================================
    print("\n12. Creating phase portraits...")

    discrete_system.plot_phase_portrait_2d(
        all_trajs,
        state_indices=(1, 3),
        state_names=("Pole Angle θ [rad]", "Angular Velocity θ̇ [rad/s]"),
        trajectory_names=trajectory_names,
        title="CartPole: Phase Portrait (θ-θ̇)",
        colorway="Vivid",
        save_html="cartpole_phase_theta.html",
        show=False,
    )
    print("  ✓ Saved: cartpole_phase_theta.html")

    discrete_system.plot_phase_portrait_2d(
        all_trajs,
        state_indices=(0, 1),
        state_names=("Cart Position x [m]", "Pole Angle θ [rad]"),
        trajectory_names=trajectory_names,
        title="CartPole: Phase Portrait (x-θ)",
        colorway="Vivid",
        save_html="cartpole_phase_x_theta.html",
        show=False,
    )
    print("  ✓ Saved: cartpole_phase_x_theta.html")

    discrete_system.plot_phase_portrait_3d(
        all_trajs,
        state_indices=(0, 1, 3),
        state_names=(
            "Cart Position x [m]",
            "Pole Angle θ [rad]",
            "Angular Velocity θ̇ [rad/s]",
        ),
        trajectory_names=trajectory_names,
        title="CartPole: 3D Phase Portrait",
        colorway="Vivid",
        save_html="cartpole_phase_3d.html",
        show=False,
        show_time_markers=True,
        marker_interval=15,
    )
    print("  ✓ Saved: cartpole_phase_3d.html")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("Demo Complete! Generated Files:")
    print("=" * 70)
    print("  1. cartpole_lyapunov_2d.html - 2D Lyapunov (θ-θ̇)")
    print("  2. cartpole_lyapunov_2d_x_theta.html - 2D Lyapunov (x-θ)")
    print("  3. cartpole_lyapunov_3d_single.html - 3D V(θ, θ̇) surface")
    print("  4. cartpole_lyapunov_3d_dual.html - 3D V and ΔV surfaces")
    print("  5. cartpole_trajectories.html - State evolution")
    print("  6. cartpole_phase_theta.html - Phase portrait (θ-θ̇)")
    print("  7. cartpole_phase_x_theta.html - Phase portrait (x-θ)")
    print("  8. cartpole_phase_3d.html - 3D phase portrait")
    print("=" * 70)
    print("\nKey Insights:")
    print("  • Red boundary: ROA limit (V = ρ)")
    print("  • Black dashed: Stability boundary (ΔV = 0)")
    print("  • Trajectories converge to equilibrium within ROA")
    print("=" * 70)
    print("\nROA Performance Summary:")
    print(f"  ROA coverage: {metrics.coverage_roa*100:.2f}% of domain")
    print(f"  Verified ROA: {metrics.percent_verified:.2f}% of ROA has ΔV ≤ 0")
    print(f"  Mean ΔV in ROA: {metrics.mean_delta_V_in_roa:.6f}")
    print(f"  Max violation: {metrics.max_violation_in_roa:.6f}")
    print("\nNote: Simplified training achieves ~30% verification.")
    print("      Full paper method achieves >95% verification.")
    print("=" * 70)
    print("\nView plots: firefox cartpole_lyapunov_2d.html")
    print("=" * 70)


if __name__ == "__main__":
    main()
