"""
Path Tracking System - Trajectory and Phase Portrait Visualization Demo

This script demonstrates comprehensive visualization capabilities for the path tracking system:
- 2D trajectory plots with state and control
- 3D trajectory visualization with time coloring
- 2D phase portraits
- 3D phase portraits
- Multiple initial conditions comparison

The path tracking system models error dynamics for a vehicle following a circular path.
State: [lateral_error, heading_error]
Control: [steering_angle]
"""

import sys
import os
from pathlib import Path

# Add repository root to Python path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

import torch
import numpy as np
from neural_lyapunov_training.symbolic_systems import PathTracking
from neural_lyapunov_training.symbolic_dynamics import (
    GenericDiscreteTimeSystem,
    IntegrationMethod,
    LinearController
)

def main():
    print("="*70)
    print("Path Tracking System - Visualization Demo")
    print("="*70)
    
    # ========================================================================
    # 1. Create Path Tracking System
    # ========================================================================
    print("\n1. Creating path tracking system...")
    
    # System parameters
    speed = 2.0         # Forward speed [m/s]
    wheelbase = 1.5     # Vehicle wheelbase [m]
    radius = 10.0       # Reference path radius [m]
    
    # Create continuous-time system
    path_tracking = PathTracking(speed=speed, length=wheelbase, radius=radius)
    
    # Print system information
    path_tracking.print_equations(simplify=True)
    
    # Create discrete-time system
    dt = 0.02  # 50 Hz control rate
    discrete_system = GenericDiscreteTimeSystem(
        path_tracking,
        dt,
        integration_method=IntegrationMethod.RK4
    )
    
    print(f"\nSystem parameters:")
    print(f"  Forward speed: {speed} m/s")
    print(f"  Wheelbase: {wheelbase} m")
    print(f"  Path radius: {radius} m")
    print(f"  Sample time: {dt} s ({1/dt} Hz)")
    
    # Check equilibrium
    print(f"\nEquilibrium state: {path_tracking.x_equilibrium}")
    print(f"Equilibrium control: {path_tracking.u_equilibrium}")
    
    is_eq, max_deriv = path_tracking.check_equilibrium(
        path_tracking.x_equilibrium,
        path_tracking.u_equilibrium
    )
    print(f"Equilibrium valid: {is_eq} (max derivative: {max_deriv:.6e})")
    
    # ========================================================================
    # 2. Design LQR Controller
    # ========================================================================
    print("\n2. Designing LQR controller...")
    
    # Penalize lateral error more than heading error
    Q = np.diag([100.0, 10.0])  # State cost: [lateral_error, heading_error]
    R = np.array([[1.0]])        # Control cost: [steering_angle]
    
    K, S = discrete_system.dlqr_control(Q, R)
    
    print(f"LQR gain K: {K}")
    print(f"LQR gain K shape: {K.shape}")
    
    # Compute closed-loop eigenvalues
    # A_cl = Ad + Bd @ K where K is (nu, nx)
    Ad, Bd = discrete_system.linearized_dynamics(
        path_tracking.x_equilibrium.unsqueeze(0),
        path_tracking.u_equilibrium.unsqueeze(0)
    )
    Ad_np = Ad.squeeze().numpy()  # (nx, nx)
    Bd_np = Bd.squeeze().numpy()  # (nx, nu)
    
    print(f"Ad shape: {Ad_np.shape}, Bd shape: {Bd_np.shape}")
    
    # Ensure Bd is 2D (nx, nu)
    if Bd_np.ndim == 1:
        Bd_np = Bd_np.reshape(-1, 1)
    
    # Bd is (nx, nu) = (2, 1) and K is (nu, nx) = (1, 2)
    # Bd @ K gives (2, 2)
    A_cl = Ad_np + Bd_np @ K
    eigs_cl = np.linalg.eigvals(A_cl)
    print(f"Closed-loop eigenvalues: {eigs_cl}")
    print(f"Stable (all |λ| < 1): {np.all(np.abs(eigs_cl) < 1)}")
    
    # Create controller
    controller = LinearController(K, path_tracking.x_equilibrium, path_tracking.u_equilibrium)
    
    # Move controller to match simulation dtype (float32)
    controller.K = controller.K.float()
    controller.x_eq = controller.x_eq.float()
    controller.u_eq = controller.u_eq.float()
    
    # ========================================================================
    # 3. Simulate Single Trajectory
    # ========================================================================
    print("\n3. Simulating single trajectory...")
    
    # Initial condition: 2m lateral error, 15° heading error
    x0_single = torch.tensor([2.0, np.deg2rad(15)], dtype=torch.float32)
    horizon = 500  # 10 seconds at 50 Hz
    
    trajectory_single, controls_single = discrete_system.simulate(
        x0_single,
        controller=controller,
        horizon=horizon,
        return_controls=True
    )
    
    print(f"Trajectory shape: {trajectory_single.shape}")
    print(f"Controls shape: {controls_single.shape}")
    print(f"Final state: lateral_error={trajectory_single[-1, 0]:.4f} m, heading_error={np.rad2deg(trajectory_single[-1, 1].item()):.4f}°")
    
    # ========================================================================
    # 4. Plot 2D Trajectory (States Only)
    # ========================================================================
    print("\n4. Plotting 2D trajectory (states only)...")
    
    discrete_system.plot_trajectory(
        trajectory_single,
        state_names=['Lateral Error d_e [m]', 'Heading Error θ_e [rad]'],
        title='Path Tracking: Single Trajectory Response (States)',
        colorway='Plotly',
        save_html='path_tracking_2d_trajectory_states.html',
        show=False  # Don't display, just save
    )
    
    print("  ✓ Saved: path_tracking_2d_trajectory_states.html")
    
    # ========================================================================
    # 5. Plot 2D Trajectory (States + Control)
    # ========================================================================
    print("\n5. Plotting 2D trajectory with states and control...")
    
    discrete_system.plot_trajectory(
        trajectory_single,
        state_names=['Lateral Error d_e [m]', 'Heading Error θ_e [rad]'],
        control_sequence=controls_single,
        title='Path Tracking: Single Trajectory Response (States + Control)',
        colorway='Plotly',
        save_html='path_tracking_2d_trajectory_with_control.html',
        show=False
    )
    
    print("  ✓ Saved: path_tracking_2d_trajectory_with_control.html")
    
    # ========================================================================
    # 5. Simulate Multiple Initial Conditions
    # ========================================================================
    print("\n6. Simulating multiple initial conditions...")
    
    # Create grid of initial conditions
    lateral_errors = np.linspace(-3, 3, 5)
    heading_errors = np.deg2rad(np.linspace(-30, 30, 4))
    
    initial_conditions = []
    for d_e in lateral_errors:
        for theta_e in heading_errors:
            initial_conditions.append([d_e, theta_e])
    
    ic_tensor = torch.tensor(initial_conditions, dtype=torch.float32)
    print(f"Number of initial conditions: {len(initial_conditions)}")
    
    # Simulate all at once (batched)
    trajectories_batch = discrete_system.simulate(
        ic_tensor,
        controller=controller,
        horizon=horizon,
        return_controls=False
    )
    
    print(f"Batch trajectories shape: {trajectories_batch.shape}")
    
    # ========================================================================
    # 6. Plot 2D Phase Portrait
    # ========================================================================
    print("\n7. Plotting 2D phase portrait...")
    
    discrete_system.plot_phase_portrait_2d(
        trajectories_batch,
        state_indices=(0, 1),
        state_names=('Lateral Error d_e [m]', 'Heading Error θ_e [rad]'),
        title='Path Tracking: Phase Portrait (Multiple ICs)',
        colorway='Dark24',
        save_html='path_tracking_phase_portrait_2d.html',
        show=False
    )
    
    print("  ✓ Saved: path_tracking_phase_portrait_2d.html")
    
    # ========================================================================
    # 7. Plot 3D Trajectory (Time-Colored)
    # ========================================================================
    print("\n8. Plotting 3D trajectory with time coloring...")
    
    # For path tracking (2D state), we'll add time as the third dimension
    # Create augmented state: [lateral_error, heading_error, time]
    time_axis = torch.arange(trajectory_single.shape[0]) * dt
    trajectory_3d = torch.cat([
        trajectory_single,
        time_axis.unsqueeze(1)
    ], dim=1)
    
    discrete_system.plot_trajectory_3d(
        trajectory_3d,
        state_indices=(0, 1, 2),
        state_names=('Lateral Error d_e [m]', 'Heading Error θ_e [rad]', 'Time [s]'),
        title='Path Tracking: 3D Trajectory (Time Evolution)',
        colorway='Viridis',
        save_html='path_tracking_3d_trajectory_time.html',
        show=False,
        show_markers=True,
        marker_size=2,
        line_width=3
    )
    
    print("  ✓ Saved: path_tracking_3d_trajectory_time.html")
    
    # ========================================================================
    # 8. Plot 3D Phase Portrait (Multiple Trajectories)
    # ========================================================================
    print("\n9. Plotting 3D phase portrait with multiple trajectories...")
    
    # Add time dimension to all trajectories
    batch_size = trajectories_batch.shape[0]
    time_axis_batch = torch.arange(trajectories_batch.shape[1]) * dt
    time_expanded = time_axis_batch.unsqueeze(0).unsqueeze(2).expand(batch_size, -1, 1)
    
    trajectories_3d_batch = torch.cat([
        trajectories_batch,
        time_expanded
    ], dim=2)
    
    # Plot subset for clarity (every 3rd trajectory)
    subset_indices = range(0, batch_size, 3)
    trajectories_subset = trajectories_3d_batch[subset_indices]
    
    discrete_system.plot_phase_portrait_3d(
        trajectories_subset,
        state_indices=(0, 1, 2),
        state_names=('Lateral Error d_e [m]', 'Heading Error θ_e [rad]', 'Time [s]'),
        title=f'Path Tracking: 3D Phase Portrait ({len(subset_indices)} Trajectories)',
        colorway='Set1',
        save_html='path_tracking_phase_portrait_3d.html',
        show=False,
        show_time_markers=True,
        marker_interval=25  # Mark every 0.5 seconds
    )
    
    print("  ✓ Saved: path_tracking_phase_portrait_3d.html")
    
    # ========================================================================
    # 9. Analyze Controller Performance
    # ========================================================================
    print("\n10. Analyzing controller performance...")
    
    # Compute metrics for all trajectories
    final_errors = trajectories_batch[:, -1, :]
    settling_time_threshold = 0.02  # 2% of initial error
    
    metrics = []
    for i, traj in enumerate(trajectories_batch):
        initial_error_norm = torch.norm(ic_tensor[i])
        final_error_norm = torch.norm(final_errors[i])
        
        # Find settling time (when error stays below threshold)
        error_norms = torch.norm(traj, dim=1)
        below_threshold = error_norms < (settling_time_threshold * initial_error_norm)
        
        if below_threshold.any():
            settling_idx = torch.where(below_threshold)[0][0].item()
            settling_time = settling_idx * dt
        else:
            settling_time = horizon * dt
        
        # Maximum control effort
        max_control = controls_single.abs().max().item() if i == 0 else float('nan')
        
        metrics.append({
            'ic_lateral': ic_tensor[i, 0].item(),
            'ic_heading': np.rad2deg(ic_tensor[i, 1].item()),
            'initial_error': initial_error_norm.item(),
            'final_error': final_error_norm.item(),
            'settling_time': settling_time,
            'reduction': (1 - final_error_norm / initial_error_norm).item() * 100
        })
    
    print(f"\n{'Initial Condition':<30} {'Settling Time':<15} {'Error Reduction':<15}")
    print("-" * 70)
    
    for m in metrics[:5]:  # Show first 5
        ic_str = f"d_e={m['ic_lateral']:+.1f}m, θ_e={m['ic_heading']:+.1f}°"
        print(f"{ic_str:<30} {m['settling_time']:>8.2f} s      {m['reduction']:>8.1f}%")
    
    print(f"... ({len(metrics) - 5} more trajectories)")
    
    # Overall statistics
    avg_settling = np.mean([m['settling_time'] for m in metrics])
    avg_reduction = np.mean([m['reduction'] for m in metrics])
    
    print(f"\nOverall Performance:")
    print(f"  Average settling time: {avg_settling:.2f} s")
    print(f"  Average error reduction: {avg_reduction:.1f}%")
    
    # ========================================================================
    # 10. Create Comparison Plot (Selected ICs)
    # ========================================================================
    print("\n11. Creating comparison plot for selected initial conditions...")
    
    # Select 4 representative trajectories
    selected_ics = [
        torch.tensor([3.0, np.deg2rad(30)], dtype=torch.float32),   # Large positive errors
        torch.tensor([-3.0, np.deg2rad(-30)], dtype=torch.float32), # Large negative errors
        torch.tensor([2.0, np.deg2rad(-20)], dtype=torch.float32),  # Mixed signs
        torch.tensor([-1.0, np.deg2rad(15)], dtype=torch.float32)   # Smaller errors
    ]
    
    selected_names = [
        'Large Positive (d=+3m, θ=+30°)',
        'Large Negative (d=-3m, θ=-30°)',
        'Mixed Signs (d=+2m, θ=-20°)',
        'Small Errors (d=-1m, θ=+15°)'
    ]
    
    # Simulate selected trajectories
    selected_tensor = torch.stack(selected_ics)
    selected_trajs = discrete_system.simulate(
        selected_tensor,
        controller=controller,
        horizon=horizon
    )
    
    # Plot with custom names and colors
    discrete_system.plot_trajectory(
        selected_trajs,
        state_names=['Lateral Error d_e [m]', 'Heading Error θ_e [rad]'],
        trajectory_names=selected_names,
        title='Path Tracking: Comparison of Representative Initial Conditions',
        colorway='Vivid',
        save_html='path_tracking_comparison.html',
        show=False
    )
    
    print("  ✓ Saved: path_tracking_comparison.html")
    
    # ========================================================================
    # 11. Phase Portrait with Selected Trajectories
    # ========================================================================
    print("\n12. Creating phase portrait with selected trajectories...")
    
    discrete_system.plot_phase_portrait_2d(
        selected_trajs,
        state_indices=(0, 1),
        state_names=('Lateral Error d_e [m]', 'Heading Error θ_e [rad]'),
        trajectory_names=selected_names,
        title='Path Tracking: Phase Portrait (Selected ICs)',
        colorway='Vivid',
        save_html='path_tracking_phase_selected.html',
        show=False
    )
    
    print("  ✓ Saved: path_tracking_phase_selected.html")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*70)
    print("Demo Complete! Generated Files:")
    print("="*70)
    print("  1. path_tracking_2d_trajectory_states.html - 2D trajectory (states only)")
    print("  2. path_tracking_2d_trajectory_with_control.html - 2D trajectory with control")
    print("  3. path_tracking_phase_portrait_2d.html - 2D phase portrait (all ICs)")
    print("  4. path_tracking_3d_trajectory_time.html - 3D trajectory with time coloring")
    print("  5. path_tracking_phase_portrait_3d.html - 3D phase portrait (subset)")
    print("  6. path_tracking_comparison.html - Comparison of selected ICs (states only)")
    print("  7. path_tracking_phase_selected.html - Phase portrait of selected ICs")
    print("="*70)
    print("\nKey Differences:")
    print("  • States only plots: Cleaner view of state evolution")
    print("  • States + control plots: See control effort alongside state response")
    print("="*70)
    print("\nAll plots are interactive HTML files - open them in your browser!")
    print("You can:")
    print("  • Zoom, pan, and rotate (3D plots)")
    print("  • Hover over trajectories for detailed information")
    print("  • Toggle trajectories on/off by clicking legend entries")
    print("  • Save plots as static images using the camera icon")
    print("="*70)

if __name__ == "__main__":
    main()