"""
Test script for Lyapunov visualization functions

This script tests the visualization functions with both:
1. State feedback (simple pendulum)
2. Output feedback with observer (pendulum with partial observations)

Run with: python test_lyapunov_visualization.py
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add the neural_lyapunov_training module to path if needed
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import neural_lyapunov_training.lyapunov_roa_visualization as lrv
import neural_lyapunov_training.symbolic_systems as ss
import neural_lyapunov_training.symbolic_dynamics as sd
import neural_lyapunov_training.lyapunov as lyapunov
import neural_lyapunov_training.controllers as controllers

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float

print(f"Using device: {device}")

# ============================================================================
# Test 1: Simple pendulum with STATE FEEDBACK
# ============================================================================
print("\n" + "=" * 80)
print("TEST 1: State Feedback - Simple Pendulum")
print("=" * 80)

# Create a simple pendulum system
pendulum = ss.SymbolicPendulum(m=0.15, l=0.5, beta=0.1, g=9.81)
dt = 0.01
dynamics_state_fb = sd.GenericDiscreteTimeSystem(
    pendulum,
    dt=dt,
    integration_method=sd.IntegrationMethod.RK4,
    position_integration=sd.IntegrationMethod.ExplicitEuler,
)
dynamics_state_fb.to(device)

print(f"Pendulum dynamics: nx={dynamics_state_fb.nx}")
print(f"Equilibrium: {dynamics_state_fb.x_equilibrium}")

# Create a simple neural network controller for state feedback
controller_state_fb = controllers.NeuralNetworkController(
    nlayer=3,
    in_dim=2,  # Full state feedback: [theta, theta_dot]
    out_dim=1,
    hidden_dim=16,
    clip_output="clamp",
    u_lo=torch.tensor([-1.0]),
    u_up=torch.tensor([1.0]),
    x_equilibrium=torch.zeros(2, dtype=dtype),
    u_equilibrium=pendulum.u_equilibrium,
)
controller_state_fb.to(device)

# Create a simple quadratic Lyapunov function
Q = np.eye(2) * 10.0
R = np.eye(1) * 100.0
K, S = pendulum.lqr_control(Q=Q, R=R)
S_torch = torch.from_numpy(S).type(dtype).to(device)
R_chol = torch.linalg.cholesky(S_torch)

lyapunov_state_fb = lyapunov.NeuralNetworkQuadraticLyapunov(
    goal_state=torch.zeros(2, dtype=dtype).to(device),
    x_dim=2,
    R_rows=2,
    eps=0.01,
    R=R_chol,
)
lyapunov_state_fb.to(device)

# Define state limits for visualization
state_limits = ((-np.pi, np.pi), (-np.pi, np.pi))  # theta  # theta_dot

print("\nGenerating 2D visualization (state feedback)...")
try:
    fig_2d_state = lrv.plot_lyapunov_2d(
        lyapunov_nn=lyapunov_state_fb,
        controller_nn=controller_state_fb,
        dynamics_system=dynamics_state_fb,
        state_limits=state_limits,
        state_indices=(0, 1),
        state_names=("θ", "θ̇"),
        rho=5.0,
        grid_resolution=50,  # Lower resolution for faster testing
        observer_nn=None,  # State feedback - no observer
        title="State Feedback: Pendulum Lyapunov Function",
        save_html="test_state_fb_2d.html",
        show=False,
    )
    print("✓ 2D state feedback visualization created successfully!")
    print(f"  Saved to: test_state_fb_2d.html")
except Exception as e:
    print(f"✗ 2D state feedback visualization failed: {e}")
    import traceback

    traceback.print_exc()

print("\nGenerating 3D visualization (state feedback)...")
try:
    fig_3d_state = lrv.plot_lyapunov_3d_surface(
        lyapunov_nn=lyapunov_state_fb,
        controller_nn=controller_state_fb,
        dynamics_system=dynamics_state_fb,
        state_limits=state_limits,
        state_indices=(0, 1),
        state_names=("θ", "θ̇"),
        rho=5.0,
        grid_resolution=40,  # Lower resolution for faster testing
        observer_nn=None,  # State feedback - no observer
        title="State Feedback: Pendulum Lyapunov Function 3D",
        save_html="test_state_fb_3d.html",
        show=False,
        show_derivative=True,
    )
    print("✓ 3D state feedback visualization created successfully!")
    print(f"  Saved to: test_state_fb_3d.html")
except Exception as e:
    print(f"✗ 3D state feedback visualization failed: {e}")
    import traceback

    traceback.print_exc()

# ============================================================================
# Test 2: Pendulum with OUTPUT FEEDBACK (Observer-based)
# ============================================================================
print("\n" + "=" * 80)
print("TEST 2: Output Feedback - Pendulum with Observer")
print("=" * 80)

# Same pendulum, but now we'll use output feedback
pendulum_output = ss.SymbolicPendulum(m=0.15, l=0.5, beta=0.1, g=9.81)
dynamics_output_fb = sd.GenericDiscreteTimeSystem(
    pendulum_output,
    dt=dt,
    integration_method=sd.IntegrationMethod.RK4,
    position_integration=sd.IntegrationMethod.ExplicitEuler,
)
dynamics_output_fb.to(device)

print(f"Pendulum dynamics: nx={dynamics_output_fb.nx}")
print(f"Output dimension: ny={pendulum_output.ny}")

# First, let's check what the observer actually outputs
# Create a dummy observer to test
h = lambda x: pendulum_output.h(x)
test_observer = controllers.NeuralNetworkLuenbergerObserver(
    z_dim=2,
    y_dim=pendulum_output.ny,
    dynamics=dynamics_output_fb,
    h=h,
    zero_obs_error=torch.zeros(1, pendulum_output.ny),
    fc_hidden_dim=[16, 16],
)
test_observer.to(device)

# Test observer output dimension
with torch.no_grad():
    test_z = torch.zeros((1, 2), device=device)
    test_u = torch.zeros((1, 1), device=device)
    test_y = torch.zeros((1, pendulum_output.ny), device=device)
    test_output = test_observer(test_z, test_u, test_y)
    observer_output_dim = test_output.shape[1]
    print(f"Observer output dimension: {observer_output_dim}")

# Controller takes observer output
controller_output_fb = controllers.NeuralNetworkController(
    nlayer=3,
    in_dim=observer_output_dim,  # Match observer output dimension
    out_dim=1,
    hidden_dim=16,
    clip_output="clamp",
    u_lo=torch.tensor([-1.0]),
    u_up=torch.tensor([1.0]),
    x_equilibrium=torch.zeros(observer_output_dim, dtype=dtype),
    u_equilibrium=pendulum_output.u_equilibrium,
)
controller_output_fb.to(device)

# Create the actual observer for visualization
observer = test_observer  # Use the one we already created

# Lyapunov function for augmented state [x, e] where e is estimation error
# For output feedback, the Lyapunov function operates on 2*nx = 4 dimensions
S_cl = torch.cat(
    (
        torch.cat((S_torch / 10, torch.zeros(2, 2, device=device)), dim=1),
        torch.cat(
            (
                torch.zeros(2, 2, device=device),
                torch.eye(2, device=device) * 5.0,
            ),
            dim=1,
        ),
    ),
    dim=0,
)
R_cl = torch.linalg.cholesky(S_cl)

lyapunov_output_fb = lyapunov.NeuralNetworkQuadraticLyapunov(
    goal_state=torch.zeros(4, dtype=dtype).to(device),  # [x, e]
    x_dim=4,
    R_rows=4,
    eps=0.01,
    R=R_cl,
)
lyapunov_output_fb.to(device)

# State limits for visualization (only showing physical state dimensions)
state_limits_output = (
    (-np.pi / 2, np.pi / 2),  # theta
    (-np.pi / 2, np.pi / 2),  # theta_dot
)

print("\nGenerating 2D visualization (output feedback)...")
try:
    fig_2d_output = lrv.plot_lyapunov_2d(
        lyapunov_nn=lyapunov_output_fb,
        controller_nn=controller_output_fb,
        dynamics_system=dynamics_output_fb,
        state_limits=state_limits_output,
        state_indices=(0, 1),
        state_names=("θ", "θ̇"),
        rho=3.0,
        grid_resolution=50,
        observer_nn=observer,  # OUTPUT FEEDBACK - with observer
        title="Output Feedback: Pendulum Lyapunov Function with Observer",
        save_html="test_output_fb_2d.html",
        show=False,
    )
    print("✓ 2D output feedback visualization created successfully!")
    print(f"  Saved to: test_output_fb_2d.html")
    print(f"  Note: Lyapunov function operates on augmented state [x, e] (dim=4)")
    print(f"        Visualization shows physical state with e=0 (perfect estimation)")
except Exception as e:
    print(f"✗ 2D output feedback visualization failed: {e}")
    import traceback

    traceback.print_exc()

print("\nGenerating 3D visualization (output feedback)...")
try:
    fig_3d_output = lrv.plot_lyapunov_3d_surface(
        lyapunov_nn=lyapunov_output_fb,
        controller_nn=controller_output_fb,
        dynamics_system=dynamics_output_fb,
        state_limits=state_limits_output,
        state_indices=(0, 1),
        state_names=("θ", "θ̇"),
        rho=3.0,
        grid_resolution=40,
        observer_nn=observer,  # OUTPUT FEEDBACK - with observer
        title="Output Feedback: Pendulum Lyapunov Function 3D with Observer",
        save_html="test_output_fb_3d.html",
        show=False,
        show_derivative=True,
    )
    print("✓ 3D output feedback visualization created successfully!")
    print(f"  Saved to: test_output_fb_3d.html")
    print(f"  Note: Lyapunov function operates on augmented state [x, e] (dim=4)")
    print(f"        Visualization shows physical state with e=0 (perfect estimation)")
except Exception as e:
    print(f"✗ 3D output feedback visualization failed: {e}")
    import traceback

    traceback.print_exc()

# ============================================================================
# Test 3: Output feedback with non-zero estimation error
# ============================================================================
print("\n" + "=" * 80)
print("TEST 3: Output Feedback with Non-Zero Estimation Error")
print("=" * 80)

# Test with a specific estimation error
estimation_error_test = torch.tensor([0.1, 0.05], device=device, dtype=dtype)

print(f"\nTesting with estimation error: {estimation_error_test.cpu().numpy()}")

print("\nGenerating 2D visualization with estimation error...")
try:
    fig_2d_error = lrv.plot_lyapunov_2d(
        lyapunov_nn=lyapunov_output_fb,
        controller_nn=controller_output_fb,
        dynamics_system=dynamics_output_fb,
        state_limits=state_limits_output,
        state_indices=(0, 1),
        state_names=("θ", "θ̇"),
        rho=3.0,
        grid_resolution=50,
        observer_nn=observer,
        estimation_error=estimation_error_test,  # Non-zero estimation error
        title=f"Output Feedback with e=[{estimation_error_test[0]:.2f}, {estimation_error_test[1]:.2f}]",
        save_html="test_output_fb_with_error_2d.html",
        show=False,
    )
    print("✓ 2D visualization with estimation error created successfully!")
    print(f"  Saved to: test_output_fb_with_error_2d.html")
except Exception as e:
    print(f"✗ 2D visualization with estimation error failed: {e}")
    import traceback

    traceback.print_exc()

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)

files_created = []
for filename in [
    "test_state_fb_2d.html",
    "test_state_fb_3d.html",
    "test_output_fb_2d.html",
    "test_output_fb_3d.html",
    "test_output_fb_with_error_2d.html",
]:
    if os.path.exists(filename):
        files_created.append(filename)

print(f"\nFiles created ({len(files_created)}/5):")
for f in files_created:
    print(f"  ✓ {f}")

if len(files_created) == 5:
    print("\n🎉 All tests passed! All visualization files created successfully.")
else:
    print(f"\n⚠️  Only {len(files_created)}/5 tests passed. Check errors above.")

print("\nKey features tested:")
print("  1. State feedback (Lyapunov operates on physical state)")
print("  2. Output feedback with observer (Lyapunov operates on [x, e])")
print("  3. Automatic dimension inference")
print("  4. 2D contour plots with Lyapunov derivative")
print("  5. 3D surface plots")
print("  6. Custom estimation error specification")

print("\nOpen any .html file in a web browser to view interactive visualizations!")
print("=" * 80)
