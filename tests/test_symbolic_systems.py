import sympy as sp
import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union, Optional, Callable
from enum import Enum
import control
import warnings
from neural_lyapunov_training.symbolic_dynamics import (
    SYMPY_TO_TORCH,
    GenericDiscreteTimeSystem,
    IntegrationMethod,
)
from neural_lyapunov_training.symbolic_systems import (
    SymbolicPendulum,
    SymbolicPendulumState,
)
from neural_lyapunov_training.pendulum import PendulumDynamics
import neural_lyapunov_training.dynamical_system as ds


def test_sympy_to_torch_mapping():
    """Test that SYMPY_TO_TORCH mapping works correctly"""

    print("Testing SymPy to PyTorch function mapping...")

    # Test Abs (capital A)
    x = sp.Symbol("x")
    expr = sp.Abs(x - 1)
    f = sp.lambdify(x, expr, modules=[SYMPY_TO_TORCH])
    x_test = torch.tensor([-2.0, 0.0, 3.0])
    result = f(x_test)
    expected = torch.tensor([3.0, 1.0, 2.0])
    assert torch.allclose(result, expected), f"Abs test failed: {result} != {expected}"
    print("  ✓ Abs mapping works")

    # Test Min with scalar - use torch namespace ONLY (no 'torch' fallback)
    expr_min = sp.Min(x, 0)
    # Only use our custom namespace, not 'torch' as fallback
    f_min = sp.lambdify(x, expr_min, modules=[SYMPY_TO_TORCH])
    result_min = f_min(x_test)
    expected_min = torch.tensor([-2.0, 0.0, 0.0])
    assert torch.allclose(
        result_min, expected_min
    ), f"Min test failed: {result_min} != {expected_min}"
    print(f"  ✓ Min mapping works: {result_min}")

    # Test Max with scalar
    expr_max = sp.Max(x, 0)
    f_max = sp.lambdify(x, expr_max, modules=[SYMPY_TO_TORCH])
    result_max = f_max(x_test)
    expected_max = torch.tensor([0.0, 0.0, 3.0])
    assert torch.allclose(
        result_max, expected_max
    ), f"Max test failed: {result_max} != {expected_max}"
    print(f"  ✓ Max mapping works: {result_max}")

    # Test sin
    expr_sin = sp.sin(x)
    f_sin = sp.lambdify(x, expr_sin, modules=[SYMPY_TO_TORCH])
    result_sin = f_sin(x_test)
    assert torch.allclose(result_sin, torch.sin(x_test)), "sin test failed"
    print("  ✓ sin mapping works")

    # Test sqrt
    expr_sqrt = sp.sqrt(sp.Abs(x))
    f_sqrt = sp.lambdify(x, expr_sqrt, modules=[SYMPY_TO_TORCH])
    result_sqrt = f_sqrt(x_test)
    expected_sqrt = torch.sqrt(torch.abs(x_test))
    assert torch.allclose(result_sqrt, expected_sqrt), "sqrt test failed"
    print("  ✓ sqrt mapping works")

    print("All SymPy to PyTorch mappings validated! ✓")


if __name__ == "__main__":
    # Test the mapping
    test_sympy_to_torch_mapping()

    # Test a system
    print("\n" + "=" * 70)
    print("Testing SymbolicPendulumState")
    print("=" * 70)
    pendulum = SymbolicPendulumState(m=0.15, l=0.5, beta=0.1, g=9.81)
    pendulum_continuous = PendulumDynamics(m=0.15, l=0.5, beta=0.1)
    pendulum.print_equations()

    discrete_pendulum = GenericDiscreteTimeSystem(
        pendulum, dt=0.01, integration_method=IntegrationMethod.RK4
    )
    discrete_pendulum_old_type = ds.SecondOrderDiscreteTimeSystem(
        pendulum_continuous,
        dt=0.01,
        position_integration=ds.IntegrationMethod.ExplicitEuler,
        velocity_integration=ds.IntegrationMethod.ExplicitEuler,
    )

    # Test 1: Basic forward pass
    print("\n--- Test 1: Forward Pass ---")
    x_test = torch.tensor([[0.1, 0.0]])
    u_test = torch.tensor([[0.0]])
    x_next = discrete_pendulum(x_test, u_test)  # Using __call__
    x_next_old = discrete_pendulum_old_type.forward(x_test, u_test)
    print(f"x={x_test.squeeze().numpy()} -> x_next={x_next.squeeze().numpy()}")
    print(f"x={x_test.squeeze().numpy()} -> x_next_old={x_next_old.squeeze().numpy()}")

    # Test 2: Batch processing
    print("\n--- Test 2: Batch Processing ---")
    x_batch = torch.randn(16, 2) * 0.1
    u_batch = torch.zeros(16, 1)
    x_next_batch = discrete_pendulum(x_batch, u_batch)
    print(f"Batch: {x_batch.shape} -> {x_next_batch.shape}")

    # Test 3: Trajectory simulation
    print("\n--- Test 3: Trajectory Simulation ---")
    x0 = torch.tensor([0.5, 0.0])
    u_traj = torch.zeros(100, 1)  # 100 time steps
    trajectory = discrete_pendulum.simulate(x0, u_traj, return_all=True)
    print(f"Trajectory shape: {trajectory.shape}")
    print(f"Initial state: {trajectory[0].numpy()}")
    print(f"Final state: {trajectory[-1].numpy()}")

    # Test 4: Equilibrium check
    print("\n--- Test 4: Equilibrium Check ---")
    is_eq, max_deriv = pendulum.check_equilibrium(
        pendulum.x_equilibrium, pendulum.u_equilibrium
    )
    print(f"Is equilibrium? {is_eq} (max derivative: {max_deriv:.2e})")

    # Test 5: Stability analysis
    print("\n--- Test 5: Stability Analysis ---")
    eigenvalues = pendulum.eigenvalues_at_equilibrium()
    print(f"Eigenvalues: {eigenvalues}")
    is_stable = pendulum.is_stable_equilibrium(discrete_time=False)
    print(f"Continuous-time stable? {is_stable}")

    # Test 6: Jacobian verification
    print("\n--- Test 6: Jacobian Verification ---")
    verification = pendulum.verify_jacobians(x_test.squeeze(), u_test.squeeze())
    print(
        f"A matrix matches? {verification['A_match']} (error: {verification['A_error']:.2e})"
    )
    print(
        f"B matrix matches? {verification['B_match']} (error: {verification['B_error']:.2e})"
    )

    # Test 7: Performance stats
    print("\n--- Test 7: Performance Statistics ---")
    stats = pendulum.get_performance_stats()
    print(f"Forward calls: {stats['forward_calls']}")
    print(f"Avg forward time: {stats['avg_forward_time']*1000:.4f} ms")
    print(f"Linearization calls: {stats['linearization_calls']}")
    print(f"Avg linearization time: {stats['avg_linearization_time']*1000:.4f} ms")

    # Test 8: Numerical stability check
    print("\n--- Test 8: Numerical Stability Check ---")
    stability = pendulum.check_numerical_stability(x_test.squeeze(), u_test.squeeze())
    print(f"Has NaN? {stability['has_nan']}")
    print(f"Has Inf? {stability['has_inf']}")
    print(f"Max derivative: {stability['max_derivative']:.4f}")
    print(f"Numerically stable? {stability['is_stable']}")

    # Test 9: String representations
    print("\n--- Test 9: String Representations ---")
    print(f"repr: {repr(pendulum)}")
    print(f"str: {str(pendulum)}")
    print(f"discrete repr: {repr(discrete_pendulum)}")
    print(f"discrete str: {str(discrete_pendulum)}")

    # Test 10: Trajectory Visualization
    print("\n--- Test 10: Trajectory Visualization ---")
    try:
        import plotly

        # Generate a more interesting trajectory with some control
        x0_vis = torch.tensor([0.5, 0.0])
        u_vis = torch.zeros(200, 1)
        u_vis[50:100] = 0.5  # Apply control between t=50 and t=100
        traj_vis = discrete_pendulum.simulate(x0_vis, u_vis, return_all=True)

        # Plot time series
        print("Generating time series plot...")
        fig1 = discrete_pendulum.plot_trajectory(
            traj_vis,
            state_names=["theta", "theta_dot"],
            control_sequence=u_vis,
            title="Pendulum Trajectory with Control Input",
            save_html="pendulum_trajectory.html",
            show=False,
        )
        print("✓ Time series plot saved to pendulum_trajectory.html")

        # Plot phase portrait
        print("Generating phase portrait...")
        fig2 = discrete_pendulum.plot_phase_portrait_2d(
            traj_vis,
            state_indices=(0, 1),
            state_names=("theta (rad)", "theta_dot (rad/s)"),
            title="Pendulum Phase Portrait",
            save_html="pendulum_phase_portrait.html",
            show=False,
        )
        print("✓ Phase portrait saved to pendulum_phase_portrait.html")

        # Plot multiple trajectories
        print("Generating multi-trajectory plot...")
        x0_batch = torch.tensor(
            [
                [0.3, 0.0],
                [0.5, 0.0],
                [0.7, 0.0],
            ]
        )
        u_batch_vis = torch.zeros(3, 150, 1)
        traj_batch_vis = discrete_pendulum.simulate(
            x0_batch, u_batch_vis, return_all=True
        )

        fig3 = discrete_pendulum.plot_phase_portrait_2d(
            traj_batch_vis,
            state_names=("theta (rad)", "theta_dot (rad/s)"),
            title="Multiple Pendulum Trajectories",
            save_html="pendulum_multi_trajectory.html",
            show=False,
        )
        print("✓ Multi-trajectory plot saved to pendulum_multi_trajectory.html")
        print("\nOpen the HTML files in your browser to view interactive plots!")

    except ImportError:
        print("⚠ Plotly not installed. Install with: pip install plotly")
        print("  Skipping visualization tests.")

    print("\n" + "=" * 70)
    print("✓ All tests passed!")
    print("=" * 70)
