import torch
import numpy as np
from typing import Optional, Tuple, List, Callable
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch.nn as nn
from neural_lyapunov_training.symbolic_dynamics import (
    SymbolicPendulum,
    GenericDiscreteTimeSystem,
    IntegrationMethod,
)
import neural_lyapunov_training.lyapunov_roa_visualization as lrv

# Example usage function
def demo_lyapunov_visualization():
    """
    Demonstrate Lyapunov visualization with a simple example
    """
    

    # Create simple neural networks for demonstration
    class SimpleLyapunov(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 16),
                nn.Tanh(),
                nn.Linear(16, 16),
                nn.Tanh(),
                nn.Linear(16, 1),
            )

        def forward(self, x):
            return torch.sum(x**2, dim=1, keepdim=True) + 0.1 * self.net(x) ** 2

    class SimpleController(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1))

        def forward(self, x):
            return self.net(x)

    # Create system
    pendulum = SymbolicPendulum(m=1.0, l=1.0, beta=0.5, g=9.81)
    discrete_pendulum = GenericDiscreteTimeSystem(
        pendulum, dt=0.01, integration_method=IntegrationMethod.RK4
    )

    # Create neural networks
    lyap_nn = SimpleLyapunov()
    controller_nn = SimpleController()

    # Set to eval mode
    lyap_nn.eval()
    controller_nn.eval()

    print("Generating Lyapunov visualizations...")

    # 2D contour plot
    lrv.plot_lyapunov_2d(
        lyap_nn,
        controller_nn,
        discrete_pendulum,
        state_limits=((-1.0, 1.0), (-2.0, 2.0)),
        state_names=("theta (rad)", "omega (rad/s)"),
        rho=1.0,
        save_html="lyapunov_2d.html",
        show=False,
    )

    # 3D surface plot
    lrv.plot_lyapunov_3d_surface(
        lyap_nn,
        state_limits=((-1.0, 1.0), (-2.0, 2.0)),
        state_names=("theta (rad)", "omega (rad/s)"),
        rho=1.0,
        nx=2,
        x_equilibrium=pendulum.x_equilibrium,
        save_html="lyapunov_3d.html",
        show=False,
    )

    # Comprehensive ROA analysis
    lrv.plot_roa_analysis(
        lyap_nn,
        controller_nn,
        discrete_pendulum,
        state_limits=((-1.0, 1.0), (-2.0, 2.0)),
        state_names=("theta (rad)", "omega (rad/s)"),
        rho=1.0,
        num_trajectories=8,
        save_html="roa_analysis.html",
        show=False,
    )

    # Animated trajectory
    x0 = torch.tensor([0.8, 0.5])
    lrv.plot_lyapunov_animation(
        lyap_nn,
        controller_nn,
        discrete_pendulum,
        x0,
        num_steps=200,
        state_limits=((-1.0, 1.0), (-2.0, 2.0)),
        state_names=("theta (rad)", "omega (rad/s)"),
        rho=1.0,
        save_html="lyapunov_animation.html",
        show=False,
    )

    print("\n✓ All visualizations generated!")
    print("  - lyapunov_2d.html: 2D contour plot with ROA")
    print("  - lyapunov_3d.html: 3D surface plot")
    print("  - roa_analysis.html: Comprehensive 4-panel analysis")
    print("  - lyapunov_animation.html: Animated trajectory")
    print("\nOpen these HTML files in your browser!")

    
if __name__ == "__main__":
    # Run demo if this file is executed
    print("Run demo_lyapunov_visualization() to see examples")
    demo_lyapunov_visualization()