"""
Lyapunov Function and Region of Attraction Visualization

This module provides comprehensive visualization tools for:
- Lyapunov function value fields
- Regions of Attraction (ROA)
- Closed-loop trajectories with neural controllers
- Lyapunov derivative fields
"""

import torch
import numpy as np
from typing import Optional, Tuple, List, Callable
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_lyapunov_2d(
    lyapunov_nn,
    controller_nn,
    dynamics_system,
    state_limits: Tuple[Tuple[float, float], Tuple[float, float]],
    state_indices: Tuple[int, int] = (0, 1),
    state_names: Optional[Tuple[str, str]] = None,
    rho: Optional[float] = None,
    grid_resolution: int = 100,
    observer_nn = None,
    trajectories: Optional[List[torch.Tensor]] = None,
    title: Optional[str] = None,
    save_html: Optional[str] = None,
    show: bool = True,
    colorscale: str = 'Viridis'
):
    """
    Plot Lyapunov function value field and Region of Attraction in 2D
    
    Args:
        lyapunov_nn: Neural network Lyapunov function V(x)
        controller_nn: Neural network controller u = π(x) or π(x̂)
        dynamics_system: Dynamical system (GenericDiscreteTimeSystem)
        state_limits: ((x_min, x_max), (y_min, y_max)) for the two plotted states
        state_indices: Which two state dimensions to plot
        state_names: Names for the axes
        rho: ROA threshold (if None, compute from boundary)
        grid_resolution: Number of grid points per dimension
        observer_nn: Optional observer for output feedback (x̂ = obs(y))
        trajectories: Optional list of trajectories to overlay
        title: Plot title
        save_html: Filename to save interactive HTML
        show: Whether to display the plot
        colorscale: Plotly colorscale name
    
    Returns:
        Plotly figure object
    """
    device = next(lyapunov_nn.parameters()).device if hasattr(lyapunov_nn, 'parameters') else 'cpu'
    
    # Create grid
    idx0, idx1 = state_indices
    x0_range = np.linspace(state_limits[0][0], state_limits[0][1], grid_resolution)
    x1_range = np.linspace(state_limits[1][0], state_limits[1][1], grid_resolution)
    X0, X1 = np.meshgrid(x0_range, x1_range)
    
    # Initialize state grid
    nx = dynamics_system.nx
    states_grid = torch.zeros((grid_resolution * grid_resolution, nx), device=device)
    
    # Fill in the two dimensions we're plotting
    states_grid[:, idx0] = torch.tensor(X0.flatten(), dtype=torch.float32, device=device)
    states_grid[:, idx1] = torch.tensor(X1.flatten(), dtype=torch.float32, device=device)
    
    # Other dimensions set to equilibrium or zero
    x_eq = dynamics_system.x_equilibrium.to(device)
    for i in range(nx):
        if i not in state_indices:
            states_grid[:, i] = x_eq[i]
    
    # Evaluate Lyapunov function
    with torch.no_grad():
        V_values = lyapunov_nn(states_grid).squeeze()
        V_grid = V_values.reshape(grid_resolution, grid_resolution).cpu().numpy()
    
    # Compute Lyapunov derivative (V̇)
    with torch.no_grad():
        if observer_nn is not None:
            # Output feedback: y = h(x), x̂ = obs(y), u = π(x̂)
            y = dynamics_system.continuous_time_system.h(states_grid)
            x_hat = observer_nn(y)
            u = controller_nn(x_hat)
        else:
            # State feedback: u = π(x)
            u = controller_nn(states_grid)
        
        # Compute x_next and Lyapunov derivative
        x_next = dynamics_system(states_grid, u)
        V_next = lyapunov_nn(x_next).squeeze()
        V_dot = V_next - V_values  # Discrete-time Lyapunov derivative
        V_dot_grid = V_dot.reshape(grid_resolution, grid_resolution).cpu().numpy()
    
    # Determine ROA threshold
    if rho is None:
        # Compute rho from boundary values
        boundary_mask = (
            (states_grid[:, idx0] == state_limits[0][0]) |
            (states_grid[:, idx0] == state_limits[0][1]) |
            (states_grid[:, idx1] == state_limits[1][0]) |
            (states_grid[:, idx1] == state_limits[1][1])
        )
        if boundary_mask.any():
            rho = V_values[boundary_mask].min().item()
        else:
            rho = V_values.max().item() * 0.8
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Lyapunov Function V(x)', 'Lyapunov Derivative ΔV(x)'),
        specs=[[{'type': 'contour'}, {'type': 'contour'}]]
    )
    
    # Plot 1: Lyapunov function
    fig.add_trace(
        go.Contour(
            x=x0_range,
            y=x1_range,
            z=V_grid,
            colorscale=colorscale,
            contours=dict(
                start=0,
                end=V_grid.max(),
                size=V_grid.max() / 20,
            ),
            colorbar=dict(title="V(x)", x=0.45),
            hovertemplate='%{xaxis.title.text}: %{x:.3f}<br>%{yaxis.title.text}: %{y:.3f}<br>V: %{z:.3f}<extra></extra>',
        ),
        row=1, col=1
    )
    
    # Add ROA boundary (V(x) = rho)
    fig.add_trace(
        go.Contour(
            x=x0_range,
            y=x1_range,
            z=V_grid,
            contours=dict(
                start=rho,
                end=rho,
                size=1,
                coloring='none'
            ),
            line=dict(color='red', width=4),
            showscale=False,
            name=f'ROA (ρ={rho:.3f})',
            hovertemplate='ROA Boundary<extra></extra>',
        ),
        row=1, col=1
    )
    
    # Plot 2: Lyapunov derivative
    fig.add_trace(
        go.Contour(
            x=x0_range,
            y=x1_range,
            z=V_dot_grid,
            colorscale='RdBu_r',  # Red for positive, blue for negative
            contours=dict(
                start=V_dot_grid.min(),
                end=V_dot_grid.max(),
                size=(V_dot_grid.max() - V_dot_grid.min()) / 20,
            ),
            colorbar=dict(title="ΔV(x)", x=1.05),
            hovertemplate='%{xaxis.title.text}: %{x:.3f}<br>%{yaxis.title.text}: %{y:.3f}<br>ΔV: %{z:.3f}<extra></extra>',
        ),
        row=1, col=2
    )
    
    # Add zero contour for V_dot (should be negative everywhere in ROA)
    fig.add_trace(
        go.Contour(
            x=x0_range,
            y=x1_range,
            z=V_dot_grid,
            contours=dict(
                start=0,
                end=0,
                size=1,
                coloring='none'
            ),
            line=dict(color='black', width=3, dash='dash'),
            showscale=False,
            name='ΔV=0',
            hovertemplate='ΔV=0 Contour<extra></extra>',
        ),
        row=1, col=2
    )
    
    # Add equilibrium point to both plots
    x_eq_np = x_eq.cpu().numpy()
    for col in [1, 2]:
        fig.add_trace(
            go.Scatter(
                x=[x_eq_np[idx0]],
                y=[x_eq_np[idx1]],
                mode='markers',
                marker=dict(size=12, color='lime', symbol='star', line=dict(width=2, color='black')),
                name='Equilibrium',
                showlegend=(col == 1),
                hovertemplate='Equilibrium<extra></extra>',
            ),
            row=1, col=col
        )
    
    # Overlay trajectories if provided
    if trajectories is not None:
        colors = ['white', 'yellow', 'cyan', 'magenta', 'orange']
        for i, traj in enumerate(trajectories):
            traj_np = traj.detach().cpu().numpy()
            color = colors[i % len(colors)]
            
            for col in [1, 2]:
                # Trajectory line
                fig.add_trace(
                    go.Scatter(
                        x=traj_np[:, idx0],
                        y=traj_np[:, idx1],
                        mode='lines',
                        line=dict(color=color, width=2),
                        name=f'Trajectory {i+1}',
                        showlegend=(col == 1),
                        hovertemplate=f'Traj {i+1}<extra></extra>',
                    ),
                    row=1, col=col
                )
                
                # Start point
                fig.add_trace(
                    go.Scatter(
                        x=[traj_np[0, idx0]],
                        y=[traj_np[0, idx1]],
                        mode='markers',
                        marker=dict(size=10, color=color, symbol='circle'),
                        showlegend=False,
                        hovertemplate='Start<extra></extra>',
                    ),
                    row=1, col=col
                )
    
    # Update axes
    if state_names is None:
        state_names = (f'x{idx0}', f'x{idx1}')
    
    fig.update_xaxes(title_text=state_names[0], row=1, col=1)
    fig.update_yaxes(title_text=state_names[1], row=1, col=1)
    fig.update_xaxes(title_text=state_names[0], row=1, col=2)
    fig.update_yaxes(title_text=state_names[1], row=1, col=2)
    
    # Update layout
    if title is None:
        title = "Lyapunov Function and Region of Attraction"
    
    fig.update_layout(
        title=title,
        height=600,
        width=1500,
        hovermode='closest',
        showlegend=True,
        legend=dict(
            x=1.12,
            y=0.5,
            xanchor='left',
            yanchor='middle',
            bgcolor='rgba(255,255,255,0.95)',
            bordercolor='black',
            borderwidth=1
        ),
        margin=dict(l=50, r=200, t=80, b=50)
    )
    
    if save_html:
        fig.write_html(save_html)
        print(f"Lyapunov visualization saved to {save_html}")
    
    if show:
        fig.show()
    
    return fig


def plot_lyapunov_3d_surface(
    lyapunov_nn,
    state_limits: Tuple[Tuple[float, float], Tuple[float, float]],
    state_indices: Tuple[int, int] = (0, 1),
    state_names: Optional[Tuple[str, str]] = None,
    rho: Optional[float] = None,
    grid_resolution: int = 80,
    nx: int = 2,
    x_equilibrium: Optional[torch.Tensor] = None,
    title: Optional[str] = None,
    save_html: Optional[str] = None,
    show: bool = True,
    colorscale: str = 'Viridis'
):
    """
    Plot Lyapunov function as a 3D surface
    
    Args:
        lyapunov_nn: Neural network Lyapunov function
        state_limits: Limits for the two plotted states
        state_indices: Which two states to plot
        state_names: Names for axes
        rho: ROA threshold to highlight
        grid_resolution: Grid density
        nx: Total state dimension
        x_equilibrium: Equilibrium point
        title: Plot title
        save_html: Filename to save
        show: Whether to display
    
    Returns:
        Plotly figure
    """
    device = next(lyapunov_nn.parameters()).device if hasattr(lyapunov_nn, 'parameters') else 'cpu'
    
    # Create grid
    idx0, idx1 = state_indices
    x0_range = np.linspace(state_limits[0][0], state_limits[0][1], grid_resolution)
    x1_range = np.linspace(state_limits[1][0], state_limits[1][1], grid_resolution)
    X0, X1 = np.meshgrid(x0_range, x1_range)
    
    # Initialize state grid
    states_grid = torch.zeros((grid_resolution * grid_resolution, nx), device=device)
    states_grid[:, idx0] = torch.tensor(X0.flatten(), dtype=torch.float32, device=device)
    states_grid[:, idx1] = torch.tensor(X1.flatten(), dtype=torch.float32, device=device)
    
    # Set other dimensions to equilibrium
    if x_equilibrium is not None:
        for i in range(nx):
            if i not in state_indices:
                states_grid[:, i] = x_equilibrium[i]
    
    # Evaluate Lyapunov function
    with torch.no_grad():
        V_values = lyapunov_nn(states_grid).squeeze()
        V_grid = V_values.reshape(grid_resolution, grid_resolution).cpu().numpy()
    
    # Create 3D surface plot
    fig = go.Figure()
    
    # Lyapunov surface
    fig.add_trace(go.Surface(
        x=X0,
        y=X1,
        z=V_grid,
        colorscale=colorscale,
        name='V(x)',
        colorbar=dict(title="V(x)"),
        hovertemplate='%{xaxis.title.text}: %{x:.3f}<br>%{yaxis.title.text}: %{y:.3f}<br>V: %{z:.3f}<extra></extra>',
    ))
    
    # Add ROA plane at V = rho
    if rho is not None:
        rho_plane = np.full_like(V_grid, rho)
        fig.add_trace(go.Surface(
            x=X0,
            y=X1,
            z=rho_plane,
            opacity=0.5,
            colorscale=[[0, 'red'], [1, 'red']],
            showscale=False,
            name=f'ROA (ρ={rho:.3f})',
            hovertemplate=f'ROA threshold: {rho:.3f}<extra></extra>',
        ))
    
    # Add equilibrium point
    if x_equilibrium is not None:
        x_eq_np = x_equilibrium.cpu().numpy()
        V_eq = lyapunov_nn(x_equilibrium.unsqueeze(0).to(device)).item()
        fig.add_trace(go.Scatter3d(
            x=[x_eq_np[idx0]],
            y=[x_eq_np[idx1]],
            z=[V_eq],
            mode='markers',
            marker=dict(size=8, color='lime', symbol='diamond'),
            name='Equilibrium',
            hovertemplate='Equilibrium<extra></extra>',
        ))
    
    # Set axis labels
    if state_names is None:
        state_names = (f'x{idx0}', f'x{idx1}')
    
    if title is None:
        title = "Lyapunov Function V(x)"
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=state_names[0],
            yaxis_title=state_names[1],
            zaxis_title='V(x)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
        ),
        height=700,
        showlegend=True,
    )
    
    if save_html:
        fig.write_html(save_html)
        print(f"3D Lyapunov surface saved to {save_html}")
    
    if show:
        fig.show()
    
    return fig


def plot_roa_analysis(
    lyapunov_nn,
    controller_nn,
    dynamics_system,
    state_limits: Tuple[Tuple[float, float], Tuple[float, float]],
    state_indices: Tuple[int, int] = (0, 1),
    state_names: Optional[Tuple[str, str]] = None,
    rho: Optional[float] = None,
    grid_resolution: int = 100,
    observer_nn = None,
    num_trajectories: int = 5,
    trajectory_steps: int = 100,
    title: Optional[str] = None,
    save_html: Optional[str] = None,
    show: bool = True
):
    """
    Comprehensive ROA analysis with multiple views
    
    Creates a 2x2 plot showing:
    - Top-left: Lyapunov function with ROA
    - Top-right: Lyapunov derivative
    - Bottom-left: Phase portrait with trajectories
    - Bottom-right: Control effort field
    
    Args:
        lyapunov_nn: Lyapunov function
        controller_nn: Neural controller
        dynamics_system: Discrete-time system
        state_limits: Plotting limits
        state_indices: Which states to plot
        state_names: State names
        rho: ROA threshold
        grid_resolution: Grid density
        observer_nn: Optional observer
        num_trajectories: Number of test trajectories
        trajectory_steps: Steps per trajectory
        title: Main title
        save_html: Save location
        show: Display plot
    
    Returns:
        Plotly figure
    """
    device = next(lyapunov_nn.parameters()).device if hasattr(lyapunov_nn, 'parameters') else 'cpu'
    
    # Create grid
    idx0, idx1 = state_indices
    x0_range = np.linspace(state_limits[0][0], state_limits[0][1], grid_resolution)
    x1_range = np.linspace(state_limits[1][0], state_limits[1][1], grid_resolution)
    X0, X1 = np.meshgrid(x0_range, x1_range)
    
    nx = dynamics_system.nx
    states_grid = torch.zeros((grid_resolution * grid_resolution, nx), device=device)
    states_grid[:, idx0] = torch.tensor(X0.flatten(), dtype=torch.float32, device=device)
    states_grid[:, idx1] = torch.tensor(X1.flatten(), dtype=torch.float32, device=device)
    
    x_eq = dynamics_system.x_equilibrium.to(device)
    for i in range(nx):
        if i not in state_indices:
            states_grid[:, i] = x_eq[i]
    
    # Evaluate functions
    with torch.no_grad():
        V_values = lyapunov_nn(states_grid).squeeze()
        V_grid = V_values.reshape(grid_resolution, grid_resolution).cpu().numpy()
        
        if observer_nn is not None:
            y = dynamics_system.continuous_time_system.h(states_grid)
            x_hat = observer_nn(y)
            u = controller_nn(x_hat)
        else:
            u = controller_nn(states_grid)
        
        u_norm = torch.norm(u, dim=1)
        u_norm_grid = u_norm.reshape(grid_resolution, grid_resolution).cpu().numpy()
        
        x_next = dynamics_system(states_grid, u)
        V_next = lyapunov_nn(x_next).squeeze()
        V_dot = V_next - V_values
        V_dot_grid = V_dot.reshape(grid_resolution, grid_resolution).cpu().numpy()
    
    # Determine rho
    if rho is None:
        boundary_mask = (
            (states_grid[:, idx0] == state_limits[0][0]) |
            (states_grid[:, idx0] == state_limits[0][1]) |
            (states_grid[:, idx1] == state_limits[1][0]) |
            (states_grid[:, idx1] == state_limits[1][1])
        )
        if boundary_mask.any():
            rho = V_values[boundary_mask].min().item()
        else:
            rho = V_values.quantile(0.7).item()
    
    # Generate test trajectories
    trajectories = []
    x0_samples = torch.zeros((num_trajectories, nx), device=device)
    for i in range(num_trajectories):
        # Sample initial conditions within ROA estimate
        angle = 2 * np.pi * i / num_trajectories
        radius = 0.6 * (state_limits[0][1] - state_limits[0][0]) / 2
        x0_samples[i, idx0] = radius * np.cos(angle)
        x0_samples[i, idx1] = radius * np.sin(angle)
        for j in range(nx):
            if j not in state_indices:
                x0_samples[i, j] = x_eq[j]
    
    u_seq = torch.zeros(num_trajectories, trajectory_steps, dynamics_system.nu, device=device)
    trajs = dynamics_system.simulate(x0_samples, u_seq, return_all=True)
    
    # Create 2x2 subplot
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Lyapunov Function V(x)',
            'Lyapunov Derivative ΔV(x)',
            'Closed-Loop Phase Portrait',
            'Control Effort ||u(x)||'
        ),
        specs=[[{'type': 'contour'}, {'type': 'contour'}],
               [{'type': 'scatter'}, {'type': 'contour'}]]
    )
    
    # Plot 1: Lyapunov function
    fig.add_trace(
        go.Contour(
            x=x0_range, y=x1_range, z=V_grid,
            colorscale='Viridis',
            colorbar=dict(title="V(x)", x=0.44, len=0.4, y=0.77, thickness=15),
            showscale=True,
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Contour(
            x=x0_range, y=x1_range, z=V_grid,
            contours=dict(start=rho, end=rho, size=1, coloring='none'),
            line=dict(color='red', width=3),
            showscale=False,
            name=f'ROA (ρ={rho:.3f})',
        ),
        row=1, col=1
    )
    
    # Plot 2: Lyapunov derivative
    fig.add_trace(
        go.Contour(
            x=x0_range, y=x1_range, z=V_dot_grid,
            colorscale='RdBu_r',
            colorbar=dict(title="ΔV(x)", x=1.01, len=0.4, y=0.77, thickness=15),
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Contour(
            x=x0_range, y=x1_range, z=V_dot_grid,
            contours=dict(start=0, end=0, size=1, coloring='none'),
            line=dict(color='black', width=2, dash='dash'),
            showscale=False,
        ),
        row=1, col=2
    )
    
    # Plot 3: Phase portrait with trajectories
    trajs_np = trajs.detach().cpu().numpy()
    colors = ['cyan', 'yellow', 'magenta', 'lime', 'orange']
    
    for i in range(num_trajectories):
        fig.add_trace(
            go.Scatter(
                x=trajs_np[i, :, idx0],
                y=trajs_np[i, :, idx1],
                mode='lines+markers',
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=3),
                name=f'Trajectory {i+1}',
            ),
            row=2, col=1
        )
    
    # Plot 4: Control effort
    fig.add_trace(
        go.Contour(
            x=x0_range, y=x1_range, z=u_norm_grid,
            colorscale='Plasma',
            colorbar=dict(title="||u||", x=1.01, len=0.4, y=0.23, thickness=15),
        ),
        row=2, col=2
    )
    
    # Add equilibrium to all plots
    x_eq_np = x_eq.cpu().numpy()
    for row_i, col_i in [(1, 1), (1, 2), (2, 1), (2, 2)]:
        fig.add_trace(
            go.Scatter(
                x=[x_eq_np[idx0]],
                y=[x_eq_np[idx1]],
                mode='markers',
                marker=dict(size=10, color='lime', symbol='star', line=dict(width=2, color='black')),
                showlegend=False,
            ),
            row=row_i, col=col_i
        )
    
    # Update axes
    if state_names is None:
        state_names = (f'x{idx0}', f'x{idx1}')
    
    for col_i in [1, 2]:
        for row_i in [1, 2]:
            fig.update_xaxes(title_text=state_names[0], row=row_i, col=col_i)
            fig.update_yaxes(title_text=state_names[1], row=row_i, col=col_i)
    
    # Update layout
    if title is None:
        title = "Comprehensive ROA Analysis"
    
    fig.update_layout(
        title=title,
        height=1050,
        width=1650,
        showlegend=True,
        legend=dict(
            x=0.5,
            y=-0.06,
            xanchor='center',
            yanchor='top',
            orientation='h',
            bgcolor='rgba(255,255,255,0.95)',
            bordercolor='black',
            borderwidth=1,
            font=dict(size=10)
        ),
        margin=dict(l=60, r=180, t=90, b=120)
    )
    
    if save_html:
        fig.write_html(save_html)
        print(f"ROA analysis saved to {save_html}")
    
    if show:
        fig.show()
    
    return fig


def plot_lyapunov_animation(
    lyapunov_nn,
    controller_nn,
    dynamics_system,
    x0: torch.Tensor,
    num_steps: int = 200,
    state_limits: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    state_indices: Tuple[int, int] = (0, 1),
    state_names: Optional[Tuple[str, str]] = None,
    rho: Optional[float] = None,
    grid_resolution: int = 60,
    observer_nn = None,
    title: Optional[str] = None,
    save_html: Optional[str] = None,
    show: bool = True
):
    """
    Create animated trajectory on Lyapunov function contour plot
    
    Args:
        lyapunov_nn: Lyapunov function
        controller_nn: Controller
        dynamics_system: System dynamics
        x0: Initial state
        num_steps: Animation steps
        state_limits: Plot limits (auto-computed if None)
        state_indices: Which states to plot
        state_names: State names
        rho: ROA threshold
        grid_resolution: Grid density
        observer_nn: Optional observer
        title: Plot title
        save_html: Save location
        show: Display plot
    
    Returns:
        Plotly figure with animation
    """
    device = next(lyapunov_nn.parameters()).device if hasattr(lyapunov_nn, 'parameters') else torch.device('cpu')
    
    # Simulate trajectory
    u_seq = torch.zeros(num_steps, dynamics_system.nu, device=device)
    trajectory = dynamics_system.simulate(x0.to(device), u_seq, return_all=True)
    traj_np = trajectory.detach().cpu().numpy()
    
    # Determine limits if not provided
    idx0, idx1 = state_indices
    if state_limits is None:
        margin = 0.2
        x0_min, x0_max = traj_np[:, idx0].min(), traj_np[:, idx0].max()
        x1_min, x1_max = traj_np[:, idx1].min(), traj_np[:, idx1].max()
        x0_range_val = x0_max - x0_min
        x1_range_val = x1_max - x1_min
        state_limits = (
            (x0_min - margin * x0_range_val, x0_max + margin * x0_range_val),
            (x1_min - margin * x1_range_val, x1_max + margin * x1_range_val)
        )
    
    x0_range = np.linspace(state_limits[0][0], state_limits[0][1], grid_resolution)
    x1_range = np.linspace(state_limits[1][0], state_limits[1][1], grid_resolution)
    X0, X1 = np.meshgrid(x0_range, x1_range)
    
    # Evaluate Lyapunov function on grid
    nx = dynamics_system.nx
    states_grid = torch.zeros((grid_resolution * grid_resolution, nx), device=device)
    states_grid[:, idx0] = torch.tensor(X0.flatten(), dtype=torch.float32, device=device)
    states_grid[:, idx1] = torch.tensor(X1.flatten(), dtype=torch.float32, device=device)
    
    with torch.no_grad():
        V_values = lyapunov_nn(states_grid).squeeze()
        V_grid = V_values.reshape(grid_resolution, grid_resolution).cpu().numpy()
    
    # Create frames for animation
    frames = []
    skip = max(1, num_steps // 50)  # Max 50 frames
    for t in range(0, num_steps + 1, skip):
        frame_data = [
            go.Contour(
                x=x0_range, y=x1_range, z=V_grid,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="V(x)"),
            ),
            go.Scatter(
                x=traj_np[:t+1, idx0],
                y=traj_np[:t+1, idx1],
                mode='lines',
                line=dict(color='white', width=3),
                name='Trajectory',
            ),
            go.Scatter(
                x=[traj_np[t, idx0]],
                y=[traj_np[t, idx1]],
                mode='markers',
                marker=dict(size=15, color='red', symbol='circle'),
                name='Current State',
            )
        ]
        
        if rho is not None:
            frame_data.append(
                go.Contour(
                    x=x0_range, y=x1_range, z=V_grid,
                    contours=dict(start=rho, end=rho, size=1, coloring='none'),
                    line=dict(color='red', width=3),
                    showscale=False,
                )
            )
        
        frames.append(go.Frame(data=frame_data, name=str(t)))
    
    # Create figure with animation
    fig = go.Figure(
        data=frames[0].data,
        frames=frames
    )
    
    # Add play/pause buttons
    fig.update_layout(
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'y': 1.15,
            'x': 0.1,
            'buttons': [
                {
                    'label': '▶ Play',
                    'method': 'animate',
                    'args': [None, {'frame': {'duration': 50, 'redraw': True}, 'fromcurrent': True}]
                },
                {
                    'label': '⏸ Pause',
                    'method': 'animate',
                    'args': [[None], {'frame': {'duration': 0}, 'mode': 'immediate'}]
                }
            ]
        }]
    )
    
    if state_names is None:
        state_names = (f'x{idx0}', f'x{idx1}')
    
    if title is None:
        title = "Animated Trajectory on Lyapunov Function"
    
    fig.update_layout(
        title=title,
        xaxis_title=state_names[0],
        yaxis_title=state_names[1],
        height=700,
        width=800,
    )
    
    if save_html:
        fig.write_html(save_html)
        print(f"Animated trajectory saved to {save_html}")
    
    if show:
        fig.show()
    
    return fig