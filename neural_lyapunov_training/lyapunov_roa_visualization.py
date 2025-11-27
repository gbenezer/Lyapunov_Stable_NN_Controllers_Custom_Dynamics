"""
Lyapunov Function and Region of Attraction Visualization

This module provides comprehensive visualization tools for:
- Lyapunov function value fields
- Regions of Attraction (ROA)
- Closed-loop trajectories with neural controllers
- Lyapunov derivative fields
- Automatic handling of output feedback (observer-based) systems
"""

import torch
import numpy as np
from typing import Optional, Tuple, List, Callable
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Convert state_limits to CPU numpy if they're tensors
def to_float(val):
    """Convert tensor or array to float"""
    if isinstance(val, torch.Tensor):
        return val.detach().cpu().item()
    elif isinstance(val, (np.ndarray, np.generic)):
        return float(val)
    else:
        return float(val)


def _infer_lyapunov_state_dim(lyapunov_nn, dynamics_system, observer_nn):
    """
    Infer the expected state dimension for the Lyapunov function.

    For output feedback systems with observers, the Lyapunov function
    typically operates on the augmented state [x, e] where:
    - x is the physical state (dim = nx)
    - e is the estimation error (dim = nx)

    Args:
        lyapunov_nn: Lyapunov neural network
        dynamics_system: Dynamical system
        observer_nn: Observer network (None for state feedback)

    Returns:
        Expected dimension for Lyapunov function input
    """
    # Get physical state dimension from underlying system
    if hasattr(dynamics_system, "continuous_time_system"):
        nx = dynamics_system.continuous_time_system.nx
    else:
        nx = dynamics_system.nx

    if observer_nn is None:
        # State feedback: Lyapunov operates on physical state only
        return nx
    else:
        # Output feedback: Lyapunov operates on [x, e]
        # The augmented state is [physical_state, estimation_error]
        return 2 * nx


def _create_augmented_state_grid(
    states_grid_physical, observer_nn, dynamics_system, device
):
    """
    Create augmented state grid [x, e] from physical state grid.

    For output feedback, we need to provide both the physical state x
    and the estimation error e to the Lyapunov function.

    Args:
        states_grid_physical: Grid of physical states (n_points, nx)
        observer_nn: Observer network
        dynamics_system: Dynamical system
        device: Torch device

    Returns:
        Augmented state grid (n_points, 2*nx)
    """
    nx = dynamics_system.nx
    n_points = states_grid_physical.shape[0]

    # For visualization, we typically want to show the behavior at zero estimation error
    # (i.e., when the observer has converged to the true state)
    # This shows the "ideal" behavior of the closed-loop system
    estimation_error = torch.zeros((n_points, nx), device=device)

    # Augmented state: [physical_state, estimation_error]
    augmented_states = torch.cat([states_grid_physical, estimation_error], dim=1)

    return augmented_states


def _prepare_state_grid(
    dynamics_system,
    observer_nn,
    state_limits: Tuple[Tuple[float, float], Tuple[float, float]],
    state_indices: Tuple[int, int],
    grid_resolution: int,
    estimation_error: Optional[torch.Tensor],
    device,
):
    """
    Prepare physical and Lyapunov state grids for visualization.

    Returns:
        Tuple of (states_grid_physical, states_grid_lyap, estimation_error_grid, X0, X1, x0_range, x1_range)
    """
    # Get the physical state dimension
    # For wrapped systems, use the underlying continuous system's dimension
    if hasattr(dynamics_system, "continuous_time_system"):
        nx = dynamics_system.continuous_time_system.nx
    else:
        nx = dynamics_system.nx

    idx0, idx1 = state_indices

    # Validate state indices are within physical state dimension
    if idx0 >= nx or idx1 >= nx:
        raise ValueError(
            f"state_indices {state_indices} must be < nx={nx}. "
            f"For output feedback, only plot physical state dimensions (0 to {nx-1}), "
            f"not augmented state dimensions."
        )

    # Create meshgrid
    x0_range = np.linspace(state_limits[0][0], state_limits[0][1], grid_resolution)
    x1_range = np.linspace(state_limits[1][0], state_limits[1][1], grid_resolution)
    X0, X1 = np.meshgrid(x0_range, x1_range)

    # Initialize physical state grid (only nx dimensions)
    states_grid_physical = torch.zeros(
        (grid_resolution * grid_resolution, nx), device=device
    )

    # Fill in the two dimensions we're plotting
    states_grid_physical[:, idx0] = torch.tensor(
        X0.flatten(), dtype=torch.float32, device=device
    )
    states_grid_physical[:, idx1] = torch.tensor(
        X1.flatten(), dtype=torch.float32, device=device
    )

    # Get equilibrium from the underlying system
    if hasattr(dynamics_system, "continuous_time_system"):
        x_eq = dynamics_system.continuous_time_system.x_equilibrium.to(device)
    else:
        x_eq = dynamics_system.x_equilibrium.to(device)

    # Debug: ensure x_eq has correct size
    if x_eq.shape[0] != nx:
        raise ValueError(
            f"Equilibrium dimension mismatch: x_eq has {x_eq.shape[0]} dimensions "
            f"but nx={nx}"
        )

    # Other physical state dimensions set to equilibrium
    for i in range(nx):
        if i not in state_indices:
            states_grid_physical[:, i] = x_eq[i]

    # Prepare estimation error grid
    estimation_error_grid = None
    if observer_nn is not None:
        n_points = states_grid_physical.shape[0]
        if estimation_error is None:
            # Default: zero estimation error (perfect estimation)
            estimation_error_grid = torch.zeros((n_points, nx), device=device)
        else:
            # User-specified estimation error
            if estimation_error.dim() == 1:
                estimation_error_grid = estimation_error.unsqueeze(0).expand(
                    n_points, -1
                )
            else:
                estimation_error_grid = estimation_error

    # Create Lyapunov input (augmented state if observer present)
    if observer_nn is not None:
        states_grid_lyap = torch.cat(
            [states_grid_physical, estimation_error_grid], dim=1
        )
    else:
        # State feedback: physical state = Lyapunov input
        states_grid_lyap = states_grid_physical

    return (
        states_grid_physical,
        states_grid_lyap,
        estimation_error_grid,
        X0,
        X1,
        x0_range,
        x1_range,
    )


def _compute_lyapunov_derivative(
    states_grid_physical,
    lyapunov_nn,
    controller_nn,
    observer_nn,
    dynamics_system,
    device,
    estimation_error_grid=None,
):
    """
    Compute Lyapunov derivative (ΔV) for discrete-time systems.

    For observers, this uses the provided estimation error (or assumes e=0).

    Args:
        estimation_error_grid: Current estimation error for each grid point
                              If None and observer present, assumes e=0

    Returns:
        Tuple of (V_values, V_dot)
    """
    with torch.no_grad():
        n_points = states_grid_physical.shape[0]

        # Get physical state dimension from underlying system
        if hasattr(dynamics_system, "continuous_time_system"):
            nx = dynamics_system.continuous_time_system.nx
        else:
            nx = dynamics_system.nx

        # Infer controller output dimension
        # Try various attributes the controller might have
        if hasattr(controller_nn, "out_dim"):
            u_dim = controller_nn.out_dim
        elif hasattr(controller_nn, "output_dim"):
            u_dim = controller_nn.output_dim
        elif hasattr(controller_nn, "u_up"):
            u_dim = controller_nn.u_up.shape[0]
        elif hasattr(controller_nn, "u_lo"):
            u_dim = controller_nn.u_lo.shape[0]
        else:
            # Default to 1 for scalar control
            u_dim = 1

        # Compute control input
        if observer_nn is not None:
            # Output feedback with observer
            # Current state estimate: z = x - e
            if estimation_error_grid is None:
                # Default: assume converged (e=0, so z=x)
                z_current = states_grid_physical
            else:
                # With estimation error: z = x - e
                z_current = states_grid_physical - estimation_error_grid

            # Get current measurement
            y = dynamics_system.continuous_time_system.h(states_grid_physical)

            # Initialize dummy control for first observer call
            u_dummy = torch.zeros((n_points, u_dim), device=device)

            # Get observer output (might be augmented for controller)
            z_estimate = observer_nn(z_current, u_dummy, y)

            # Check if we need to augment the observer output for the controller
            # The controller might expect [z, y] or some other augmentation
            controller_in_dim = (
                controller_nn.net[0].in_features
                if hasattr(controller_nn, "net")
                else None
            )

            print(f"z_estimate shape: {z_estimate.shape}")
            print(f"y shape: {y.shape}")
            print(f"Controller expects: {controller_in_dim}")
            
            if controller_in_dim is None:
                # Try to infer from x_equilibrium
                if hasattr(controller_nn, "x_equilibrium"):
                    controller_in_dim = controller_nn.x_equilibrium.shape[0]

            if (
                controller_in_dim is not None
                and z_estimate.shape[1] < controller_in_dim
            ):
                # Controller expects more inputs than observer provides
                # Common pattern: controller takes [z_estimate, y]
                # where y is the measurement used for control
                deficit = controller_in_dim - z_estimate.shape[1]

                # Augment with measurement (most common case)
                if deficit == y.shape[1]:
                    z_estimate = torch.cat([z_estimate, y], dim=1)
                else:
                    # Pad with zeros as fallback
                    padding = torch.zeros((n_points, deficit), device=device)
                    z_estimate = torch.cat([z_estimate, padding], dim=1)

            # Compute control based on observer estimate
            u = controller_nn(z_estimate)
        else:
            # State feedback: u = π(x)
            u = controller_nn(states_grid_physical)

        # Current Lyapunov value
        if observer_nn is not None:
            # For output feedback, Lyapunov operates on [x, e]
            if estimation_error_grid is None:
                e_current = torch.zeros((n_points, nx), device=device)
            else:
                e_current = estimation_error_grid
            states_current_lyap = torch.cat([states_grid_physical, e_current], dim=1)
        else:
            states_current_lyap = states_grid_physical
        V_values = lyapunov_nn(states_current_lyap).squeeze()

        # Compute x_next (next physical state)
        x_next = dynamics_system(states_grid_physical, u)

        # For observer case, need to compute augmented next state
        if observer_nn is not None:
            # Get next measurement
            y_next = dynamics_system.continuous_time_system.h(x_next)

            # Update observer estimate
            # z_next = observer(z_current, u, y_next)
            z_next = observer_nn(z_current, u, y_next)

            # The observer output might be augmented, extract state estimate
            if z_next.shape[1] > nx:
                # Observer output is augmented, extract state estimate
                z_next_state = z_next[:, :nx]
            else:
                z_next_state = z_next

            # Compute next estimation error: e_next = x_next - z_next
            e_next = x_next - z_next_state
            states_next_lyap = torch.cat([x_next, e_next], dim=1)
        else:
            states_next_lyap = x_next

        # Lyapunov derivative
        V_next = lyapunov_nn(states_next_lyap).squeeze()
        V_dot = V_next - V_values  # Discrete-time Lyapunov derivative

    return V_values, V_dot


def plot_lyapunov_2d(
    lyapunov_nn,
    controller_nn,
    dynamics_system,
    state_limits: Tuple[Tuple[float, float], Tuple[float, float]],
    state_indices: Tuple[int, int] = (0, 1),
    state_names: Optional[Tuple[str, str]] = None,
    rho: Optional[float] = None,
    grid_resolution: int = 100,
    observer_nn=None,
    trajectories: Optional[List[torch.Tensor]] = None,
    title: Optional[str] = None,
    save_html: Optional[str] = None,
    show: bool = True,
    colorscale: str = "Viridis",
    estimation_error: Optional[torch.Tensor] = None,
    trajectory_colorscale: str = "Plotly",
):
    """
    Plot Lyapunov function value field and Region of Attraction in 2D

    Automatically handles both state feedback and output feedback (observer-based) systems.
    For output feedback systems, the Lyapunov function operates on [x, e] where e is the
    estimation error. By default, e is set to zero for visualization (showing ideal behavior).

    Args:
        lyapunov_nn: Neural network Lyapunov function V(x) or V([x, e])
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
        estimation_error: Optional fixed estimation error for visualization
                         (default: zeros, meaning perfect state estimation)
        trajectory_colorscale: Plotly qualitative color sequence for trajectories
                              (e.g., "Plotly", "D3", "Vivid", "Dark24", "Set1")

    Returns:
        Plotly figure object
    """

    state_limits = tuple((to_float(lim[0]), to_float(lim[1])) for lim in state_limits)

    device = (
        next(lyapunov_nn.parameters()).device
        if hasattr(lyapunov_nn, "parameters")
        else "cpu"
    )
    idx0, idx1 = state_indices

    # Prepare state grids
    (
        states_grid_physical,
        states_grid_lyap,
        estimation_error_grid,
        X0,
        X1,
        x0_range,
        x1_range,
    ) = _prepare_state_grid(
        dynamics_system,
        observer_nn,
        state_limits,
        state_indices,
        grid_resolution,
        estimation_error,
        device,
    )

    # Evaluate Lyapunov function
    with torch.no_grad():
        V_values = lyapunov_nn(states_grid_lyap).squeeze()
        V_grid = V_values.reshape(grid_resolution, grid_resolution).cpu().numpy()

    # Compute Lyapunov derivative
    V_values_deriv, V_dot = _compute_lyapunov_derivative(
        states_grid_physical,
        lyapunov_nn,
        controller_nn,
        observer_nn,
        dynamics_system,
        device,
        estimation_error_grid,
    )
    V_dot_grid = V_dot.reshape(grid_resolution, grid_resolution).cpu().numpy()

    # Determine ROA threshold
    if rho is None:
        # Compute rho from boundary values
        boundary_mask = (
            (states_grid_physical[:, idx0] == state_limits[0][0])
            | (states_grid_physical[:, idx0] == state_limits[0][1])
            | (states_grid_physical[:, idx1] == state_limits[1][0])
            | (states_grid_physical[:, idx1] == state_limits[1][1])
        )
        if boundary_mask.any():
            rho = V_values[boundary_mask].min().item()
        else:
            rho = V_values.max().item() * 0.8

    # Create subplots
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Lyapunov Function V(x)", "Lyapunov Derivative ΔV(x)"),
        specs=[[{"type": "contour"}, {"type": "contour"}]],
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
            hovertemplate="%{xaxis.title.text}: %{x:.3f}<br>%{yaxis.title.text}: %{y:.3f}<br>V: %{z:.3f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Add ROA boundary (V(x) = rho)
    fig.add_trace(
        go.Contour(
            x=x0_range,
            y=x1_range,
            z=V_grid,
            contours=dict(start=rho, end=rho, size=1, coloring="none"),
            line=dict(color="red", width=4),
            showscale=False,
            name=f"ROA (ρ={rho:.3f})",
            hovertemplate="ROA Boundary<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Plot 2: Lyapunov derivative
    fig.add_trace(
        go.Contour(
            x=x0_range,
            y=x1_range,
            z=V_dot_grid,
            colorscale="RdBu_r",  # Red for positive, blue for negative
            contours=dict(
                start=V_dot_grid.min(),
                end=V_dot_grid.max(),
                size=(V_dot_grid.max() - V_dot_grid.min()) / 20,
            ),
            colorbar=dict(title="ΔV(x)", x=1.05),
            hovertemplate="%{xaxis.title.text}: %{x:.3f}<br>%{yaxis.title.text}: %{y:.3f}<br>ΔV: %{z:.3f}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    # Add zero contour for V_dot (should be negative everywhere in ROA)
    fig.add_trace(
        go.Contour(
            x=x0_range,
            y=x1_range,
            z=V_dot_grid,
            contours=dict(start=0, end=0, size=1, coloring="none"),
            line=dict(color="black", width=3, dash="dash"),
            showscale=False,
            name="ΔV=0",
            hovertemplate="ΔV=0 Contour<extra></extra>",
        ),
        row=1,
        col=2,
    )

    # Add equilibrium point to both plots
    x_eq = dynamics_system.x_equilibrium.to(device)
    x_eq_np = x_eq.cpu().numpy()
    for col in [1, 2]:
        fig.add_trace(
            go.Scatter(
                x=[x_eq_np[idx0]],
                y=[x_eq_np[idx1]],
                mode="markers",
                marker=dict(
                    size=12,
                    color="lime",
                    symbol="star",
                    line=dict(width=2, color="black"),
                ),
                name="Equilibrium",
                showlegend=(col == 1),
                hovertemplate="Equilibrium<extra></extra>",
            ),
            row=1,
            col=col,
        )

    # Overlay trajectories if provided
    if trajectories is not None:
        import plotly.express as px
        
        # Get the color sequence
        try:
            color_sequence = getattr(px.colors.qualitative, trajectory_colorscale)
        except AttributeError:
            print(f"Warning: Color sequence '{trajectory_colorscale}' not found. Using 'Plotly' instead.")
            color_sequence = px.colors.qualitative.Plotly
        
        for i, traj in enumerate(trajectories):
            traj_np = traj.detach().cpu().numpy()
            color = color_sequence[i % len(color_sequence)]

            for col in [1, 2]:
                # Trajectory line
                fig.add_trace(
                    go.Scatter(
                        x=traj_np[:, idx0],
                        y=traj_np[:, idx1],
                        mode="lines",
                        line=dict(color=color, width=2),
                        name=f"Trajectory {i+1}",
                        showlegend=(col == 1),
                        hovertemplate=f"Traj {i+1}<extra></extra>",
                    ),
                    row=1,
                    col=col,
                )

                # Start point
                fig.add_trace(
                    go.Scatter(
                        x=[traj_np[0, idx0]],
                        y=[traj_np[0, idx1]],
                        mode="markers",
                        marker=dict(size=10, color=color, symbol="circle"),
                        showlegend=False,
                        hovertemplate="Start<extra></extra>",
                    ),
                    row=1,
                    col=col,
                )

    # Update axes
    if state_names is None:
        state_names = (f"x{idx0}", f"x{idx1}")

    fig.update_xaxes(title_text=state_names[0], row=1, col=1)
    fig.update_yaxes(title_text=state_names[1], row=1, col=1)
    fig.update_xaxes(title_text=state_names[0], row=1, col=2)
    fig.update_yaxes(title_text=state_names[1], row=1, col=2)

    # Update layout
    if title is None:
        feedback_type = (
            "Output Feedback" if observer_nn is not None else "State Feedback"
        )
        title = f"Lyapunov Function and Region of Attraction ({feedback_type})"

    fig.update_layout(
        title=title,
        height=650,
        width=1600,
        hovermode="closest",
        showlegend=True,
        legend=dict(
            x=1.15,
            y=0.5,
            xanchor="left",
            yanchor="middle",
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=11),
        ),
        margin=dict(l=60, r=250, t=90, b=60),
    )

    # Add more horizontal spacing between subplots
    fig.update_xaxes(domain=[0.0, 0.42], row=1, col=1)
    fig.update_xaxes(domain=[0.59, 0.99], row=1, col=2)

    if save_html:
        fig.write_html(save_html)
        print(f"Lyapunov visualization saved to {save_html}")

    if show:
        fig.show()

    return fig


def plot_lyapunov_3d_surface(
    lyapunov_nn,
    state_limits: Tuple[Tuple[float, float], Tuple[float, float]],
    controller_nn=None,
    dynamics_system=None,
    observer_nn=None,
    state_indices: Tuple[int, int] = (0, 1),
    state_names: Optional[Tuple[str, str]] = None,
    rho: Optional[float] = None,
    grid_resolution: int = 80,
    title: Optional[str] = None,
    save_html: Optional[str] = None,
    show: bool = True,
    colorscale: str = "Viridis",
    show_derivative: bool = False,
    estimation_error: Optional[torch.Tensor] = None,
    trajectories: Optional[List[torch.Tensor]] = None,
    trajectory_colorscale: str = "Plotly",
):
    """
    Plot Lyapunov function as a 3D surface, optionally with derivative surface and trajectories

    Automatically handles both state feedback and output feedback (observer-based) systems.

    Args:
        lyapunov_nn: Neural network Lyapunov function
        state_limits: Limits for the two plotted states
        controller_nn: Optional controller (required if show_derivative=True)
        dynamics_system: Optional dynamics (required if show_derivative=True)
        observer_nn: Optional observer for output feedback
        state_indices: Which two states to plot
        state_names: Names for axes
        rho: ROA threshold to highlight
        grid_resolution: Grid density
        title: Plot title
        save_html: Filename to save
        show: Whether to display
        colorscale: Plotly colorscale
        show_derivative: If True, create side-by-side plot with V and ΔV
        estimation_error: Optional fixed estimation error for visualization
        trajectories: Optional list of trajectories (timesteps, state_dim) to overlay
        trajectory_colorscale: Plotly qualitative color sequence for trajectories
                              (e.g., "Plotly", "D3", "Vivid", "Dark24", "Set1")

    Returns:
        Plotly figure
    """

    state_limits = tuple((to_float(lim[0]), to_float(lim[1])) for lim in state_limits)

    device = (
        next(lyapunov_nn.parameters()).device
        if hasattr(lyapunov_nn, "parameters")
        else "cpu"
    )
    idx0, idx1 = state_indices

    # Prepare state grids
    (
        states_grid_physical,
        states_grid_lyap,
        estimation_error_grid,
        X0,
        X1,
        x0_range,
        x1_range,
    ) = _prepare_state_grid(
        dynamics_system,
        observer_nn,
        state_limits,
        state_indices,
        grid_resolution,
        estimation_error,
        device,
    )

    # Evaluate Lyapunov function
    with torch.no_grad():
        V_values = lyapunov_nn(states_grid_lyap).squeeze()
        V_grid = V_values.reshape(grid_resolution, grid_resolution).cpu().numpy()

    # Optionally compute Lyapunov derivative
    V_dot_grid = None
    if show_derivative:
        if controller_nn is None or dynamics_system is None:
            raise ValueError(
                "controller_nn and dynamics_system required when show_derivative=True"
            )

        V_values_deriv, V_dot = _compute_lyapunov_derivative(
            states_grid_physical,
            lyapunov_nn,
            controller_nn,
            observer_nn,
            dynamics_system,
            device,
            estimation_error_grid,
        )
        V_dot_grid = V_dot.reshape(grid_resolution, grid_resolution).cpu().numpy()

    # Get equilibrium info
    if hasattr(dynamics_system, "continuous_time_system"):
        x_equilibrium = dynamics_system.continuous_time_system.x_equilibrium.to(device)
    else:
        x_equilibrium = dynamics_system.x_equilibrium.to(device)

    x_eq_np = x_equilibrium.cpu().numpy()

    # Compute V at equilibrium for plotting
    if observer_nn is not None:
        eq_error = torch.zeros_like(x_equilibrium)
        eq_lyap_input = torch.cat([x_equilibrium, eq_error], dim=0).unsqueeze(0)
    else:
        eq_lyap_input = x_equilibrium.unsqueeze(0)

    V_eq = lyapunov_nn(eq_lyap_input).item()

    # Create figure - dual view if showing derivative
    if show_derivative:
        fig = make_subplots(
            rows=1,
            cols=2,
            specs=[[{"type": "surface"}, {"type": "surface"}]],
            subplot_titles=("Lyapunov Function V(x)", "Lyapunov Derivative ΔV(x)"),
        )

        # Left: V(x) surface
        fig.add_trace(
            go.Surface(
                x=X0,
                y=X1,
                z=V_grid,
                colorscale=colorscale,
                name="V(x)",
                colorbar=dict(title="V(x)", x=0.42, len=0.85, thickness=20),
                hovertemplate="%{xaxis.title.text}: %{x:.3f}<br>%{yaxis.title.text}: %{y:.3f}<br>V: %{z:.3f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # Right: ΔV(x) surface
        fig.add_trace(
            go.Surface(
                x=X0,
                y=X1,
                z=V_dot_grid,
                colorscale="RdBu_r",
                name="ΔV(x)",
                colorbar=dict(title="ΔV(x)", x=1.02, len=0.85, thickness=20),
                hovertemplate="%{xaxis.title.text}: %{x:.3f}<br>%{yaxis.title.text}: %{y:.3f}<br>ΔV: %{z:.3f}<extra></extra>",
            ),
            row=1,
            col=2,
        )

        # Add ROA plane to V(x) plot
        if rho is not None:
            rho_plane = np.full_like(V_grid, rho)
            fig.add_trace(
                go.Surface(
                    x=X0,
                    y=X1,
                    z=rho_plane,
                    opacity=0.4,
                    colorscale=[[0, "red"], [1, "red"]],
                    showscale=False,
                    name=f"ROA (ρ={rho:.3f})",
                    hovertemplate=f"ROA boundary<extra></extra>",
                ),
                row=1,
                col=1,
            )

        # Add zero plane to ΔV(x) plot
        zero_plane = np.zeros_like(V_dot_grid)
        fig.add_trace(
            go.Surface(
                x=X0,
                y=X1,
                z=zero_plane,
                opacity=0.4,
                colorscale=[[0, "black"], [1, "black"]],
                showscale=False,
                name="ΔV=0",
                hovertemplate="ΔV=0 plane<extra></extra>",
            ),
            row=1,
            col=2,
        )

        # Add equilibrium to both plots
        for col in [1, 2]:
            z_val = V_eq if col == 1 else 0.0
            fig.add_trace(
                go.Scatter3d(
                    x=[x_eq_np[idx0]],
                    y=[x_eq_np[idx1]],
                    z=[z_val],
                    mode="markers",
                    marker=dict(
                        size=8,
                        color="lime",
                        symbol="diamond",
                        line=dict(width=2, color="black"),
                    ),
                    showlegend=(col == 1),
                    name="Equilibrium",
                    hovertemplate="Equilibrium<extra></extra>",
                ),
                row=1,
                col=col,
            )

        # Add trajectories to both plots
        if trajectories is not None:
            import plotly.express as px
            
            # Get the color sequence
            try:
                color_sequence = getattr(px.colors.qualitative, trajectory_colorscale)
            except AttributeError:
                print(f"Warning: Color sequence '{trajectory_colorscale}' not found. Using 'Plotly' instead.")
                color_sequence = px.colors.qualitative.Plotly
            
            for i, traj in enumerate(trajectories):
                traj_np = traj.detach().cpu().numpy()
                color = color_sequence[i % len(color_sequence)]
                
                # Compute V and ΔV along trajectory
                with torch.no_grad():
                    # Prepare trajectory states for Lyapunov evaluation
                    traj_physical = traj[:, :x_equilibrium.shape[0]]  # Extract physical states
                    
                    if observer_nn is not None:
                        # For output feedback, augment with zero estimation error
                        traj_error = torch.zeros_like(traj_physical)
                        traj_lyap = torch.cat([traj_physical, traj_error], dim=1)
                    else:
                        traj_lyap = traj_physical
                    
                    # Compute V along trajectory
                    V_traj = lyapunov_nn(traj_lyap).squeeze().cpu().numpy()
                    
                    # Compute ΔV along trajectory
                    V_traj_deriv, V_dot_traj = _compute_lyapunov_derivative(
                        traj_physical,
                        lyapunov_nn,
                        controller_nn,
                        observer_nn,
                        dynamics_system,
                        device,
                        None if observer_nn is None else torch.zeros((traj_physical.shape[0], x_equilibrium.shape[0]), device=device),
                    )
                    V_dot_traj_np = V_dot_traj.cpu().numpy()
                
                # Add trajectory to V(x) plot
                fig.add_trace(
                    go.Scatter3d(
                        x=traj_np[:, idx0],
                        y=traj_np[:, idx1],
                        z=V_traj,
                        mode="lines",
                        line=dict(color=color, width=4),
                        name=f"Trajectory {i+1}",
                        showlegend=True,
                        hovertemplate=f"Traj {i+1}<br>V: %{{z:.3f}}<extra></extra>",
                    ),
                    row=1,
                    col=1,
                )
                
                # Start point on V(x)
                fig.add_trace(
                    go.Scatter3d(
                        x=[traj_np[0, idx0]],
                        y=[traj_np[0, idx1]],
                        z=[V_traj[0]],
                        mode="markers",
                        marker=dict(size=6, color=color, symbol="circle"),
                        showlegend=False,
                        hovertemplate="Start<extra></extra>",
                    ),
                    row=1,
                    col=1,
                )
                
                # Add trajectory to ΔV(x) plot
                fig.add_trace(
                    go.Scatter3d(
                        x=traj_np[:, idx0],
                        y=traj_np[:, idx1],
                        z=V_dot_traj_np,
                        mode="lines",
                        line=dict(color=color, width=4),
                        showlegend=False,
                        hovertemplate=f"Traj {i+1}<br>ΔV: %{{z:.3f}}<extra></extra>",
                    ),
                    row=1,
                    col=2,
                )
                
                # Start point on ΔV(x)
                fig.add_trace(
                    go.Scatter3d(
                        x=[traj_np[0, idx0]],
                        y=[traj_np[0, idx1]],
                        z=[V_dot_traj_np[0]],
                        mode="markers",
                        marker=dict(size=6, color=color, symbol="circle"),
                        showlegend=False,
                        hovertemplate="Start<extra></extra>",
                    ),
                    row=1,
                    col=2,
                )

        # Update scene settings for both plots
        if state_names is None:
            state_names = (f"x{idx0}", f"x{idx1}")

        for col in [1, 2]:
            fig.update_scenes(
                xaxis_title=state_names[0],
                yaxis_title=state_names[1],
                zaxis_title="V(x)" if col == 1 else "ΔV(x)",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.3)),
                row=1,
                col=col,
            )

        if title is None:
            feedback_type = (
                "Output Feedback" if observer_nn is not None else "State Feedback"
            )
            title = f"Lyapunov Function and Derivative 3D ({feedback_type})"

        fig.update_layout(
            title=title,
            height=700,
            width=1500,
            showlegend=True,
            margin=dict(l=50, r=50, t=80, b=50),
        )
        
    else:
        # Single plot - just V(x)
        fig = go.Figure()

        # Lyapunov surface
        fig.add_trace(
            go.Surface(
                x=X0,
                y=X1,
                z=V_grid,
                colorscale=colorscale,
                name="V(x)",
                colorbar=dict(title="V(x)", thickness=20),
                hovertemplate="%{xaxis.title.text}: %{x:.3f}<br>%{yaxis.title.text}: %{y:.3f}<br>V: %{z:.3f}<extra></extra>",
            )
        )

        # Add ROA plane at V = rho
        if rho is not None:
            rho_plane = np.full_like(V_grid, rho)
            fig.add_trace(
                go.Surface(
                    x=X0,
                    y=X1,
                    z=rho_plane,
                    opacity=0.5,
                    colorscale=[[0, "red"], [1, "red"]],
                    showscale=False,
                    name=f"ROA (ρ={rho:.3f})",
                    hovertemplate=f"ROA threshold: {rho:.3f}<extra></extra>",
                )
            )

        # Add equilibrium point
        fig.add_trace(
            go.Scatter3d(
                x=[x_eq_np[idx0]],
                y=[x_eq_np[idx1]],
                z=[V_eq],
                mode="markers",
                marker=dict(
                    size=8,
                    color="lime",
                    symbol="diamond",
                    line=dict(width=2, color="black"),
                ),
                name="Equilibrium",
                hovertemplate="Equilibrium<extra></extra>",
            )
        )

        # Add trajectories to single plot
        if trajectories is not None:
            import plotly.express as px
            
            # Get the color sequence
            try:
                color_sequence = getattr(px.colors.qualitative, trajectory_colorscale)
            except AttributeError:
                print(f"Warning: Color sequence '{trajectory_colorscale}' not found. Using 'Plotly' instead.")
                color_sequence = px.colors.qualitative.Plotly
            
            for i, traj in enumerate(trajectories):
                traj_np = traj.detach().cpu().numpy()
                color = color_sequence[i % len(color_sequence)]
                
                # Compute V along trajectory
                with torch.no_grad():
                    traj_physical = traj[:, :x_equilibrium.shape[0]]
                    
                    if observer_nn is not None:
                        traj_error = torch.zeros_like(traj_physical)
                        traj_lyap = torch.cat([traj_physical, traj_error], dim=1)
                    else:
                        traj_lyap = traj_physical
                    
                    V_traj = lyapunov_nn(traj_lyap).squeeze().cpu().numpy()
                
                # Trajectory line
                fig.add_trace(
                    go.Scatter3d(
                        x=traj_np[:, idx0],
                        y=traj_np[:, idx1],
                        z=V_traj,
                        mode="lines",
                        line=dict(color=color, width=4),
                        name=f"Trajectory {i+1}",
                        hovertemplate=f"Traj {i+1}<br>V: %{{z:.3f}}<extra></extra>",
                    )
                )
                
                # Start point
                fig.add_trace(
                    go.Scatter3d(
                        x=[traj_np[0, idx0]],
                        y=[traj_np[0, idx1]],
                        z=[V_traj[0]],
                        mode="markers",
                        marker=dict(size=6, color=color, symbol="circle"),
                        showlegend=False,
                        hovertemplate="Start<extra></extra>",
                    )
                )

        # Set axis labels
        if state_names is None:
            state_names = (f"x{idx0}", f"x{idx1}")

        if title is None:
            feedback_type = (
                "Output Feedback" if observer_nn is not None else "State Feedback"
            )
            title = f"Lyapunov Function V(x) ({feedback_type})"

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=state_names[0],
                yaxis_title=state_names[1],
                zaxis_title="V(x)",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.3)),
            ),
            height=700,
            width=900,
            showlegend=True,
            margin=dict(l=50, r=50, t=80, b=50),
        )

    fig.update_layout(
        legend=dict(
            x=1.14,
            y=0.5,
            xanchor="left",
            yanchor="middle",
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=11),
        )
    )

    if save_html:
        fig.write_html(save_html)
        print(f"3D Lyapunov surface saved to {save_html}")

    if show:
        fig.show()

    return fig
