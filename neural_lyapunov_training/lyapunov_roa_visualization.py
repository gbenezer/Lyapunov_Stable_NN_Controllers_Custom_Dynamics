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
        if hasattr(controller_nn, "control_dim"):
            u_dim = controller_nn.control_dim
        elif hasattr(controller_nn, "out_dim"):
            u_dim = controller_nn.out_dim
        elif hasattr(controller_nn, "output_dim"):
            u_dim = controller_nn.output_dim
        elif hasattr(controller_nn, "u_up"):
            u_dim = controller_nn.u_up.shape[0]
        elif hasattr(controller_nn, "u_lo"):
            u_dim = controller_nn.u_lo.shape[0]
        else:
            # Default to 1 for scalar control
            print(
                "Derivative computation function was unable to infer controller output dimension."
            )
            print("Current assumption is scalar control (u_dim = 1)")
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
            print(
                f"Warning: Color sequence '{trajectory_colorscale}' not found. Using 'Plotly' instead."
            )
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


def plot_lyapunov_2d_error_slices(
    lyapunov_nn,
    controller_nn,
    dynamics_system,
    observer_nn,
    state_limits: Tuple[Tuple[float, float], Tuple[float, float]],
    error_values: List[float],
    error_dim: int = 0,
    state_indices: Tuple[int, int] = (0, 1),
    state_names: Optional[Tuple[str, str]] = None,
    rho: Optional[float] = None,
    grid_resolution: int = 100,
    title: Optional[str] = None,
    save_html: Optional[str] = None,
    show: bool = True,
):
    """
    Plot Lyapunov function AND derivative showing impact of estimation error

    For each error value, shows:
    - V([x, e]) contours
    - ROA boundary (V = ρ)
    - Decreasing region (ΔV ≤ 0)
    - Verified ROA (both conditions)

    This USES controller and observer to compute closed-loop ΔV.
    """

    state_limits = tuple((to_float(lim[0]), to_float(lim[1])) for lim in state_limits)
    device = next(lyapunov_nn.parameters()).device
    idx0, idx1 = state_indices

    # Get physical state dimension
    if hasattr(dynamics_system, "continuous_time_system"):
        nx = dynamics_system.continuous_time_system.nx
    else:
        nx = dynamics_system.nx

    # Create physical state grid
    x0_range = np.linspace(state_limits[0][0], state_limits[0][1], grid_resolution)
    x1_range = np.linspace(state_limits[1][0], state_limits[1][1], grid_resolution)
    X0, X1 = np.meshgrid(x0_range, x1_range)

    n_points = grid_resolution * grid_resolution
    states_physical = torch.zeros((n_points, nx), device=device)
    states_physical[:, idx0] = torch.tensor(
        X0.flatten(), dtype=torch.float32, device=device
    )
    states_physical[:, idx1] = torch.tensor(
        X1.flatten(), dtype=torch.float32, device=device
    )

    # Set equilibrium for other dimensions
    if hasattr(dynamics_system, "continuous_time_system"):
        x_eq = dynamics_system.continuous_time_system.x_equilibrium.to(device)
    else:
        x_eq = dynamics_system.x_equilibrium.to(device)

    for i in range(nx):
        if i not in state_indices:
            states_physical[:, i] = x_eq[i]

    # Create subplots: 2 rows (V and ΔV) × n_slices columns
    n_slices = len(error_values)
    fig = make_subplots(
        rows=2,
        cols=n_slices,
        subplot_titles=[f"e_{error_dim}={e:.3f}" for e in error_values]
        + [f"ΔV at e_{error_dim}={e:.3f}" for e in error_values],
        specs=[[{"type": "contour"}] * n_slices] * 2,
        vertical_spacing=0.15,
        horizontal_spacing=0.08,
    )

    # Compute V and ΔV for each error value
    for i, error_val in enumerate(error_values):
        col = i + 1

        # Create estimation error vector
        estimation_error = torch.zeros((n_points, nx), device=device)
        estimation_error[:, error_dim] = error_val

        # Augmented state [x, e]
        states_augmented = torch.cat([states_physical, estimation_error], dim=1)

        # Compute V and ΔV using the controller and observer
        with torch.no_grad():
            # Current V([x, e])
            V_current = lyapunov_nn(states_augmented).squeeze()

            # Extract components
            x_true = states_augmented[:, :nx]
            e_current = states_augmented[:, nx:]
            x_hat = x_true - e_current

            # Get measurement
            if hasattr(dynamics_system, "continuous_time_system"):
                y = dynamics_system.continuous_time_system.h(x_true)
            else:
                y = dynamics_system.h(x_true)

            # Controller input (auto-detect augmentation)
            controller_in_dim = None
            if hasattr(controller_nn, "net"):
                controller_in_dim = controller_nn.net[0].in_features
            elif hasattr(controller_nn, "x_equilibrium"):
                controller_in_dim = controller_nn.x_equilibrium.shape[0]

            if controller_in_dim is not None and x_hat.shape[1] < controller_in_dim:
                deficit = controller_in_dim - x_hat.shape[1]
                if deficit == y.shape[1]:
                    controller_input = torch.cat([x_hat, y], dim=1)
                else:
                    padding = torch.zeros((x_hat.shape[0], deficit), device=device)
                    controller_input = torch.cat([x_hat, padding], dim=1)
            else:
                controller_input = x_hat

            # Compute control
            u = controller_nn(controller_input)

            # Evolve physical state
            x_next = dynamics_system(x_true, u)

            # Next measurement
            if hasattr(dynamics_system, "continuous_time_system"):
                y_next = dynamics_system.continuous_time_system.h(x_next)
            else:
                y_next = dynamics_system.h(x_next)

            # Evolve observer
            x_hat_next = observer_nn(x_hat, u, y_next)

            # Next error
            e_next = x_next - x_hat_next

            # Next augmented state
            states_next = torch.cat([x_next, e_next], dim=1)
            V_next = lyapunov_nn(states_next).squeeze()

            # Lyapunov difference
            delta_V = V_next - V_current

        # Reshape for plotting
        V_grid = V_current.reshape(grid_resolution, grid_resolution).cpu().numpy()
        delta_V_grid = delta_V.reshape(grid_resolution, grid_resolution).cpu().numpy()

        # Row 1: V([x, e]) contours
        fig.add_trace(
            go.Contour(
                x=x0_range,
                y=x1_range,
                z=V_grid,
                colorscale="Viridis",
                showscale=(col == n_slices),
                colorbar=dict(title="V([x,e])", x=1.02) if col == n_slices else None,
                contours=dict(start=0, end=V_grid.max(), size=V_grid.max() / 20),
            ),
            row=1,
            col=col,
        )

        # Add ROA boundary
        if rho is not None:
            fig.add_trace(
                go.Contour(
                    x=x0_range,
                    y=x1_range,
                    z=V_grid,
                    contours=dict(start=rho, end=rho, size=1, coloring="none"),
                    line=dict(color="red", width=3),
                    showscale=False,
                    showlegend=(col == 1),
                    name=f"V=ρ",
                ),
                row=1,
                col=col,
            )

        # Row 2: ΔV([x, e]) contours
        fig.add_trace(
            go.Contour(
                x=x0_range,
                y=x1_range,
                z=delta_V_grid,
                colorscale="RdBu_r",
                showscale=(col == n_slices),
                colorbar=dict(title="ΔV([x,e])", x=1.02) if col == n_slices else None,
                contours=dict(
                    start=delta_V_grid.min(),
                    end=delta_V_grid.max(),
                    size=(delta_V_grid.max() - delta_V_grid.min()) / 20,
                ),
            ),
            row=2,
            col=col,
        )

        # Add ΔV=0 contour (stability boundary)
        fig.add_trace(
            go.Contour(
                x=x0_range,
                y=x1_range,
                z=delta_V_grid,
                contours=dict(start=0, end=0, size=1, coloring="none"),
                line=dict(color="black", width=3, dash="dash"),
                showscale=False,
                showlegend=(col == 1),
                name="ΔV=0",
            ),
            row=2,
            col=col,
        )

        # Add equilibrium to both rows
        for row in [1, 2]:
            fig.add_trace(
                go.Scatter(
                    x=[x_eq[idx0].cpu().item()],
                    y=[x_eq[idx1].cpu().item()],
                    mode="markers",
                    marker=dict(size=8, color="lime", symbol="star"),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

    # Update axes
    if state_names is None:
        state_names = (f"x{idx0}", f"x{idx1}")

    for col in range(1, n_slices + 1):
        fig.update_xaxes(title_text=state_names[0], row=1, col=col)
        fig.update_yaxes(title_text=state_names[1], row=1, col=col)
        fig.update_xaxes(title_text=state_names[0], row=2, col=col)
        fig.update_yaxes(title_text=state_names[1], row=2, col=col)

    if title is None:
        title = f"Lyapunov Analysis: Impact of Estimation Error e_{error_dim}"

    fig.update_layout(
        title=title,
        height=800,
        width=400 * n_slices,
        showlegend=True,
    )

    if save_html:
        fig.write_html(save_html)
        print(f"Error slice visualization saved to {save_html}")

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
    surface_opacity: Optional[float] = None,
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
        surface_opacity: Opacity of the plotted 3D surfaces (default 0.5 with trajectories and 1.0 without)

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

    # Automatically adjust opacity if trajectories are present
    if surface_opacity is None:
        surface_opacity = 0.5 if trajectories is not None else 1.0

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
                opacity=surface_opacity,
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
                opacity=surface_opacity,
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
                print(
                    f"Warning: Color sequence '{trajectory_colorscale}' not found. Using 'Plotly' instead."
                )
                color_sequence = px.colors.qualitative.Plotly

            for i, traj in enumerate(trajectories):
                traj_np = traj.detach().cpu().numpy()
                color = color_sequence[i % len(color_sequence)]

                # Compute V and ΔV along trajectory
                with torch.no_grad():
                    # Prepare trajectory states for Lyapunov evaluation
                    traj_physical = traj[
                        :, : x_equilibrium.shape[0]
                    ]  # Extract physical states

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
                        (
                            None
                            if observer_nn is None
                            else torch.zeros(
                                (traj_physical.shape[0], x_equilibrium.shape[0]),
                                device=device,
                            )
                        ),
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
                opacity=surface_opacity,
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
                    opacity=0.4,
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
                print(
                    f"Warning: Color sequence '{trajectory_colorscale}' not found. Using 'Plotly' instead."
                )
                color_sequence = px.colors.qualitative.Plotly

            for i, traj in enumerate(trajectories):
                traj_np = traj.detach().cpu().numpy()
                color = color_sequence[i % len(color_sequence)]

                # Compute V along trajectory
                with torch.no_grad():
                    traj_physical = traj[:, : x_equilibrium.shape[0]]

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


def plot_roa_vs_error(
    lyapunov_nn,
    controller_nn,
    dynamics_system,
    observer_nn,
    state_limits: Tuple[Tuple[float, float], Tuple[float, float]],
    error_range: Tuple[float, float],
    rho: float,
    state_dim: int = 0,  # Which physical dimension to plot (other held at equilibrium)
    error_dim: int = 0,  # Which error dimension to vary
    grid_resolution: int = 80,
    title: Optional[str] = None,
    save_html: Optional[str] = None,
    show: bool = True,
):
    """
    Plot verified ROA boundary as a 3D surface in (physical_state, error) space

    Shows the set of (x, e) where BOTH:
    - V([x, e]) ≤ ρ
    - ΔV([x, e]) ≤ 0 (computed with actual closed-loop dynamics)

    This uses controller and observer to compute ΔV at each point.

    Args:
        lyapunov_nn: Lyapunov function V([x, e])
        controller_nn: Controller u = π([x_hat, y])
        dynamics_system: Discrete dynamics
        observer_nn: Observer
        state_limits: Bounds for one physical state dimension (other at equilibrium)
        error_range: (min, max) for estimation error
        rho: ROA threshold
        state_dim: Which physical state dimension to plot (0 for θ, 1 for θ̇)
        error_dim: Which error dimension to vary
        grid_resolution: Grid density
        title: Plot title
        save_html: Save path
        show: Display plot

    Returns:
        Plotly 3D figure
    """

    state_limits = tuple((to_float(lim[0]), to_float(lim[1])) for lim in state_limits)
    error_range = (to_float(error_range[0]), to_float(error_range[1]))
    device = next(lyapunov_nn.parameters()).device

    # Get physical state dimension
    if hasattr(dynamics_system, "continuous_time_system"):
        nx = dynamics_system.continuous_time_system.nx
        x_eq = dynamics_system.continuous_time_system.x_equilibrium.to(device)
    else:
        nx = dynamics_system.nx
        x_eq = dynamics_system.x_equilibrium.to(device)

    # Create 2D grid: (physical_state[state_dim], error[error_dim])
    x_range = np.linspace(state_limits[0][0], state_limits[0][1], grid_resolution)
    e_range = np.linspace(error_range[0], error_range[1], grid_resolution)
    X_grid, E_grid = np.meshgrid(x_range, e_range)

    n_points = grid_resolution * grid_resolution

    # Build physical states (other dims at equilibrium)
    states_physical = x_eq.unsqueeze(0).expand(n_points, -1).clone()
    states_physical[:, state_dim] = torch.tensor(
        X_grid.flatten(), dtype=torch.float32, device=device
    )

    # Build estimation errors (other dims at zero)
    estimation_errors = torch.zeros((n_points, nx), device=device)
    estimation_errors[:, error_dim] = torch.tensor(
        E_grid.flatten(), dtype=torch.float32, device=device
    )

    # Augmented states [x, e]
    states_augmented = torch.cat([states_physical, estimation_errors], dim=1)

    # Compute V and ΔV using actual closed-loop dynamics
    with torch.no_grad():
        # Current V([x, e])
        V_current = lyapunov_nn(states_augmented).squeeze()

        # Compute ΔV using controller and observer
        x_true = states_augmented[:, :nx]
        e_current = states_augmented[:, nx:]
        x_hat = x_true - e_current

        # Get measurement
        if hasattr(dynamics_system, "continuous_time_system"):
            y = dynamics_system.continuous_time_system.h(x_true)
        else:
            y = dynamics_system.h(x_true)

        # Controller input (auto-detect augmentation)
        controller_in_dim = None
        if hasattr(controller_nn, "net"):
            controller_in_dim = controller_nn.net[0].in_features
        elif hasattr(controller_nn, "x_equilibrium"):
            controller_in_dim = controller_nn.x_equilibrium.shape[0]

        if controller_in_dim is not None and x_hat.shape[1] < controller_in_dim:
            deficit = controller_in_dim - x_hat.shape[1]
            if deficit == y.shape[1]:
                controller_input = torch.cat([x_hat, y], dim=1)
            else:
                padding = torch.zeros((x_hat.shape[0], deficit), device=device)
                controller_input = torch.cat([x_hat, padding], dim=1)
        else:
            controller_input = x_hat

        # Compute control
        u = controller_nn(controller_input)

        # Evolve physical state
        x_next = dynamics_system(x_true, u)

        # Next measurement
        if hasattr(dynamics_system, "continuous_time_system"):
            y_next = dynamics_system.continuous_time_system.h(x_next)
        else:
            y_next = dynamics_system.h(x_next)

        # Evolve observer
        x_hat_next = observer_nn(x_hat, u, y_next)

        # Next error
        e_next = x_next - x_hat_next

        # Next augmented state
        states_next = torch.cat([x_next, e_next], dim=1)
        V_next = lyapunov_nn(states_next).squeeze()

        # Lyapunov difference
        delta_V = V_next - V_current

    # Reshape for plotting
    V_grid = V_current.reshape(grid_resolution, grid_resolution).cpu().numpy()
    delta_V_grid = delta_V.reshape(grid_resolution, grid_resolution).cpu().numpy()

    # Create masks for different regions
    in_roa = (V_current <= rho).reshape(grid_resolution, grid_resolution).cpu().numpy()
    is_decreasing = (
        (delta_V <= 0).reshape(grid_resolution, grid_resolution).cpu().numpy()
    )
    verified_roa = in_roa & is_decreasing

    # Create figure with multiple surfaces
    fig = go.Figure()

    # 1. V([x, e]) surface
    fig.add_trace(
        go.Surface(
            x=X_grid,
            y=E_grid,
            z=V_grid,
            colorscale="Viridis",
            name="V([x,e])",
            opacity=0.7,
            colorbar=dict(title="V([x,e])", x=0.45, len=0.8),
            hovertemplate=f"x_{state_dim}: %{{x:.3f}}<br>"
            f"e_{error_dim}: %{{y:.3f}}<br>"
            f"V: %{{z:.3f}}<extra></extra>",
        )
    )

    # 2. ROA threshold plane
    rho_plane = np.full_like(V_grid, rho)
    fig.add_trace(
        go.Surface(
            x=X_grid,
            y=E_grid,
            z=rho_plane,
            opacity=0.5,
            colorscale=[[0, "red"], [1, "red"]],
            showscale=False,
            name=f"ROA threshold (ρ={rho:.3f})",
            hovertemplate=f"ρ = {rho:.3f}<extra></extra>",
        )
    )

    # 3. Highlight e=0 slice (ideal behavior)
    e_zero_idx = np.argmin(np.abs(e_range))
    fig.add_trace(
        go.Scatter3d(
            x=x_range,
            y=np.zeros_like(x_range),
            z=V_grid[e_zero_idx, :],
            mode="lines",
            line=dict(color="cyan", width=6),
            name="e=0 (ideal)",
            hovertemplate="Ideal behavior (e=0)<extra></extra>",
        )
    )

    # 4. Show where ΔV > 0 (violations) as scatter points
    violation_mask = (V_current <= rho) & (delta_V > 0)
    if violation_mask.any():
        violation_indices = torch.where(violation_mask)[0]
        # Sample violations to avoid too many points
        if len(violation_indices) > 1000:
            violation_indices = violation_indices[:: len(violation_indices) // 1000]

        x_viol = states_physical[violation_indices, state_dim].cpu().numpy()
        e_viol = estimation_errors[violation_indices, error_dim].cpu().numpy()
        V_viol = V_current[violation_indices].cpu().numpy()

        fig.add_trace(
            go.Scatter3d(
                x=x_viol,
                y=e_viol,
                z=V_viol,
                mode="markers",
                marker=dict(size=3, color="orange", symbol="x"),
                name="ΔV>0 violations",
                hovertemplate="Violation: V≤ρ but ΔV>0<extra></extra>",
            )
        )

    if title is None:
        title = f"ROA vs Estimation Error: x_{state_dim} and e_{error_dim}"

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=f"x_{state_dim} (physical state)",
            yaxis_title=f"e_{error_dim} (estimation error)",
            zaxis_title="V([x,e])",
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.3)),
        ),
        height=700,
        width=1000,
        showlegend=True,
    )

    if save_html:
        fig.write_html(save_html)
        print(f"ROA vs error 3D plot saved to {save_html}")

    if show:
        fig.show()

    return fig


def create_roa_error_animation(
    lyapunov_nn,
    controller_nn,
    dynamics_system,
    observer_nn,
    state_limits: Tuple[Tuple[float, float], Tuple[float, float]],
    error_range: Tuple[float, float],
    error_dim: int = 0,
    n_frames: int = 20,
    state_indices: Tuple[int, int] = (0, 1),
    state_names: Optional[Tuple[str, str]] = None,
    rho: Optional[float] = None,
    grid_resolution: int = 100,
    save_html: Optional[str] = None,
    show: bool = True,
):
    """
    Create animated visualization showing how verified ROA changes with estimation error

    Each frame shows the physical state space with:
    - V([x, e]) contours
    - ROA boundary (V = ρ)
    - Stability boundary (ΔV = 0)
    - Verified ROA (shaded region where both conditions hold)

    Uses controller and observer to compute actual closed-loop ΔV.

    Args:
        lyapunov_nn: Lyapunov function V([x, e])
        controller_nn: Controller
        dynamics_system: Dynamics
        observer_nn: Observer
        state_limits: Bounds for physical state
        error_range: (min, max) error values for animation
        error_dim: Which error dimension to animate
        n_frames: Number of animation frames
        state_indices: Which physical states to plot
        state_names: Axis labels
        rho: ROA threshold
        grid_resolution: Grid density
        save_html: Save path
        show: Display plot

    Returns:
        Plotly figure with animation
    """

    state_limits = tuple((to_float(lim[0]), to_float(lim[1])) for lim in state_limits)
    error_range = (to_float(error_range[0]), to_float(error_range[1]))
    device = next(lyapunov_nn.parameters()).device
    idx0, idx1 = state_indices

    # Get physical state dimension
    if hasattr(dynamics_system, "continuous_time_system"):
        nx = dynamics_system.continuous_time_system.nx
        x_eq = dynamics_system.continuous_time_system.x_equilibrium.to(device)
    else:
        nx = dynamics_system.nx
        x_eq = dynamics_system.x_equilibrium.to(device)

    # Create physical state grid
    x0_range = np.linspace(state_limits[0][0], state_limits[0][1], grid_resolution)
    x1_range = np.linspace(state_limits[1][0], state_limits[1][1], grid_resolution)
    X0, X1 = np.meshgrid(x0_range, x1_range)

    n_points = grid_resolution * grid_resolution
    states_physical = torch.zeros((n_points, nx), device=device)
    states_physical[:, idx0] = torch.tensor(
        X0.flatten(), dtype=torch.float32, device=device
    )
    states_physical[:, idx1] = torch.tensor(
        X1.flatten(), dtype=torch.float32, device=device
    )

    # Set other physical dimensions to equilibrium
    for i in range(nx):
        if i not in state_indices:
            states_physical[:, i] = x_eq[i]

    # Generate error values for animation
    error_values = np.linspace(error_range[0], error_range[1], n_frames)

    # Precompute V, ΔV, and verified ROA for all error values
    frames_V = []
    frames_delta_V = []
    frames_verified = []

    print(f"Precomputing {n_frames} frames...")
    for frame_idx, error_val in enumerate(error_values):
        if frame_idx % 5 == 0:
            print(f"  Frame {frame_idx+1}/{n_frames} (e={error_val:.3f})")

        # Create estimation error
        estimation_error = torch.zeros((n_points, nx), device=device)
        estimation_error[:, error_dim] = error_val

        # Augmented state
        states_augmented = torch.cat([states_physical, estimation_error], dim=1)

        # Compute V and ΔV with actual dynamics
        with torch.no_grad():
            V_current = lyapunov_nn(states_augmented).squeeze()

            # Compute ΔV using closed-loop dynamics
            x_true = states_augmented[:, :nx]
            e_current = states_augmented[:, nx:]
            x_hat = x_true - e_current

            # Get measurement
            if hasattr(dynamics_system, "continuous_time_system"):
                y = dynamics_system.continuous_time_system.h(x_true)
            else:
                y = dynamics_system.h(x_true)

            # Controller input (auto-detect augmentation)
            controller_in_dim = None
            if hasattr(controller_nn, "net"):
                controller_in_dim = controller_nn.net[0].in_features
            elif hasattr(controller_nn, "x_equilibrium"):
                controller_in_dim = controller_nn.x_equilibrium.shape[0]

            if controller_in_dim is not None and x_hat.shape[1] < controller_in_dim:
                deficit = controller_in_dim - x_hat.shape[1]
                if deficit == y.shape[1]:
                    controller_input = torch.cat([x_hat, y], dim=1)
                else:
                    padding = torch.zeros((x_hat.shape[0], deficit), device=device)
                    controller_input = torch.cat([x_hat, padding], dim=1)
            else:
                controller_input = x_hat

            # Compute control
            u = controller_nn(controller_input)

            # Evolve physical state
            x_next = dynamics_system(x_true, u)

            # Next measurement
            if hasattr(dynamics_system, "continuous_time_system"):
                y_next = dynamics_system.continuous_time_system.h(x_next)
            else:
                y_next = dynamics_system.h(x_next)

            # Evolve observer
            x_hat_next = observer_nn(x_hat, u, y_next)

            # Next error
            e_next = x_next - x_hat_next

            # Next augmented state
            states_next = torch.cat([x_next, e_next], dim=1)
            V_next = lyapunov_nn(states_next).squeeze()

            # Lyapunov difference
            delta_V = V_next - V_current

        # Reshape
        V_grid = V_current.reshape(grid_resolution, grid_resolution).cpu().numpy()
        delta_V_grid = delta_V.reshape(grid_resolution, grid_resolution).cpu().numpy()

        # Compute verified ROA (both conditions)
        in_roa = (V_current <= rho).cpu().numpy()
        is_decreasing = (delta_V <= 0).cpu().numpy()
        verified = (in_roa & is_decreasing).reshape(grid_resolution, grid_resolution)

        frames_V.append(V_grid)
        frames_delta_V.append(delta_V_grid)
        frames_verified.append(verified)

    print("Creating animation...")

    # Create initial frame
    if state_names is None:
        state_names = (f"x{idx0}", f"x{idx1}")

    # Initial traces
    initial_traces = [
        # V contours
        go.Contour(
            x=x0_range,
            y=x1_range,
            z=frames_V[0],
            colorscale="Viridis",
            colorbar=dict(title="V([x,e])", len=0.8),
            name="V([x,e])",
            contours=dict(
                start=0,
                end=max(f.max() for f in frames_V),
                size=max(f.max() for f in frames_V) / 20,
            ),
        ),
        # ROA boundary (V = ρ)
        go.Contour(
            x=x0_range,
            y=x1_range,
            z=frames_V[0],
            contours=dict(start=rho, end=rho, size=1, coloring="none"),
            line=dict(color="red", width=4),
            showscale=False,
            name=f"V=ρ",
        ),
        # Stability boundary (ΔV = 0)
        go.Contour(
            x=x0_range,
            y=x1_range,
            z=frames_delta_V[0],
            contours=dict(start=0, end=0, size=1, coloring="none"),
            line=dict(color="orange", width=3, dash="dash"),
            showscale=False,
            name="ΔV=0",
        ),
        # Verified ROA (shaded)
        go.Contour(
            x=x0_range,
            y=x1_range,
            z=frames_verified[0].astype(float),
            contours=dict(start=0.5, end=0.5, size=1, coloring="heatmap"),
            colorscale=[[0, "rgba(0,255,0,0)"], [1, "rgba(0,255,0,0.3)"]],
            showscale=False,
            name="Verified ROA",
        ),
        # Equilibrium
        go.Scatter(
            x=[x_eq[idx0].cpu().item()],
            y=[x_eq[idx1].cpu().item()],
            mode="markers",
            marker=dict(
                size=12, color="lime", symbol="star", line=dict(width=2, color="black")
            ),
            name="Equilibrium",
        ),
    ]

    # Create frames for animation
    animation_frames = []
    for frame_idx, (e_val, V_frame, dV_frame, verified_frame) in enumerate(
        zip(error_values, frames_V, frames_delta_V, frames_verified)
    ):
        animation_frames.append(
            go.Frame(
                data=[
                    go.Contour(x=x0_range, y=x1_range, z=V_frame),
                    go.Contour(
                        x=x0_range,
                        y=x1_range,
                        z=V_frame,
                        contours=dict(start=rho, end=rho, size=1, coloring="none"),
                        line=dict(color="red", width=4),
                    ),
                    go.Contour(
                        x=x0_range,
                        y=x1_range,
                        z=dV_frame,
                        contours=dict(start=0, end=0, size=1, coloring="none"),
                        line=dict(color="orange", width=3, dash="dash"),
                    ),
                    go.Contour(
                        x=x0_range,
                        y=x1_range,
                        z=verified_frame.astype(float),
                        contours=dict(start=0.5, end=0.5, size=1, coloring="heatmap"),
                        colorscale=[[0, "rgba(0,255,0,0)"], [1, "rgba(0,255,0,0.3)"]],
                    ),
                    go.Scatter(
                        x=[x_eq[idx0].cpu().item()],
                        y=[x_eq[idx1].cpu().item()],
                        mode="markers",
                    ),
                ],
                name=f"{e_val:.3f}",
                layout=go.Layout(
                    annotations=[
                        dict(
                            text=f"e_{error_dim} = {e_val:.3f}",
                            x=0.5,
                            y=1.05,
                            xref="paper",
                            yref="paper",
                            showarrow=False,
                            font=dict(size=16, color="black"),
                        )
                    ]
                ),
            )
        )

    fig = go.Figure(data=initial_traces, frames=animation_frames)

    # Add animation controls
    fig.update_layout(
        title=f"Verified ROA Evolution with Estimation Error e_{error_dim}<br>"
        f"<sub>Green: Verified ROA (V≤ρ AND ΔV≤0) | Red: V=ρ | Orange: ΔV=0</sub>",
        xaxis_title=state_names[0],
        yaxis_title=state_names[1],
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(
                        label="▶ Play",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {"duration": 300, "redraw": True},
                                "fromcurrent": True,
                                "mode": "immediate",
                            },
                        ],
                    ),
                    dict(
                        label="⏸ Pause",
                        method="animate",
                        args=[
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                            },
                        ],
                    ),
                ],
                x=0.1,
                y=1.15,
                xanchor="left",
                yanchor="top",
            )
        ],
        sliders=[
            dict(
                active=0,
                steps=[
                    dict(
                        method="animate",
                        args=[
                            [f"{e_val:.3f}"],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                            },
                        ],
                        label=f"{e_val:.3f}",
                    )
                    for e_val in error_values
                ],
                x=0.1,
                xanchor="left",
                y=0,
                yanchor="top",
                currentvalue=dict(
                    prefix=f"e_{error_dim} = ",
                    visible=True,
                    xanchor="right",
                ),
                len=0.9,
                pad=dict(t=50),
            )
        ],
        height=700,
        width=900,
    )

    if save_html:
        fig.write_html(save_html)
        print(f"Animated ROA vs error saved to {save_html}")

    if show:
        fig.show()

    return fig
