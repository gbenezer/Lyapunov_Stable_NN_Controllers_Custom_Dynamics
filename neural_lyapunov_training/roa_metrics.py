"""
Region of Attraction (ROA) Metrics

Quantitative measurement of ROA area/volume using:
- Monte Carlo (random sampling)
- Quasi-Monte Carlo (low-discrepancy sequences: Sobol, Halton)
- Grid-based methods
"""

import torch
import numpy as np
from typing import Optional, Tuple, Dict, Callable
from dataclasses import dataclass
from neural_lyapunov_training.dynamical_system import DiscreteTimeSystem


@dataclass
class ROAMetrics:
    """Container for ROA metrics"""

    rho: float  # ROA threshold (V(x) <= rho)
    area_roa: float  # Area/volume of ROA
    area_domain: float  # Area/volume of full domain
    coverage_ratio: float  # ROA area / domain area
    num_samples_in_roa: int  # Number of samples inside ROA
    num_samples_total: int  # Total number of samples
    domain_bounds: Tuple  # Domain boundaries used
    method: str  # Computation method ('monte_carlo', 'qmc_sobol', 'qmc_halton', 'grid')
    grid_resolution: Optional[int] = None  # For grid method
    discrepancy: Optional[float] = None  # For QMC methods


@dataclass
class LyapunovDifferenceMetrics:
    """Container for Lyapunov difference metrics in discrete-time systems"""

    rho: float  # ROA threshold

    # Area/volume measurements
    area_roa: float  # Area where V(x) ≤ ρ
    area_decreasing: float  # Area where ΔV(x) ≤ 0
    area_verified_roa: float  # Area where both V(x) ≤ ρ AND ΔV(x) ≤ 0
    area_domain: float  # Total domain area

    # Coverage ratios
    coverage_roa: float  # Fraction in ROA
    coverage_decreasing: float  # Fraction with ΔV ≤ 0
    coverage_verified_roa: float  # Fraction satisfying both

    # Sample counts
    num_samples_in_roa: int
    num_samples_decreasing: int  # ΔV ≤ 0
    num_samples_verified_roa: int  # Both conditions
    num_samples_total: int

    # Lyapunov difference statistics
    mean_delta_V: float  # Mean ΔV over all samples
    mean_delta_V_in_roa: float  # Mean ΔV in ROA only
    max_delta_V: float  # Maximum ΔV (worst violation)
    min_delta_V: float  # Minimum ΔV (best decrease)
    std_delta_V: float  # Standard deviation of ΔV

    # Additional statistics
    max_violation_in_roa: float  # Worst ΔV violation within ROA
    percent_verified: float  # Percentage of ROA that is verified (ΔV ≤ 0)

    # Metadata
    domain_bounds: Tuple
    method: str
    stability_threshold: float = 0.0  # Threshold for ΔV ≤ threshold
    grid_resolution: Optional[int] = None
    discrepancy: Optional[float] = None


# Convert state_limits to CPU numpy if they're tensors
def to_float(val):
    """Convert tensor or array to float"""
    if isinstance(val, torch.Tensor):
        return val.detach().cpu().item()
    elif isinstance(val, (np.ndarray, np.generic)):
        return float(val)
    else:
        return float(val)


def round_to_power_of_2(n: int, direction: str = "nearest") -> int:
    """
    Round number to nearest power of 2

    Args:
        n: Input number
        direction: 'up', 'down', or 'nearest'

    Returns:
        Closest power of 2
    """
    if n <= 0:
        return 1

    log2_n = np.log2(n)

    if direction == "up":
        return int(2 ** np.ceil(log2_n))
    elif direction == "down":
        return int(2 ** np.floor(log2_n))
    else:  # nearest
        return int(2 ** np.round(log2_n))


def generate_sobol_samples(
    n_samples: int,
    n_dims: int,
    bounds: Tuple[Tuple[float, float], ...],
    device: str = "cpu",
    round_to_pow2: bool = True,
) -> torch.Tensor:
    """
    Generate samples using Sobol low-discrepancy sequence

    Sobol sequences have better uniformity than random sampling,
    leading to faster convergence (O(1/N) vs O(1/sqrt(N)))

    Args:
        n_samples: Number of samples (will be rounded to power of 2 if round_to_pow2=True)
        n_dims: Number of dimensions
        bounds: (min, max) for each dimension
        device: Computing device
        round_to_pow2: Whether to round n_samples to nearest power of 2

    Returns:
        Tensor of samples (actual_n_samples, n_dims)
        Note: actual_n_samples may differ from n_samples if rounded
    """
    from scipy.stats import qmc

    # Round to power of 2 for optimal Sobol properties
    if round_to_pow2:
        n_samples_actual = round_to_power_of_2(n_samples, direction="nearest")
        if n_samples_actual != n_samples:
            import warnings

            warnings.warn(
                f"Rounded n_samples from {n_samples} to {n_samples_actual} (power of 2) "
                f"for optimal Sobol sequence properties.",
                UserWarning,
            )
    else:
        n_samples_actual = n_samples

    # Generate samples in unit hypercube [0,1]^d
    sobol_engine = qmc.Sobol(d=n_dims, scramble=True, seed=42)
    samples_unit = sobol_engine.random(n_samples_actual)

    # Scale to actual bounds
    samples = np.zeros_like(samples_unit)
    for i in range(n_dims):
        low, high = bounds[i]
        samples[:, i] = samples_unit[:, i] * (high - low) + low

    return torch.tensor(samples, dtype=torch.float32, device=device)


def generate_halton_samples(
    n_samples: int,
    n_dims: int,
    bounds: Tuple[Tuple[float, float], ...],
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate samples using Halton low-discrepancy sequence

    Halton sequences are another type of low-discrepancy sequence,
    often better for lower dimensions (d <= 10)

    Args:
        n_samples: Number of samples
        n_dims: Number of dimensions
        bounds: (min, max) for each dimension
        device: Computing device

    Returns:
        Tensor of samples (n_samples, n_dims)
    """
    from scipy.stats import qmc

    # Generate samples in unit hypercube
    halton_engine = qmc.Halton(d=n_dims, scramble=True, seed=42)
    samples_unit = halton_engine.random(n_samples)

    # Scale to actual bounds
    samples = np.zeros_like(samples_unit)
    for i in range(n_dims):
        low, high = bounds[i]
        samples[:, i] = samples_unit[:, i] * (high - low) + low

    return torch.tensor(samples, dtype=torch.float32, device=device)


def compute_discrepancy(samples: np.ndarray, method: str = "star") -> float:
    """
    Compute discrepancy of a point set

    Lower discrepancy = better uniformity
    Random sequences: D ≈ O(sqrt(log N / N))
    QMC sequences: D ≈ O((log N)^d / N)

    Args:
        samples: Samples in [0,1]^d (n_samples, n_dims)
        method: 'star', 'L2', or 'centered'

    Returns:
        Discrepancy measure
    """
    from scipy.stats import qmc

    try:
        if method == "star":
            disc = qmc.discrepancy(samples, method="SD")  # Star discrepancy
        elif method == "L2":
            disc = qmc.discrepancy(samples, method="L2-star")
        elif method == "centered":
            disc = qmc.discrepancy(samples, method="CD")  # Centered discrepancy
        else:
            disc = qmc.discrepancy(samples)  # Default

        # If discrepancy is suspiciously small, try alternate computation
        if disc < 1e-10 and len(samples) < 1e6:
            # Compute a simple uniformity metric as backup
            # Check coverage of hypercube subdivisions
            n_bins = int(len(samples) ** (1 / samples.shape[1]))
            hist, _ = np.histogramdd(samples, bins=[n_bins] * samples.shape[1])
            expected = len(samples) / (n_bins ** samples.shape[1])
            disc_alternate = np.sqrt(np.mean((hist - expected) ** 2)) / expected

            if disc_alternate > disc:
                return disc_alternate

        return disc
    except Exception as e:
        # Fallback: compute simple uniformity metric
        print(f"Warning: scipy discrepancy failed ({e}), using fallback metric")
        n_bins = max(10, int(len(samples) ** (1 / samples.shape[1])))
        hist, _ = np.histogramdd(samples, bins=[n_bins] * samples.shape[1])
        expected = len(samples) / (n_bins ** samples.shape[1])
        return np.sqrt(np.mean((hist - expected) ** 2)) / expected


def compute_roa_area_qmc_sobol(
    lyapunov_nn,
    state_limits: Tuple[Tuple[float, float], ...],
    rho: float,
    num_samples: int = 100000,
    device: str = "cpu",
    state_indices: Optional[Tuple[int, ...]] = None,
    compute_discrepancy_metric: bool = False,
    round_to_pow2: bool = True,
) -> ROAMetrics:
    """
    Compute ROA area using Quasi-Monte Carlo with Sobol sequence

    Sobol sequences provide better coverage than random sampling,
    typically requiring 10-100x fewer samples for same accuracy.

    Args:
        lyapunov_nn: Lyapunov function V(x)
        state_limits: Tuple of (min, max) for EACH dimension in state_indices
                     Example: state_indices=(1,3), state_limits has 2 entries
        rho: ROA threshold
        num_samples: Number of Sobol samples
        device: Computing device
        state_indices: Which state dimensions to consider
        compute_discrepancy_metric: Whether to compute discrepancy (slow for large N)

    Returns:
        ROAMetrics object with area measurements
    """

    # convert to floats if not already floats
    state_limits = tuple((to_float(lim[0]), to_float(lim[1])) for lim in state_limits)

    # Determine dimensions
    if state_indices is None:
        state_indices = tuple(range(len(state_limits)))

    n_dims = len(state_indices)

    # FIXED: Compute domain volume using enumeration index
    domain_volume = 1.0
    for i in range(len(state_limits)):
        domain_volume *= state_limits[i][1] - state_limits[i][0]

    # Infer total state dimension
    if hasattr(lyapunov_nn, "parameters"):
        try:
            first_layer = next(lyapunov_nn.parameters())
            nx_total = (
                first_layer.shape[1]
                if len(first_layer.shape) > 1
                else first_layer.shape[0]
            )
        except StopIteration:
            # No parameters - assume nx equals number of limits
            nx_total = len(state_limits)
    else:
        nx_total = len(state_limits)

    # FIXED: Generate Sobol samples using enumeration
    bounds_for_sobol = tuple(state_limits[i] for i in range(len(state_limits)))
    sobol_samples = generate_sobol_samples(
        num_samples, n_dims, bounds_for_sobol, device, round_to_pow2
    )

    ## CRITICAL FIX: Use actual number of samples generated, not requested
    num_samples_actual = sobol_samples.shape[0]

    # Create full state samples with correct size
    samples = torch.zeros((num_samples_actual, nx_total), device=device)
    for i, idx in enumerate(state_indices):
        samples[:, idx] = sobol_samples[:, i]

    # Evaluate Lyapunov function
    with torch.no_grad():
        V_values = lyapunov_nn(samples).squeeze()

    # Count samples in ROA
    in_roa = (V_values <= rho).sum().item()

    # Estimate ROA volume
    roa_volume = domain_volume * (in_roa / num_samples_actual)
    coverage_ratio = in_roa / num_samples_actual

    # Optionally compute discrepancy
    discrepancy_val = None
    if compute_discrepancy_metric:
        # Normalize samples to [0,1]^d for discrepancy computation
        samples_normalized = np.zeros((num_samples_actual, n_dims))
        for i in range(len(state_limits)):
            low, high = state_limits[i]
            samples_normalized[:, i] = (sobol_samples[:, i].cpu().numpy() - low) / (
                high - low
            )

        # Clip to [0,1] to handle any floating point errors
        samples_normalized = np.clip(samples_normalized, 0.0, 1.0)
        discrepancy_val = compute_discrepancy(
            samples_normalized, method="CD"
        )  # Use centered discrepancy

    return ROAMetrics(
        rho=rho,
        area_roa=roa_volume,
        area_domain=domain_volume,
        coverage_ratio=coverage_ratio,
        num_samples_in_roa=in_roa,
        num_samples_total=num_samples_actual,  # Use actual count
        domain_bounds=state_limits,
        method="qmc_sobol",
        discrepancy=discrepancy_val,
    )


def compute_roa_area_qmc_halton(
    lyapunov_nn,
    state_limits: Tuple[Tuple[float, float], ...],
    rho: float,
    num_samples: int = 100000,
    device: str = "cpu",
    state_indices: Optional[Tuple[int, ...]] = None,
    compute_discrepancy_metric: bool = False,
    round_to_pow2: bool = False,  # Halton doesn't require power of 2, but support for consistency
) -> ROAMetrics:
    """
    Compute ROA area using Quasi-Monte Carlo with Halton sequence

    Halton sequences are particularly good for lower dimensions (d <= 10).

    Args:
        lyapunov_nn: Lyapunov function V(x)
        state_limits: Tuple of (min, max) for EACH dimension in state_indices
        rho: ROA threshold
        num_samples: Number of Halton samples
        device: Computing device
        state_indices: Which state dimensions to consider
        compute_discrepancy_metric: Whether to compute discrepancy
        round_to_pow2: For consistency with Sobol (Halton doesn't require it)

    Returns:
        ROAMetrics object with area measurements
    """

    # convert to floats if not already floats
    state_limits = tuple((to_float(lim[0]), to_float(lim[1])) for lim in state_limits)

    # Determine dimensions
    if state_indices is None:
        state_indices = tuple(range(len(state_limits)))

    n_dims = len(state_indices)

    # FIXED: Compute domain volume using enumeration index
    domain_volume = 1.0
    for i in range(len(state_limits)):
        domain_volume *= state_limits[i][1] - state_limits[i][0]

    # Infer total state dimension
    if hasattr(lyapunov_nn, "parameters"):
        try:
            first_layer = next(lyapunov_nn.parameters())
            nx_total = (
                first_layer.shape[1]
                if len(first_layer.shape) > 1
                else first_layer.shape[0]
            )
        except StopIteration:
            nx_total = len(state_limits)
    else:
        nx_total = len(state_limits)

    # FIXED: Generate Halton samples using enumeration
    bounds_for_halton = tuple(state_limits[i] for i in range(len(state_limits)))
    halton_samples = generate_halton_samples(
        num_samples, n_dims, bounds_for_halton, device
    )

    # Halton doesn't require power of 2, so num_samples stays the same
    num_samples_actual = halton_samples.shape[0]

    # Create full state samples
    samples = torch.zeros((num_samples_actual, nx_total), device=device)
    for i, idx in enumerate(state_indices):
        samples[:, idx] = halton_samples[:, i]

    # Evaluate Lyapunov function
    with torch.no_grad():
        V_values = lyapunov_nn(samples).squeeze()

    # Count samples in ROA
    in_roa = (V_values <= rho).sum().item()

    # Estimate ROA volume
    roa_volume = domain_volume * (in_roa / num_samples_actual)
    coverage_ratio = in_roa / num_samples_actual

    # Optionally compute discrepancy
    discrepancy_val = None
    if compute_discrepancy_metric:
        samples_normalized = np.zeros((num_samples_actual, n_dims))
        for i in range(len(state_limits)):
            low, high = state_limits[i]
            samples_normalized[:, i] = (halton_samples[:, i].cpu().numpy() - low) / (
                high - low
            )

        # Clip to [0,1]
        samples_normalized = np.clip(samples_normalized, 0.0, 1.0)
        discrepancy_val = compute_discrepancy(samples_normalized, method="CD")

    return ROAMetrics(
        rho=rho,
        area_roa=roa_volume,
        area_domain=domain_volume,
        coverage_ratio=coverage_ratio,
        num_samples_in_roa=in_roa,
        num_samples_total=num_samples_actual,
        domain_bounds=state_limits,
        method="qmc_halton",
        discrepancy=discrepancy_val,
    )


def compute_roa_area_monte_carlo(
    lyapunov_nn,
    state_limits: Tuple[Tuple[float, float], ...],
    rho: float,
    num_samples: int = 100000,
    device: str = "cpu",
    state_indices: Optional[Tuple[int, ...]] = None,
) -> ROAMetrics:
    """
    Compute ROA area using Monte Carlo sampling

    Args:
        lyapunov_nn: Lyapunov function V(x)
        state_limits: Tuple of (min, max) for EACH dimension in state_indices
        rho: ROA threshold (points where V(x) <= rho are in ROA)
        num_samples: Number of random samples
        device: Computing device
        state_indices: Which state dimensions to consider (None = all)

    Returns:
        ROAMetrics object with area measurements
    """

    # convert to floats if not already floats
    state_limits = tuple((to_float(lim[0]), to_float(lim[1])) for lim in state_limits)

    # Determine dimensions
    if state_indices is None:
        state_indices = tuple(range(len(state_limits)))

    n_dims = len(state_indices)

    # FIXED: Compute domain volume using enumeration index
    domain_volume = 1.0
    for i in range(len(state_limits)):
        domain_volume *= state_limits[i][1] - state_limits[i][0]

    # Get total state dimension from Lyapunov network
    if hasattr(lyapunov_nn, "parameters"):
        try:
            first_layer = next(lyapunov_nn.parameters())
            nx_total = (
                first_layer.shape[1]
                if len(first_layer.shape) > 1
                else first_layer.shape[0]
            )
        except StopIteration:
            # No parameters - assume nx equals number of limits
            nx_total = len(state_limits)
    else:
        nx_total = len(state_limits)

    # Generate random samples in domain
    samples = torch.zeros((num_samples, nx_total), device=device)

    for i, idx in enumerate(state_indices):
        low, high = state_limits[i]  # FIXED: use i not idx
        samples[:, idx] = torch.rand(num_samples, device=device) * (high - low) + low

    # Set other dimensions to zero or equilibrium
    for idx in range(nx_total):
        if idx not in state_indices:
            samples[:, idx] = 0.0

    # Evaluate Lyapunov function
    with torch.no_grad():
        V_values = lyapunov_nn(samples).squeeze()

    # Count samples in ROA
    in_roa = (V_values <= rho).sum().item()

    # Estimate ROA volume
    roa_volume = domain_volume * (in_roa / num_samples)
    coverage_ratio = in_roa / num_samples

    return ROAMetrics(
        rho=rho,
        area_roa=roa_volume,
        area_domain=domain_volume,
        coverage_ratio=coverage_ratio,
        num_samples_in_roa=in_roa,
        num_samples_total=num_samples,
        domain_bounds=state_limits,
        method="monte_carlo",
    )


def compute_roa_area_grid(
    lyapunov_nn,
    state_limits: Tuple[Tuple[float, float], Tuple[float, float]],
    rho: float,
    grid_resolution: int = 200,
    device: str = "cpu",
    state_indices: Tuple[int, int] = (0, 1),
) -> ROAMetrics:
    """
    Compute ROA area using uniform grid (more accurate for 2D)

    Args:
        lyapunov_nn: Lyapunov function V(x)
        state_limits: ((x_min, x_max), (y_min, y_max))
        rho: ROA threshold
        grid_resolution: Number of grid points per dimension
        device: Computing device
        state_indices: Which two state dimensions to use

    Returns:
        ROAMetrics object with area measurements
    """

    # convert to floats if not already floats
    state_limits = tuple((to_float(lim[0]), to_float(lim[1])) for lim in state_limits)

    idx0, idx1 = state_indices

    # Create grid
    x0_range = np.linspace(state_limits[0][0], state_limits[0][1], grid_resolution)
    x1_range = np.linspace(state_limits[1][0], state_limits[1][1], grid_resolution)
    X0, X1 = np.meshgrid(x0_range, x1_range, indexing="ij")

    # Cell area
    dx0 = (state_limits[0][1] - state_limits[0][0]) / grid_resolution
    dx1 = (state_limits[1][1] - state_limits[1][0]) / grid_resolution
    cell_area = dx0 * dx1

    # Total domain area
    domain_area = (state_limits[0][1] - state_limits[0][0]) * (
        state_limits[1][1] - state_limits[1][0]
    )

    # Infer total state dimension
    if hasattr(lyapunov_nn, "parameters"):
        first_layer = next(lyapunov_nn.parameters())
        nx_total = (
            first_layer.shape[1] if len(first_layer.shape) > 1 else first_layer.shape[0]
        )
    else:
        nx_total = max(state_indices) + 1

    # Initialize state grid
    num_points = grid_resolution * grid_resolution
    states_grid = torch.zeros((num_points, nx_total), device=device)
    states_grid[:, idx0] = torch.tensor(
        X0.flatten(), dtype=torch.float32, device=device
    )
    states_grid[:, idx1] = torch.tensor(
        X1.flatten(), dtype=torch.float32, device=device
    )

    # Evaluate Lyapunov function
    with torch.no_grad():
        V_values = lyapunov_nn(states_grid).squeeze()

    # Count cells in ROA
    in_roa = (V_values <= rho).sum().item()

    # Compute ROA area
    roa_area = in_roa * cell_area
    coverage_ratio = in_roa / num_points

    return ROAMetrics(
        rho=rho,
        area_roa=roa_area,
        area_domain=domain_area,
        coverage_ratio=coverage_ratio,
        num_samples_in_roa=in_roa,
        num_samples_total=num_points,
        domain_bounds=state_limits,
        method="grid",
        grid_resolution=grid_resolution,
    )


def compute_roa_volume_nd(
    lyapunov_nn,
    state_limits: Tuple[Tuple[float, float], ...],
    rho: float,
    samples_per_dim: int = 50,
    device: str = "cpu",
) -> ROAMetrics:
    """
    Compute ROA volume for arbitrary dimensional systems using grid method

    WARNING: Computational cost grows exponentially with dimension!
    For n_dims=3, samples_per_dim=50 -> 125,000 evaluations
    For n_dims=4, samples_per_dim=50 -> 6,250,000 evaluations

    Args:
        lyapunov_nn: Lyapunov function
        state_limits: Limits for each dimension
        rho: ROA threshold
        samples_per_dim: Grid points per dimension
        device: Computing device

    Returns:
        ROAMetrics with volume computation
    """

    # convert to floats if not already floats
    state_limits = tuple((to_float(lim[0]), to_float(lim[1])) for lim in state_limits)

    n_dims = len(state_limits)

    # Warn for high-dimensional cases
    total_points = samples_per_dim**n_dims
    if total_points > 1e7:
        import warnings

        warnings.warn(
            f"Computing {total_points:.2e} points in {n_dims}D. This may be slow. "
            f"Consider using Monte Carlo method instead."
        )

    # Create grid for each dimension
    grids = []
    cell_volume = 1.0
    domain_volume = 1.0

    for idx in range(n_dims):
        low, high = state_limits[idx]
        grid = np.linspace(low, high, samples_per_dim)
        grids.append(grid)

        dx = (high - low) / samples_per_dim
        cell_volume *= dx
        domain_volume *= high - low

    # Create meshgrid
    mesh = np.meshgrid(*grids, indexing="ij")

    # Flatten and create state tensor
    states = torch.zeros((total_points, n_dims), device=device, dtype=torch.float32)
    for i in range(n_dims):
        states[:, i] = torch.tensor(
            mesh[i].flatten(), dtype=torch.float32, device=device
        )

    # Evaluate Lyapunov function in batches to avoid memory issues
    batch_size = 10000
    V_values = []

    with torch.no_grad():
        for i in range(0, total_points, batch_size):
            batch = states[i : i + batch_size]
            V_batch = lyapunov_nn(batch).squeeze()
            V_values.append(V_batch)

    V_values = torch.cat(V_values)

    # Count points in ROA
    in_roa = (V_values <= rho).sum().item()

    # Compute volume
    roa_volume = in_roa * cell_volume
    coverage_ratio = in_roa / total_points

    return ROAMetrics(
        rho=rho,
        area_roa=roa_volume,
        area_domain=domain_volume,
        coverage_ratio=coverage_ratio,
        num_samples_in_roa=in_roa,
        num_samples_total=int(total_points),
        domain_bounds=state_limits,
        method=f"grid_{n_dims}d",
        grid_resolution=samples_per_dim,
    )


def estimate_rho_from_boundary(
    lyapunov_nn,
    state_limits: Tuple[Tuple[float, float], ...],
    grid_resolution: int = 100,
    device: str = "cpu",
    margin_factor: float = 0.9,
) -> float:
    """
    Estimate rho (ROA threshold) from minimum Lyapunov value on domain boundary

    Args:
        lyapunov_nn: Lyapunov function
        state_limits: Domain limits
        grid_resolution: Points per dimension on boundary
        device: Computing device
        margin_factor: Safety margin (rho = margin_factor * V_min_boundary)

    Returns:
        Estimated rho value
    """

    # convert to floats if not already floats
    state_limits = tuple((to_float(lim[0]), to_float(lim[1])) for lim in state_limits)

    n_dims = len(state_limits)

    # Sample points on each face of the hyperrectangle boundary
    boundary_points = []

    for dim in range(n_dims):
        # Create grid on lower face (dim-th coordinate at minimum)
        grids = []
        for d in range(n_dims):
            if d == dim:
                grids.append([state_limits[d][0]])  # Fixed at min
            else:
                grids.append(
                    np.linspace(state_limits[d][0], state_limits[d][1], grid_resolution)
                )

        mesh = np.meshgrid(*grids, indexing="ij")
        points_lower = np.stack([m.flatten() for m in mesh], axis=1)
        boundary_points.append(points_lower)

        # Create grid on upper face (dim-th coordinate at maximum)
        grids[dim] = [state_limits[dim][1]]  # Fixed at max
        mesh = np.meshgrid(*grids, indexing="ij")
        points_upper = np.stack([m.flatten() for m in mesh], axis=1)
        boundary_points.append(points_upper)

    # Concatenate all boundary points
    boundary_points = np.vstack(boundary_points)
    boundary_tensor = torch.tensor(boundary_points, dtype=torch.float32, device=device)

    # Evaluate Lyapunov function on boundary
    with torch.no_grad():
        V_boundary = lyapunov_nn(boundary_tensor).squeeze()

    # Minimum value on boundary
    rho = V_boundary.min().item() * margin_factor

    return rho


def compare_roa_metrics(
    lyapunov_nn,
    state_limits: Tuple[Tuple[float, float], ...],
    rho: float,
    device: str = "cpu",
    mc_samples: int = 100000,
    grid_resolution: int = 200,
    state_indices: Optional[Tuple[int, int]] = None,
    compute_discrepancy: bool = False,
) -> Dict[str, ROAMetrics]:
    """
    Compare ROA area estimates using different methods

    Args:
        lyapunov_nn: Lyapunov function
        state_limits: Domain limits
        rho: ROA threshold
        device: Computing device
        mc_samples: Samples for Monte Carlo and QMC
        grid_resolution: Resolution for grid method
        state_indices: Which dimensions to analyze (for 2D visualization)
        compute_discrepancy: Whether to compute discrepancy metric

    Returns:
        Dict with metrics from different methods
    """

    # convert to floats if not already floats
    state_limits = tuple((to_float(lim[0]), to_float(lim[1])) for lim in state_limits)

    results = {}

    # Random Monte Carlo
    print("Computing ROA area via Monte Carlo (random)...")
    mc_metrics = compute_roa_area_monte_carlo(
        lyapunov_nn, state_limits, rho, num_samples=mc_samples, device=device
    )
    results["monte_carlo"] = mc_metrics

    # Sobol QMC
    print("Computing ROA area via Quasi-Monte Carlo (Sobol)...")
    sobol_metrics = compute_roa_area_qmc_sobol(
        lyapunov_nn,
        state_limits,
        rho,
        num_samples=mc_samples,
        device=device,
        state_indices=state_indices,
        compute_discrepancy_metric=compute_discrepancy,
    )
    results["qmc_sobol"] = sobol_metrics

    # Halton QMC
    print("Computing ROA area via Quasi-Monte Carlo (Halton)...")
    halton_metrics = compute_roa_area_qmc_halton(
        lyapunov_nn,
        state_limits,
        rho,
        num_samples=mc_samples,
        device=device,
        state_indices=state_indices,
        compute_discrepancy_metric=compute_discrepancy,
    )
    results["qmc_halton"] = halton_metrics

    # Grid method (for 2D only)
    if state_indices is not None or len(state_limits) == 2:
        print("Computing ROA area via grid method...")
        if state_indices is None:
            state_indices = (0, 1)
            limits_2d = (state_limits[0], state_limits[1])
        else:
            limits_2d = (state_limits[state_indices[0]], state_limits[state_indices[1]])

        grid_metrics = compute_roa_area_grid(
            lyapunov_nn,
            limits_2d,
            rho,
            grid_resolution=grid_resolution,
            device=device,
            state_indices=state_indices,
        )
        results["grid"] = grid_metrics

    # Print comparison
    print("\n" + "=" * 70)
    print("Method Comparison")
    print("=" * 70)
    print(f"{'Method':<20} {'Area':<12} {'Coverage':<12} {'Samples in ROA':<15}")
    print("-" * 70)
    for method_name, metrics in results.items():
        print(
            f"{method_name:<20} {metrics.area_roa:<12.6f} {metrics.coverage_ratio*100:<11.2f}% "
            f"{metrics.num_samples_in_roa:>14,}"
        )

    if compute_discrepancy:
        print("\nDiscrepancy (lower = more uniform):")
        for method_name, metrics in results.items():
            if metrics.discrepancy is not None:
                print(f"  {method_name:<20}: {metrics.discrepancy:.6f}")

    print("=" * 70)

    return results


def compute_roa_area_with_controller(
    lyapunov_nn,
    controller_nn,
    dynamics_system,
    state_limits: Tuple[Tuple[float, float], ...],
    rho: Optional[float] = None,
    method: str = "monte_carlo",
    num_samples: int = 100000,
    grid_resolution: int = 200,
    device: str = "cpu",
    verify_lyapunov_decrease: bool = True,
    state_indices: Optional[Tuple[int, int]] = None,
) -> Dict:
    """
    Compute ROA area and verify Lyapunov decrease condition

    Args:
        lyapunov_nn: Lyapunov function
        controller_nn: Neural controller
        dynamics_system: Dynamical system
        state_limits: Domain limits
        rho: ROA threshold (auto-computed if None)
        method: 'monte_carlo' or 'grid'
        num_samples: For Monte Carlo
        grid_resolution: For grid method
        device: Computing device
        verify_lyapunov_decrease: Check ΔV < 0 in ROA
        state_indices: Dimensions to analyze

    Returns:
        Dict with metrics and verification results
    """

    # convert to floats if not already floats
    state_limits = tuple((to_float(lim[0]), to_float(lim[1])) for lim in state_limits)

    # Auto-compute rho if not provided
    if rho is None:
        print("Estimating rho from boundary...")
        rho = estimate_rho_from_boundary(
            lyapunov_nn, state_limits, grid_resolution=50, device=device
        )
        print(f"Estimated rho = {rho:.4f}")

    # Compute ROA area
    if method == "monte_carlo":
        metrics = compute_roa_area_monte_carlo(
            lyapunov_nn,
            state_limits,
            rho,
            num_samples=num_samples,
            device=device,
            state_indices=state_indices,
        )
    elif method == "grid" and len(state_limits) == 2:
        metrics = compute_roa_area_grid(
            lyapunov_nn,
            state_limits,
            rho,
            grid_resolution=grid_resolution,
            device=device,
            state_indices=state_indices or (0, 1),
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    result = {"metrics": metrics}

    # Verify Lyapunov decrease condition
    if verify_lyapunov_decrease:
        print("\nVerifying Lyapunov decrease condition ΔV < 0...")

        # Sample points in ROA
        if method == "monte_carlo":
            # Generate samples
            n_verify = min(10000, num_samples)
            if state_indices is None:
                state_indices = tuple(range(len(state_limits)))

            nx_total = max(state_indices) + 1
            samples = torch.zeros((n_verify, nx_total), device=device)

            for idx in state_indices:
                low, high = state_limits[idx]
                samples[:, idx] = (
                    torch.rand(n_verify, device=device) * (high - low) + low
                )
        else:
            # Use grid samples
            idx0, idx1 = state_indices
            x0_range = np.linspace(
                state_limits[0][0], state_limits[0][1], grid_resolution
            )
            x1_range = np.linspace(
                state_limits[1][0], state_limits[1][1], grid_resolution
            )
            X0, X1 = np.meshgrid(x0_range, x1_range, indexing="ij")

            nx_total = dynamics_system.nx
            samples = torch.zeros((grid_resolution**2, nx_total), device=device)
            samples[:, idx0] = torch.tensor(
                X0.flatten(), dtype=torch.float32, device=device
            )
            samples[:, idx1] = torch.tensor(
                X1.flatten(), dtype=torch.float32, device=device
            )

        # Evaluate V and ΔV
        with torch.no_grad():
            V_samples = lyapunov_nn(samples).squeeze()
            in_roa_mask = V_samples <= rho

            # Only check points in ROA
            samples_in_roa = samples[in_roa_mask]

            if len(samples_in_roa) > 0:
                # Compute control and next state
                u_samples = controller_nn(samples_in_roa)
                x_next = dynamics_system(samples_in_roa, u_samples)
                V_next = lyapunov_nn(x_next).squeeze()

                # Lyapunov decrease
                V_current = V_samples[in_roa_mask]
                delta_V = V_next - V_current

                # Statistics
                num_decreasing = (delta_V < 0).sum().item()
                num_total = len(delta_V)
                decrease_ratio = num_decreasing / num_total
                max_violation = delta_V.max().item()
                mean_decrease = delta_V.mean().item()

                result["lyapunov_verification"] = {
                    "decrease_ratio": decrease_ratio,
                    "num_decreasing": num_decreasing,
                    "num_total": num_total,
                    "max_violation": max_violation,
                    "mean_delta_V": mean_decrease,
                    "all_decreasing": decrease_ratio == 1.0,
                }

                print(f"  Decreasing at {decrease_ratio*100:.2f}% of ROA points")
                print(f"  Max ΔV violation: {max_violation:.6f}")
                print(f"  Mean ΔV: {mean_decrease:.6f}")
            else:
                print("  Warning: No samples found in ROA!")
                result["lyapunov_verification"] = None

    return result


def compare_roa_sizes(
    lyapunov_models: Dict[str, torch.nn.Module],
    state_limits: Tuple[Tuple[float, float], ...],
    rho_values: Optional[Dict[str, float]] = None,
    method: str = "monte_carlo",
    num_samples: int = 100000,
    grid_resolution: int = 200,
    device: str = "cpu",
    **kwargs,
) -> Dict[str, ROAMetrics]:
    """
    Compare ROA sizes for multiple Lyapunov functions

    Useful for comparing different training runs or architectures

    Args:
        lyapunov_models: Dict of {name: lyapunov_nn}
        state_limits: Domain limits
        rho_values: Optional dict of {name: rho}
        method: 'monte_carlo', 'qmc_sobol', 'qmc_halton', or 'grid'
        num_samples: Number of samples for MC/QMC methods
        grid_resolution: Resolution for grid method
        device: Computing device
        **kwargs: Additional arguments for computation methods

    Returns:
        Dict of {name: ROAMetrics}
    """

    # convert to floats if not already floats
    state_limits = tuple((to_float(lim[0]), to_float(lim[1])) for lim in state_limits)

    results = {}

    for name, lyap_nn in lyapunov_models.items():
        print(f"\nComputing ROA for '{name}'...")

        rho = rho_values[name] if rho_values and name in rho_values else None

        if rho is None:
            rho = estimate_rho_from_boundary(lyap_nn, state_limits, device=device)
            print(f"  Estimated rho = {rho:.4f}")

        if method == "monte_carlo":
            metrics = compute_roa_area_monte_carlo(
                lyap_nn, state_limits, rho, num_samples=num_samples, device=device
            )
        elif method == "qmc_sobol":
            metrics = compute_roa_area_qmc_sobol(
                lyap_nn, state_limits, rho, num_samples=num_samples, device=device
            )
        elif method == "qmc_halton":
            metrics = compute_roa_area_qmc_halton(
                lyap_nn, state_limits, rho, num_samples=num_samples, device=device
            )
        elif method == "grid":
            if len(state_limits) != 2:
                raise ValueError("Grid method only supports 2D")
            metrics = compute_roa_area_grid(
                lyap_nn,
                state_limits,
                rho,
                grid_resolution=grid_resolution,
                device=device,
                state_indices=(0, 1),
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        results[name] = metrics
        print(f"  ROA area: {metrics.area_roa:.4f}")
        print(f"  Coverage: {metrics.coverage_ratio*100:.2f}%")

    return results


def print_roa_metrics(metrics: ROAMetrics, title: Optional[str] = None):
    """
    Pretty print ROA metrics

    Args:
        metrics: ROAMetrics object
        title: Optional title
    """
    if title:
        print(f"\n{'='*70}")
        print(f"{title}")
        print(f"{'='*70}")

    print(f"Method: {metrics.method}")
    if metrics.grid_resolution:
        print(f"Grid resolution: {metrics.grid_resolution}")
    print(f"Total samples: {metrics.num_samples_total:,}")
    print(f"Samples in ROA: {metrics.num_samples_in_roa:,}")
    print(f"\nROA threshold (ρ): {metrics.rho:.6f}")
    print(f"Domain area/volume: {metrics.area_domain:.6f}")
    print(f"ROA area/volume: {metrics.area_roa:.6f}")
    print(f"Coverage ratio: {metrics.coverage_ratio*100:.6f}%")
    print(f"{'='*70}")


def compute_lyapunov_difference_discrete(
    states: torch.Tensor,
    lyapunov_nn: torch.nn.Module,
    controller_nn: torch.nn.Module,
    dynamics_fn: Callable,
    observer_nn: Optional[torch.nn.Module] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Lyapunov difference ΔV(z) = V(z_{k+1}) - V(z_k) for discrete-time systems

    For output feedback with observer:
    - Assumes converged observer (estimation error e = 0)
    - Evaluates V([x, 0]) showing ideal closed-loop behavior
    - Controller may receive [x_hat, y] (auto-detected based on input dimension)

    Args:
        states: State samples - ALWAYS AUGMENTED when observer is present
                - State feedback: (batch_size, nx) containing physical state x
                - Output feedback: (batch_size, 2*nx) containing [x, e]
                  where x is physical state and e is estimation error
        lyapunov_nn: Lyapunov function
                    - State feedback: V(x), input dimension = nx
                    - Output feedback: V([x, e]), input dimension = 2*nx
        controller_nn: Controller
                      - State feedback: u = π(x), input dimension = nx
                      - Output feedback: u = π(x_hat) or u = π([x_hat, y])
                        Input dimension auto-detected and augmented if needed
        dynamics_fn: Discrete dynamics x_{k+1} = f(x_k, u_k)
                    Must have .h(x) method (or .continuous_time_system.h(x) for legacy)
                    Must have .nx or .continuous_time_system.nx attribute
        observer_nn: Optional Luenberger observer x_hat_{k+1} = g(x_hat_k, u_k, y_{k+1})
                    If provided, assumes output feedback with augmented Lyapunov V([x,e])

    Returns:
        V_current: V(z_k) values (batch_size,)
                  - State feedback: V(x_k)
                  - Output feedback: V([x_k, e_k])
        V_next: V(z_{k+1}) values (batch_size,)
                - State feedback: V(x_{k+1})
                - Output feedback: V([x_{k+1}, e_{k+1}])
        delta_V: ΔV = V(z_{k+1}) - V(z_k) values (batch_size,)

    Notes:
        - For ideal behavior analysis (typical use), the metric functions pass
          states = [x, 0] where e=0 represents converged observer
        - Controller input is automatically augmented with measurement y if needed
          based on controller's expected input dimension
        - Supports both legacy DiscreteTimeSystem and new GenericDiscreteTimeSystem
    """
    with torch.no_grad():
        legacy = False
        # Check if the system is legacy in nature
        if isinstance(dynamics_fn, DiscreteTimeSystem):
            legacy = True

        # Get physical state dimension
        if hasattr(dynamics_fn, "continuous_time_system"):
            nx = dynamics_fn.continuous_time_system.nx
        else:
            nx = dynamics_fn.nx

        if observer_nn is not None:
            # Output feedback: states = [x, e]
            if states.shape[1] != 2 * nx:
                raise ValueError(
                    f"For output feedback, states must have shape (batch, 2*nx={2*nx}), "
                    f"got shape {states.shape}"
                )

            x_true = states[:, :nx]  # Physical state
            e_current = states[:, nx:]  # Estimation error
            x_hat_current = x_true - e_current  # Observer estimate

            # Current Lyapunov value V([x, e])
            V_current = lyapunov_nn(states).squeeze()

            # Get measurement from TRUE state
            if legacy:
                y = dynamics_fn.continuous_time_system.h(x_true)
            else:
                y = dynamics_fn.h(x_true)

            # Prepare controller input - may need to augment with measurement
            # Infer expected controller input dimension
            controller_in_dim = None
            if hasattr(controller_nn, "net"):
                controller_in_dim = controller_nn.net[0].in_features
            elif hasattr(controller_nn, "x_equilibrium"):
                controller_in_dim = controller_nn.x_equilibrium.shape[0]

            # Check if we need to augment observer estimate with measurement
            if (
                controller_in_dim is not None
                and x_hat_current.shape[1] < controller_in_dim
            ):
                # Controller expects more inputs than just x_hat
                deficit = controller_in_dim - x_hat_current.shape[1]

                if deficit == y.shape[1]:
                    # Most common: controller takes [x_hat, y]
                    controller_input = torch.cat([x_hat_current, y], dim=1)
                else:
                    # Fallback: pad with zeros
                    padding = torch.zeros(
                        (x_hat_current.shape[0], deficit), device=x_hat_current.device
                    )
                    controller_input = torch.cat([x_hat_current, padding], dim=1)
            else:
                # Controller takes just x_hat
                controller_input = x_hat_current

            # Compute control
            u = controller_nn(controller_input)

            # TRUE state evolves
            x_next = dynamics_fn(x_true, u)

            # Next measurement
            if legacy:
                y_next = dynamics_fn.continuous_time_system.h(x_next)
            else:
                y_next = dynamics_fn.h(x_next)

            # Observer update
            x_hat_next = observer_nn(x_hat_current, u, y_next)

            # Next estimation error
            e_next = x_next - x_hat_next

            # Next augmented state z_{k+1} = [x_{k+1}, e_{k+1}]
            states_next = torch.cat([x_next, e_next], dim=1)
            V_next = lyapunov_nn(states_next).squeeze()

        else:
            # State feedback: states = x
            if states.shape[1] != nx:
                raise ValueError(
                    f"For state feedback, states must have shape (batch, nx={nx}), "
                    f"got shape {states.shape}"
                )

            x_true = states

            # Current Lyapunov value V(x)
            V_current = lyapunov_nn(x_true).squeeze()

            # Direct state feedback
            u = controller_nn(x_true)

            # Next state
            x_next = dynamics_fn(x_true, u)

            # Next Lyapunov value V(x_{k+1})
            V_next = lyapunov_nn(x_next).squeeze()

        # Lyapunov difference
        delta_V = V_next - V_current

    return V_current, V_next, delta_V


def compute_lyapunov_difference_metrics_monte_carlo(
    lyapunov_nn: torch.nn.Module,
    controller_nn: torch.nn.Module,
    dynamics_fn: Callable,
    state_limits: Tuple[Tuple[float, float], ...],
    rho: float,
    num_samples: int = 100000,
    device: str = "cpu",
    observer_nn: Optional[torch.nn.Module] = None,
    state_indices: Optional[Tuple[int, ...]] = None,
    stability_threshold: float = 0.0,
) -> LyapunovDifferenceMetrics:
    """
    Compute Lyapunov difference metrics using Monte Carlo sampling

    Samples over PHYSICAL state space only. For output feedback with observer,
    assumes converged observer (e=0) and augments samples internally with [x, 0].

    Args:
        lyapunov_nn: Lyapunov function
                    - State feedback: V(x), input dimension = nx
                    - Output feedback: V([x, e]), input dimension = 2*nx
        controller_nn: Controller π(x) or π(x_hat)
        dynamics_fn: Discrete dynamics x_{k+1} = f(x_k, u_k)
        state_limits: Domain bounds for dimensions in state_indices (IN ORDER)
        rho: ROA threshold
        num_samples: Number of random samples
        device: Computing device
        observer_nn: Optional observer
        state_indices: Which PHYSICAL dimensions to sample (None = all)
        stability_threshold: Threshold for stability (typically 0.0)

    Returns:
        LyapunovDifferenceMetrics with comprehensive statistics
    """

    # Convert to floats
    state_limits = tuple((to_float(lim[0]), to_float(lim[1])) for lim in state_limits)

    # Determine dimensions
    if state_indices is None:
        state_indices = tuple(range(len(state_limits)))

    n_dims = len(state_indices)

    # FIXED: Compute domain volume using enumeration index
    domain_volume = 1.0
    for i in range(len(state_limits)):
        domain_volume *= state_limits[i][1] - state_limits[i][0]

    # Get PHYSICAL state dimension from dynamics system
    if hasattr(dynamics_fn, "continuous_time_system"):
        nx = dynamics_fn.continuous_time_system.nx
    elif hasattr(dynamics_fn, "nx"):
        nx = dynamics_fn.nx
    else:
        # Fallback: infer from state_limits
        nx = len(state_limits)

    # VALIDATION: Ensure state_indices are valid for PHYSICAL state
    if max(state_indices) >= nx:
        raise ValueError(
            f"state_indices {state_indices} contains index >= nx={nx}. "
            f"For output feedback, state_indices must reference PHYSICAL state dimensions only (0 to {nx-1}), "
            f"not augmented state dimensions. "
            f"Valid indices: {tuple(range(nx))}"
        )

    if len(state_limits) != len(state_indices):
        raise ValueError(
            f"state_limits has {len(state_limits)} entries but state_indices has {len(state_indices)} entries. "
            f"These should match."
        )

    # Generate random samples for PHYSICAL state only
    samples_x = torch.zeros((num_samples, nx), device=device)
    for i, idx in enumerate(state_indices):
        low, high = state_limits[i]  # FIXED: use i not idx
        samples_x[:, idx] = torch.rand(num_samples, device=device) * (high - low) + low

    # Set other PHYSICAL dimensions to equilibrium
    if hasattr(dynamics_fn, "continuous_time_system"):
        x_eq = dynamics_fn.continuous_time_system.x_equilibrium.to(device)
    else:
        x_eq = dynamics_fn.x_equilibrium.to(device)

    for idx in range(nx):
        if idx not in state_indices:
            samples_x[:, idx] = x_eq[idx]

    # Prepare samples for Lyapunov evaluation
    if observer_nn is not None:
        # Output feedback: augment with zero estimation error (ideal behavior)
        samples_e = torch.zeros((num_samples, nx), device=device)
        samples = torch.cat([samples_x, samples_e], dim=1)
    else:
        samples = samples_x

    # Compute Lyapunov values and differences
    V_current, V_next, delta_V = compute_lyapunov_difference_discrete(
        samples, lyapunov_nn, controller_nn, dynamics_fn, observer_nn=observer_nn
    )

    # Classify samples
    in_roa = V_current <= rho
    is_decreasing = delta_V <= stability_threshold
    verified_roa = in_roa & is_decreasing

    # Count samples
    num_in_roa = in_roa.sum().item()
    num_decreasing = is_decreasing.sum().item()
    num_verified_roa = verified_roa.sum().item()

    # Estimate areas/volumes
    area_roa = domain_volume * (num_in_roa / num_samples)
    area_decreasing = domain_volume * (num_decreasing / num_samples)
    area_verified_roa = domain_volume * (num_verified_roa / num_samples)

    # Compute statistics
    mean_delta_V = delta_V.mean().item()
    max_delta_V = delta_V.max().item()
    min_delta_V = delta_V.min().item()
    std_delta_V = delta_V.std().item()

    # Statistics within ROA
    if num_in_roa > 0:
        delta_V_in_roa = delta_V[in_roa]
        mean_delta_V_in_roa = delta_V_in_roa.mean().item()
        max_violation_in_roa = delta_V_in_roa.max().item()
        percent_verified = (num_verified_roa / num_in_roa) * 100.0
    else:
        mean_delta_V_in_roa = 0.0
        max_violation_in_roa = 0.0
        percent_verified = 0.0

    return LyapunovDifferenceMetrics(
        rho=rho,
        area_roa=area_roa,
        area_decreasing=area_decreasing,
        area_verified_roa=area_verified_roa,
        area_domain=domain_volume,
        coverage_roa=num_in_roa / num_samples,
        coverage_decreasing=num_decreasing / num_samples,
        coverage_verified_roa=num_verified_roa / num_samples,
        num_samples_in_roa=num_in_roa,
        num_samples_decreasing=num_decreasing,
        num_samples_verified_roa=num_verified_roa,
        num_samples_total=num_samples,
        mean_delta_V=mean_delta_V,
        mean_delta_V_in_roa=mean_delta_V_in_roa,
        max_delta_V=max_delta_V,
        min_delta_V=min_delta_V,
        std_delta_V=std_delta_V,
        max_violation_in_roa=max_violation_in_roa,
        percent_verified=percent_verified,
        domain_bounds=state_limits,
        method="monte_carlo",
        stability_threshold=stability_threshold,
    )


def compute_lyapunov_difference_metrics_qmc_sobol(
    lyapunov_nn: torch.nn.Module,
    controller_nn: torch.nn.Module,
    dynamics_fn: Callable,
    state_limits: Tuple[Tuple[float, float], ...],
    rho: float,
    num_samples: int = 100000,
    device: str = "cpu",
    observer_nn: Optional[torch.nn.Module] = None,
    state_indices: Optional[Tuple[int, ...]] = None,
    stability_threshold: float = 0.0,
    compute_discrepancy_metric: bool = False,
    round_to_pow2: bool = True,
) -> LyapunovDifferenceMetrics:
    """
    Compute Lyapunov difference metrics using Quasi-Monte Carlo (Sobol sequence)

    Sobol sequences provide better coverage than random sampling,
    typically requiring 10-100x fewer samples for same accuracy.

    Samples over PHYSICAL state space only. For output feedback with observer,
    assumes converged observer (e=0) and augments samples internally with [x, 0].

    Args:
        lyapunov_nn: Lyapunov function
        controller_nn: Controller
        dynamics_fn: Discrete dynamics
        state_limits: Domain bounds for dimensions in state_indices (IN ORDER)
        rho: ROA threshold
        num_samples: Number of Sobol samples (will be rounded to power of 2)
        device: Computing device
        observer_nn: Optional observer
        state_indices: Which PHYSICAL dimensions to sample (None = all)
        stability_threshold: Threshold for stability (typically 0.0)
        compute_discrepancy_metric: Whether to compute discrepancy (slow for large N)
        round_to_pow2: Round sample count to power of 2 for optimal Sobol properties

    Returns:
        LyapunovDifferenceMetrics with comprehensive statistics
    """

    # Convert to floats
    state_limits = tuple((to_float(lim[0]), to_float(lim[1])) for lim in state_limits)

    # Determine dimensions
    if state_indices is None:
        state_indices = tuple(range(len(state_limits)))

    n_dims = len(state_indices)

    # FIXED: Compute domain volume using enumeration index
    domain_volume = 1.0
    for i in range(len(state_limits)):
        domain_volume *= state_limits[i][1] - state_limits[i][0]

    # Get PHYSICAL state dimension from dynamics system
    if hasattr(dynamics_fn, "continuous_time_system"):
        nx = dynamics_fn.continuous_time_system.nx
    elif hasattr(dynamics_fn, "nx"):
        nx = dynamics_fn.nx
    else:
        # Fallback: assume state_limits defines the full physical state
        nx = len(state_limits)

    # VALIDATION
    if max(state_indices) >= nx:
        raise ValueError(
            f"state_indices {state_indices} contains index >= nx={nx}. "
            f"For output feedback, state_indices must reference PHYSICAL state dimensions only (0 to {nx-1}), "
            f"not augmented state dimensions. "
            f"Valid indices: {tuple(range(nx))}"
        )

    if len(state_limits) != len(state_indices):
        raise ValueError(
            f"state_limits has {len(state_limits)} entries but state_indices has {len(state_indices)} entries. "
            f"These should match."
        )

    # FIXED: Generate Sobol samples using enumeration
    bounds_for_sobol = tuple(state_limits[i] for i in range(len(state_limits)))
    sobol_samples = generate_sobol_samples(
        num_samples, n_dims, bounds_for_sobol, device, round_to_pow2
    )

    num_samples_actual = sobol_samples.shape[0]

    # Create PHYSICAL state samples
    samples_x = torch.zeros((num_samples_actual, nx), device=device)
    for i, idx in enumerate(state_indices):
        samples_x[:, idx] = sobol_samples[:, i]

    # Set other PHYSICAL dimensions to equilibrium
    if hasattr(dynamics_fn, "continuous_time_system"):
        x_eq = dynamics_fn.continuous_time_system.x_equilibrium.to(device)
    else:
        x_eq = dynamics_fn.x_equilibrium.to(device)

    for idx in range(nx):
        if idx not in state_indices:
            samples_x[:, idx] = x_eq[idx]

    # Prepare input for Lyapunov function
    if observer_nn is not None:
        # Output feedback: augment with zero estimation error (ideal behavior)
        samples_e = torch.zeros((num_samples_actual, nx), device=device)
        samples = torch.cat([samples_x, samples_e], dim=1)
    else:
        samples = samples_x

    # Compute Lyapunov values and differences
    # Pass AUGMENTED samples to compute_lyapunov_difference_discrete
    V_current, V_next, delta_V = compute_lyapunov_difference_discrete(
        samples, lyapunov_nn, controller_nn, dynamics_fn, observer_nn=observer_nn
    )

    # Classify samples
    in_roa = V_current <= rho
    is_decreasing = delta_V <= stability_threshold
    verified_roa = in_roa & is_decreasing

    # Count samples
    num_in_roa = in_roa.sum().item()
    num_decreasing = is_decreasing.sum().item()
    num_verified_roa = verified_roa.sum().item()

    # Estimate areas/volumes
    area_roa = domain_volume * (num_in_roa / num_samples_actual)
    area_decreasing = domain_volume * (num_decreasing / num_samples_actual)
    area_verified_roa = domain_volume * (num_verified_roa / num_samples_actual)

    # Compute statistics
    mean_delta_V = delta_V.mean().item()
    max_delta_V = delta_V.max().item()
    min_delta_V = delta_V.min().item()
    std_delta_V = delta_V.std().item()

    # Statistics within ROA
    if num_in_roa > 0:
        delta_V_in_roa = delta_V[in_roa]
        mean_delta_V_in_roa = delta_V_in_roa.mean().item()
        max_violation_in_roa = delta_V_in_roa.max().item()
        percent_verified = (num_verified_roa / num_in_roa) * 100.0
    else:
        mean_delta_V_in_roa = 0.0
        max_violation_in_roa = 0.0
        percent_verified = 0.0

    # Optionally compute discrepancy
    discrepancy_val = None
    if compute_discrepancy_metric:
        samples_normalized = np.zeros((num_samples_actual, n_dims))
        for i in range(len(state_limits)):  # FIXED: use range(len(state_limits))
            low, high = state_limits[i]
            samples_normalized[:, i] = (sobol_samples[:, i].cpu().numpy() - low) / (
                high - low
            )
        samples_normalized = np.clip(samples_normalized, 0.0, 1.0)
        discrepancy_val = compute_discrepancy(samples_normalized, method="CD")

    return LyapunovDifferenceMetrics(
        rho=rho,
        area_roa=area_roa,
        area_decreasing=area_decreasing,
        area_verified_roa=area_verified_roa,
        area_domain=domain_volume,
        coverage_roa=num_in_roa / num_samples_actual,
        coverage_decreasing=num_decreasing / num_samples_actual,
        coverage_verified_roa=num_verified_roa / num_samples_actual,
        num_samples_in_roa=num_in_roa,
        num_samples_decreasing=num_decreasing,
        num_samples_verified_roa=num_verified_roa,
        num_samples_total=num_samples_actual,
        mean_delta_V=mean_delta_V,
        mean_delta_V_in_roa=mean_delta_V_in_roa,
        max_delta_V=max_delta_V,
        min_delta_V=min_delta_V,
        std_delta_V=std_delta_V,
        max_violation_in_roa=max_violation_in_roa,
        percent_verified=percent_verified,
        domain_bounds=state_limits,
        method="qmc_sobol",
        stability_threshold=stability_threshold,
        discrepancy=discrepancy_val,
    )


def compute_lyapunov_difference_metrics_qmc_halton(
    lyapunov_nn: torch.nn.Module,
    controller_nn: torch.nn.Module,
    dynamics_fn: Callable,
    state_limits: Tuple[Tuple[float, float], ...],
    rho: float,
    num_samples: int = 100000,
    device: str = "cpu",
    observer_nn: Optional[torch.nn.Module] = None,
    state_indices: Optional[Tuple[int, ...]] = None,
    stability_threshold: float = 0.0,
    compute_discrepancy_metric: bool = False,
) -> LyapunovDifferenceMetrics:
    """
    Compute Lyapunov difference metrics using Quasi-Monte Carlo (Halton sequence)

    Halton sequences are particularly good for lower dimensions (d ≤ 10).

    Samples over PHYSICAL state space only. For output feedback with observer,
    assumes converged observer (e=0) and augments samples internally with [x, 0].

    Args:
        lyapunov_nn: Lyapunov function
        controller_nn: Controller
        dynamics_fn: Discrete dynamics
        state_limits: Domain bounds for dimensions in state_indices (IN ORDER)
        rho: ROA threshold
        num_samples: Number of Halton samples
        device: Computing device
        observer_nn: Optional observer
        state_indices: Which PHYSICAL dimensions to sample (None = all)
        stability_threshold: Threshold for stability (typically 0.0)
        compute_discrepancy_metric: Whether to compute discrepancy

    Returns:
        LyapunovDifferenceMetrics with comprehensive statistics
    """

    # Convert to floats
    state_limits = tuple((to_float(lim[0]), to_float(lim[1])) for lim in state_limits)

    # Determine dimensions
    if state_indices is None:
        state_indices = tuple(range(len(state_limits)))

    n_dims = len(state_indices)

    # FIXED: Compute domain volume using enumeration index
    domain_volume = 1.0
    for i in range(len(state_limits)):
        domain_volume *= state_limits[i][1] - state_limits[i][0]

    # Get PHYSICAL state dimension
    if hasattr(dynamics_fn, "continuous_time_system"):
        nx = dynamics_fn.continuous_time_system.nx
    elif hasattr(dynamics_fn, "nx"):
        nx = dynamics_fn.nx
    else:
        # Fallback: infer from state_limits
        nx = len(state_limits)

    # VALIDATION
    if max(state_indices) >= nx:
        raise ValueError(
            f"state_indices {state_indices} contains index >= nx={nx}. "
            f"For output feedback, state_indices must reference PHYSICAL state dimensions only (0 to {nx-1}), "
            f"not augmented state dimensions. "
            f"Valid indices: {tuple(range(nx))}"
        )

    if len(state_limits) != len(state_indices):
        raise ValueError(
            f"state_limits has {len(state_limits)} entries but state_indices has {len(state_indices)} entries. "
            f"These should match."
        )

    # FIXED: Generate Halton samples using enumeration
    bounds_for_halton = tuple(state_limits[i] for i in range(len(state_limits)))
    halton_samples = generate_halton_samples(
        num_samples, n_dims, bounds_for_halton, device
    )

    num_samples_actual = halton_samples.shape[0]

    # Create PHYSICAL state samples
    samples_x = torch.zeros((num_samples_actual, nx), device=device)
    for i, idx in enumerate(state_indices):
        samples_x[:, idx] = halton_samples[:, i]

    # Set other dimensions to equilibrium
    if hasattr(dynamics_fn, "continuous_time_system"):
        x_eq = dynamics_fn.continuous_time_system.x_equilibrium.to(device)
    else:
        x_eq = dynamics_fn.x_equilibrium.to(device)

    for idx in range(nx):
        if idx not in state_indices:
            samples_x[:, idx] = x_eq[idx]

    # Prepare samples for Lyapunov evaluation
    if observer_nn is not None:
        # Output feedback: augment with zero estimation error (ideal behavior)
        samples_e = torch.zeros((num_samples_actual, nx), device=device)
        samples = torch.cat([samples_x, samples_e], dim=1)
    else:
        samples = samples_x

    # Compute Lyapunov values and differences
    V_current, V_next, delta_V = compute_lyapunov_difference_discrete(
        samples, lyapunov_nn, controller_nn, dynamics_fn, observer_nn=observer_nn
    )

    # Classify samples
    in_roa = V_current <= rho
    is_decreasing = delta_V <= stability_threshold
    verified_roa = in_roa & is_decreasing

    # Count samples
    num_in_roa = in_roa.sum().item()
    num_decreasing = is_decreasing.sum().item()
    num_verified_roa = verified_roa.sum().item()

    # Estimate areas/volumes
    area_roa = domain_volume * (num_in_roa / num_samples_actual)
    area_decreasing = domain_volume * (num_decreasing / num_samples_actual)
    area_verified_roa = domain_volume * (num_verified_roa / num_samples_actual)

    # Compute statistics
    mean_delta_V = delta_V.mean().item()
    max_delta_V = delta_V.max().item()
    min_delta_V = delta_V.min().item()
    std_delta_V = delta_V.std().item()

    # Statistics within ROA
    if num_in_roa > 0:
        delta_V_in_roa = delta_V[in_roa]
        mean_delta_V_in_roa = delta_V_in_roa.mean().item()
        max_violation_in_roa = delta_V_in_roa.max().item()
        percent_verified = (num_verified_roa / num_in_roa) * 100.0
    else:
        mean_delta_V_in_roa = 0.0
        max_violation_in_roa = 0.0
        percent_verified = 0.0

    # Optionally compute discrepancy
    discrepancy_val = None
    if compute_discrepancy_metric:
        samples_normalized = np.zeros((num_samples_actual, n_dims))
        for i in range(len(state_limits)):  # FIXED: use range(len(state_limits))
            low, high = state_limits[i]
            samples_normalized[:, i] = (halton_samples[:, i].cpu().numpy() - low) / (
                high - low
            )
        samples_normalized = np.clip(samples_normalized, 0.0, 1.0)
        discrepancy_val = compute_discrepancy(samples_normalized, method="CD")

    return LyapunovDifferenceMetrics(
        rho=rho,
        area_roa=area_roa,
        area_decreasing=area_decreasing,
        area_verified_roa=area_verified_roa,
        area_domain=domain_volume,
        coverage_roa=num_in_roa / num_samples_actual,
        coverage_decreasing=num_decreasing / num_samples_actual,
        coverage_verified_roa=num_verified_roa / num_samples_actual,
        num_samples_in_roa=num_in_roa,
        num_samples_decreasing=num_decreasing,
        num_samples_verified_roa=num_verified_roa,
        num_samples_total=num_samples_actual,
        mean_delta_V=mean_delta_V,
        mean_delta_V_in_roa=mean_delta_V_in_roa,
        max_delta_V=max_delta_V,
        min_delta_V=min_delta_V,
        std_delta_V=std_delta_V,
        max_violation_in_roa=max_violation_in_roa,
        percent_verified=percent_verified,
        domain_bounds=state_limits,
        method="qmc_halton",
        stability_threshold=stability_threshold,
        discrepancy=discrepancy_val,
    )


def print_lyapunov_difference_metrics(
    metrics: LyapunovDifferenceMetrics, title: Optional[str] = None
):
    """
    Pretty print Lyapunov difference metrics

    Args:
        metrics: LyapunovDifferenceMetrics object
        title: Optional title for the output
    """
    if title:
        print(f"\n{'='*80}")
        print(f"{title:^80}")
        print(f"{'='*80}")
    else:
        print(f"\n{'='*80}")
        print(f"{'Lyapunov Difference Analysis (Discrete-Time)':^80}")
        print(f"{'='*80}")

    print(f"\n{'Configuration':^80}")
    print(f"{'-'*80}")
    print(f"  Method: {metrics.method}")
    print(f"  Total samples: {metrics.num_samples_total:,}")
    print(f"  ROA threshold (ρ): {metrics.rho:.6f}")
    print(f"  Stability threshold: ΔV ≤ {metrics.stability_threshold:.6f}")
    print(f"  Domain volume: {metrics.area_domain:.6f}")
    if metrics.discrepancy is not None:
        print(f"  Discrepancy: {metrics.discrepancy:.6e} (lower = better uniformity)")

    print(f"\n{'Region Classifications':^80}")
    print(f"{'-'*80}")

    # ROA
    print(f"  ROA (V(x) ≤ ρ):")
    print(f"    Volume: {metrics.area_roa:.6f}")
    print(f"    Coverage: {metrics.coverage_roa*100:.2f}%")
    print(f"    Samples: {metrics.num_samples_in_roa:,}")

    # Decreasing region
    print(f"\n  Decreasing region (ΔV ≤ {metrics.stability_threshold}):")
    print(f"    Volume: {metrics.area_decreasing:.6f}")
    print(f"    Coverage: {metrics.coverage_decreasing*100:.2f}%")
    print(f"    Samples: {metrics.num_samples_decreasing:,}")

    # Verified ROA
    print(f"\n  Verified ROA (V ≤ ρ AND ΔV ≤ {metrics.stability_threshold}):")
    print(f"    Volume: {metrics.area_verified_roa:.6f}")
    print(f"    Coverage: {metrics.coverage_verified_roa*100:.2f}%")
    print(f"    Samples: {metrics.num_samples_verified_roa:,}")

    print(f"\n{'Lyapunov Difference Statistics (ΔV)':^80}")
    print(f"{'-'*80}")
    print(f"  All samples:")
    print(f"    Mean: {metrics.mean_delta_V:.6f}")
    print(f"    Std Dev: {metrics.std_delta_V:.6f}")
    print(f"    Min: {metrics.min_delta_V:.6f}")
    print(f"    Max: {metrics.max_delta_V:.6f}")

    if metrics.num_samples_in_roa > 0:
        print(f"\n  Within ROA:")
        print(f"    Mean: {metrics.mean_delta_V_in_roa:.6f}")
        print(f"    Max violation: {metrics.max_violation_in_roa:.6f}")

    print(f"\n{'Verification Status':^80}")
    print(f"{'-'*80}")

    if metrics.num_samples_in_roa > 0:
        print(
            f"  ROA verification: {metrics.percent_verified:.2f}% of ROA has ΔV ≤ {metrics.stability_threshold}"
        )

        if metrics.percent_verified == 100.0:
            print(
                f"  ✓ All ROA points satisfy stability condition (ΔV ≤ {metrics.stability_threshold})"
            )
        else:
            violation_pct = 100.0 - metrics.percent_verified
            print(f"  ✗ {violation_pct:.2f}% of ROA points violate stability condition")
            print(
                f"    ({metrics.num_samples_in_roa - metrics.num_samples_verified_roa:,} samples)"
            )
    else:
        print(f"  ⚠ No samples found in ROA")

    print(f"{'='*80}\n")


def compare_lyapunov_difference_methods(
    lyapunov_nn: torch.nn.Module,
    controller_nn: torch.nn.Module,
    dynamics_fn: Callable,
    state_limits: Tuple[Tuple[float, float], ...],
    rho: float,
    device: str = "cpu",
    num_samples: int = 100000,
    observer_nn: Optional[torch.nn.Module] = None,
    state_indices: Optional[Tuple[int, ...]] = None,
    stability_threshold: float = 0.0,
    compute_discrepancy: bool = False,
) -> Dict[str, LyapunovDifferenceMetrics]:
    """
    Compare Lyapunov difference estimates using different sampling methods

    Args:
        lyapunov_nn: Lyapunov function
        controller_nn: Controller
        dynamics_fn: Discrete dynamics
        state_limits: Domain bounds
        rho: ROA threshold
        device: Computing device
        num_samples: Samples for Monte Carlo and QMC methods
        observer_nn: Optional observer
        state_indices: Which dimensions to analyze
        stability_threshold: Threshold for ΔV
        compute_discrepancy: Whether to compute discrepancy metric

    Returns:
        Dict with metrics from different methods
    """

    results = {}

    # Monte Carlo
    print("Computing Lyapunov difference via Monte Carlo (random)...")
    mc_metrics = compute_lyapunov_difference_metrics_monte_carlo(
        lyapunov_nn,
        controller_nn,
        dynamics_fn,
        state_limits,
        rho,
        num_samples=num_samples,
        device=device,
        observer_nn=observer_nn,
        state_indices=state_indices,
        stability_threshold=stability_threshold,
    )
    results["monte_carlo"] = mc_metrics

    # Sobol QMC
    print("Computing Lyapunov difference via Quasi-Monte Carlo (Sobol)...")
    sobol_metrics = compute_lyapunov_difference_metrics_qmc_sobol(
        lyapunov_nn,
        controller_nn,
        dynamics_fn,
        state_limits,
        rho,
        num_samples=num_samples,
        device=device,
        observer_nn=observer_nn,
        state_indices=state_indices,
        stability_threshold=stability_threshold,
        compute_discrepancy_metric=compute_discrepancy,
    )
    results["qmc_sobol"] = sobol_metrics

    # Halton QMC
    print("Computing Lyapunov difference via Quasi-Monte Carlo (Halton)...")
    halton_metrics = compute_lyapunov_difference_metrics_qmc_halton(
        lyapunov_nn,
        controller_nn,
        dynamics_fn,
        state_limits,
        rho,
        num_samples=num_samples,
        device=device,
        observer_nn=observer_nn,
        state_indices=state_indices,
        stability_threshold=stability_threshold,
        compute_discrepancy_metric=compute_discrepancy,
    )
    results["qmc_halton"] = halton_metrics

    # Print comparison
    print("\n" + "=" * 90)
    print(f"{'Method Comparison':^90}")
    print("=" * 90)
    print(
        f"{'Method':<20} {'ROA Vol':<12} {'Verified Vol':<12} {'% Verified':<12} {'Mean ΔV':<12}"
    )
    print("-" * 90)
    for method_name, metrics in results.items():
        print(
            f"{method_name:<20} {metrics.area_roa:<12.6f} {metrics.area_verified_roa:<12.6f} "
            f"{metrics.percent_verified:<11.2f}% {metrics.mean_delta_V_in_roa:<12.6f}"
        )

    if compute_discrepancy:
        print(f"\n{'Discrepancy (lower = more uniform)':^90}")
        print("-" * 90)
        for method_name, metrics in results.items():
            if metrics.discrepancy is not None:
                print(f"  {method_name:<20}: {metrics.discrepancy:.6e}")

    print("=" * 90 + "\n")

    return results
