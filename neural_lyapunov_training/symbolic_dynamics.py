"""
Symbolic Dynamical Systems Framework

A framework for defining dynamical systems symbolically using SymPy and
automatically generating PyTorch-compatible numerical functions.

Key improvements:
- Better error handling and validation
- Caching for improved performance
- Comprehensive documentation
- Backward compatibility properties
- Type safety improvements
"""

import sympy as sp
import numpy as np
import scipy
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union, Optional, Callable
from enum import Enum
import control
import warnings


# Standard mapping from SymPy functions to PyTorch functions
# CRITICAL: SymPy uses capital letters for some functions (Abs, Min, Max, Pow)
def _torch_min(*args):
    """Handle Min for both scalars and tensors"""
    if len(args) == 2:
        a, b = args
        # Convert both to tensors to handle scalar/tensor combinations
        a_tensor = torch.as_tensor(a) if not isinstance(a, torch.Tensor) else a
        b_tensor = torch.as_tensor(b) if not isinstance(b, torch.Tensor) else b
        return torch.minimum(a_tensor, b_tensor)
    else:
        # Multiple arguments - stack and take min
        tensors = [
            torch.as_tensor(a) if not isinstance(a, torch.Tensor) else a for a in args
        ]
        return torch.min(torch.stack(tensors))


def _torch_max(*args):
    """Handle Max for both scalars and tensors"""
    if len(args) == 2:
        a, b = args
        # Convert both to tensors to handle scalar/tensor combinations
        a_tensor = torch.as_tensor(a) if not isinstance(a, torch.Tensor) else a
        b_tensor = torch.as_tensor(b) if not isinstance(b, torch.Tensor) else b
        return torch.maximum(a_tensor, b_tensor)
    else:
        # Multiple arguments - stack and take max
        tensors = [
            torch.as_tensor(a) if not isinstance(a, torch.Tensor) else a for a in args
        ]
        return torch.max(torch.stack(tensors))
        
def _identity_matrix(*args):
    """Handle ImmutableDenseMatrix - just return the args as tuple"""
    return args if len(args) > 1 else args[0]

SYMPY_TO_TORCH = {
    # Trigonometric
    'sin': torch.sin,
    'cos': torch.cos,
    'tan': torch.tan,
    'asin': torch.asin,
    'acos': torch.acos,
    'atan': torch.atan,
    'atan2': torch.atan2,
    'sinh': torch.sinh,
    'cosh': torch.cosh,
    'tanh': torch.tanh,
    
    # Exponential/Logarithmic
    'exp': torch.exp,
    'log': torch.log,
    'sqrt': torch.sqrt,
    
    # Absolute value and sign - CRITICAL: SymPy uses 'Abs' not 'abs'
    'Abs': torch.abs,
    'abs': torch.abs,  # Include lowercase for safety
    'sign': torch.sign,
    
    # Min/Max - CRITICAL: SymPy uses 'Min' and 'Max' (capital letters)
    # Use helper functions to handle scalar/tensor combinations
    'Min': _torch_min,
    'Max': _torch_max,
    
    # Power - CRITICAL: SymPy uses 'Pow' (capital P)
    'Pow': torch.pow,
    
    # Rounding
    'floor': torch.floor,
    'ceil': torch.ceil,
    'round': torch.round,
    
    # Additional useful functions
    'clip': torch.clamp,
    'minimum': torch.minimum,
    'maximum': torch.maximum,
    
    # Matrix handling - SymPy sometimes uses these
    'ImmutableDenseMatrix': _identity_matrix,
    'MutableDenseMatrix': _identity_matrix,
    'Matrix': _identity_matrix,
}

class IntegrationMethod(Enum):
    """Available numerical integration methods"""

    ExplicitEuler = 1
    MidPoint = 2
    RK4 = 3


class SymbolicDynamicalSystem(ABC, nn.Module):
    """
    Base class for dynamical systems defined symbolically with SymPy.
    Provides automatic generation of numerical functions and linearizations.
    Compatible with the DiscreteTimeSystem interface.

    Attributes:
        state_vars: List of symbolic state variables
        control_vars: List of symbolic control variables
        output_vars: List of symbolic output variables
        parameters: Dict mapping SymPy symbols to numerical values
        order: System order (1=first-order, 2=second-order, etc.)
    """

    def __init__(self):
        super().__init__()
        # To be defined by subclasses
        self.state_vars: List[sp.Symbol] = []
        self.control_vars: List[sp.Symbol] = []
        self.output_vars: List[sp.Symbol] = []
        self.parameters: Dict[sp.Symbol, float] = {}  # Symbols as keys!

        # Symbolic expressions (to be defined)
        self._f_sym: Optional[sp.Matrix] = None  # State dynamics: dx/dt = f(x, u)
        self._h_sym: Optional[sp.Matrix] = None  # Output: y = h(x)

        # System order (1 for first-order, 2 for second-order, etc.)
        self.order: int = 1

        # Cached numerical functions
        self._f_numpy: Optional[Callable] = None
        self._f_torch: Optional[Callable] = None
        self._h_torch: Optional[Callable] = None

        # Cached Jacobians for efficiency
        self._A_sym_cached: Optional[sp.Matrix] = None
        self._B_sym_cached: Optional[sp.Matrix] = None
        self._C_sym_cached: Optional[sp.Matrix] = None

        # Flag to track if system has been properly initialized
        self._initialized: bool = False

        # Performance statistics
        self._perf_stats = {
            "forward_calls": 0,
            "forward_time": 0.0,
            "linearization_calls": 0,
            "linearization_time": 0.0,
        }

    @abstractmethod
    def define_system(self, *args, **kwargs):
        """
        Define the symbolic system. Must set:
        - self.state_vars: List of state symbols
        - self.control_vars: List of control symbols
        - self.output_vars: List of output symbols (optional)
        - self.parameters: Dict with Symbol keys (not strings!)
        - self._f_sym: Symbolic dynamics matrix
        - self._h_sym: Symbolic output matrix (optional)
        - self.order: System order (default: 1)

        CRITICAL: self.parameters must use SymPy Symbol objects as keys!
        Example: {m: 1.0, l: 0.5} NOT {'m': 1.0, 'l': 0.5}

        Args:
            *args, **kwargs: System-specific parameters
        """
        pass

    def _validate_system(self) -> bool:
        """Validate that the system is properly defined"""
        errors = []

        if not self.state_vars:
            errors.append("state_vars is empty")

        if not self.control_vars:
            errors.append("control_vars is empty")

        if self._f_sym is None:
            errors.append("_f_sym is not defined")

        if self.parameters:
            for key in self.parameters.keys():
                if not isinstance(key, sp.Symbol):
                    errors.append(f"Parameter key {key} is not a SymPy Symbol")

        if errors:
            error_msg = "System validation failed:\n" + "\n".join(
                f"  - {e}" for e in errors
            )
            raise ValueError(error_msg)

        self._initialized = True
        return True

    @property
    def nx(self) -> int:
        """Number of states"""
        return len(self.state_vars)

    @property
    def nu(self) -> int:
        """Number of controls"""
        return len(self.control_vars)

    @property
    def ny(self) -> int:
        """Number of outputs"""
        if self.output_vars:
            return len(self.output_vars)
        elif self._h_sym is not None:
            return self._h_sym.shape[0]
        else:
            return self.nx

    @property
    def nq(self) -> int:
        """Number of generalized coordinates (for higher-order systems)"""
        return self.nx // self.order if self.order > 1 else self.nx

    @property
    def x_equilibrium(self) -> torch.Tensor:
        """Equilibrium state (override in subclass if needed)"""
        return torch.zeros(self.nx)

    @property
    def u_equilibrium(self) -> torch.Tensor:
        """Equilibrium control (override in subclass if needed)"""
        return torch.zeros(self.nu)

    def substitute_parameters(
        self, expr: Union[sp.Expr, sp.Matrix]
    ) -> Union[sp.Expr, sp.Matrix]:
        """
        Substitute numerical parameter values into symbolic expression

        Args:
            expr: SymPy expression or matrix

        Returns:
            Expression with parameters substituted
        """
        return expr.subs(self.parameters)

    def _cache_jacobians(self):
        """Cache symbolic Jacobians for improved performance"""
        if self._f_sym is not None and self._A_sym_cached is None:
            self._A_sym_cached = self._f_sym.jacobian(self.state_vars)
            self._B_sym_cached = self._f_sym.jacobian(self.control_vars)

        if self._h_sym is not None and self._C_sym_cached is None:
            self._C_sym_cached = self._h_sym.jacobian(self.state_vars)

    def linearized_dynamics_symbolic(
        self, x_eq: Optional[sp.Matrix] = None, u_eq: Optional[sp.Matrix] = None
    ) -> Tuple[sp.Matrix, sp.Matrix]:
        """
        Compute symbolic linearization A = df/dx, B = df/du

        Args:
            x_eq: Equilibrium state (zeros if None)
            u_eq: Equilibrium control (zeros if None)

        Returns:
            (A, B): Linearized state and control matrices
        """
        if x_eq is None:
            x_eq = sp.Matrix([0] * self.nx)
        if u_eq is None:
            u_eq = sp.Matrix([0] * self.nu)

        # Use cached Jacobians if available
        if self._A_sym_cached is None:
            self._cache_jacobians()

        # Substitute equilibrium point
        subs_dict = dict(
            zip(self.state_vars + self.control_vars, list(x_eq) + list(u_eq))
        )
        A = self._A_sym_cached.subs(subs_dict)
        B = self._B_sym_cached.subs(subs_dict)

        # Substitute parameters
        A = self.substitute_parameters(A)
        B = self.substitute_parameters(B)

        return A, B

    def linearized_observation_symbolic(
        self, x_eq: Optional[sp.Matrix] = None
    ) -> sp.Matrix:
        """
        Compute symbolic linearization C = dh/dx

        Args:
            x_eq: Equilibrium state (zeros if None)

        Returns:
            C: Linearized output matrix
        """
        if self._h_sym is None:
            return sp.eye(self.nx)

        if x_eq is None:
            x_eq = sp.Matrix([0] * self.nx)

        # Use cached Jacobian if available
        if self._C_sym_cached is None:
            self._cache_jacobians()

        subs_dict = dict(zip(self.state_vars, list(x_eq)))
        C = self._C_sym_cached.subs(subs_dict)
        C = self.substitute_parameters(C)

        return C

    def generate_numpy_function(self) -> Callable:
        """
        Generate lambdified NumPy function for dynamics

        Returns:
            Callable function compatible with NumPy
        """
        f_with_params = self.substitute_parameters(self._f_sym)
        all_vars = self.state_vars + self.control_vars
        self._f_numpy = sp.lambdify(all_vars, f_with_params, modules="numpy")
        return self._f_numpy

    def generate_torch_function(self) -> Callable:
        """
        Generate PyTorch-compatible function for dynamics using code generation

        This method generates Python code as a string and executes it to create
        a function that uses PyTorch operations. This approach avoids issues with
        SymPy's lambdify and PyTorch tensor operations.

        Returns:
            Callable function compatible with PyTorch tensors
        """
        f_with_params = self.substitute_parameters(self._f_sym)
        f_with_params = sp.simplify(f_with_params)

        all_vars = self.state_vars + self.control_vars

        from sympy.printing.pycode import pycode

        # Generate function signature
        func_code_lines = [
            "def dynamics_func(" + ", ".join([str(v) for v in all_vars]) + "):",
            "    import torch",
        ]

        # Generate code for each output component
        results = []
        for i, expr in enumerate(f_with_params):
            code = pycode(expr)
            # Replace module prefixes with torch
            code = code.replace("numpy.", "torch.")
            code = code.replace("math.", "torch.")

            var_name = f"result_{i}"
            func_code_lines.append(f"    {var_name} = {code}")
            results.append(var_name)

        # Return tuple of results
        func_code_lines.append(f"    return ({', '.join(results)},)")
        func_code = "\n".join(func_code_lines)

        # Execute the generated code
        namespace = {"torch": torch}
        exec(func_code, namespace)
        base_func = namespace["dynamics_func"]

        # Wrap to ensure proper tensor handling
        def wrapped_func(*args):
            result = base_func(*args)

            if isinstance(result, (list, tuple)):
                return torch.stack(list(result), dim=-1)
            elif isinstance(result, torch.Tensor):
                if len(result.shape) == 1:
                    return result.unsqueeze(-1)
                return result
            else:
                return torch.tensor([result]).unsqueeze(0)

        self._f_torch = wrapped_func
        return self._f_torch

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Evaluate dynamics: dx/dt = f(x, u)

        Args:
            x: State tensor (batch_size, nx) or (nx,)
            u: Control tensor (batch_size, nu) or (nu,)

        Returns:
            State derivative tensor (same shape as input)

        Raises:
            ValueError: If input dimensions don't match system dimensions
        """
        import time

        start_time = time.time()

        # Input validation - handle edge cases
        if len(x.shape) == 0 or len(u.shape) == 0:
            raise ValueError("Input tensors must be at least 1D")

        # Check dimensions only if tensors are at least 1D
        if len(x.shape) >= 1 and x.shape[-1] != self.nx:
            raise ValueError(f"Expected state dimension {self.nx}, got {x.shape[-1]}")
        if len(u.shape) >= 1 and u.shape[-1] != self.nu:
            raise ValueError(f"Expected control dimension {self.nu}, got {u.shape[-1]}")

        if self._f_torch is None:
            self.generate_torch_function()

        # Handle batched vs single evaluation
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            u = u.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Prepare arguments
        x_list = [x[:, i] for i in range(self.nx)]
        u_list = [u[:, i] for i in range(self.nu)]
        all_args = x_list + u_list

        # Call generated function
        result = self._f_torch(*all_args)

        if squeeze_output:
            result = result.squeeze(0)

        # Update performance stats
        self._perf_stats["forward_calls"] += 1
        self._perf_stats["forward_time"] += time.time() - start_time

        return result

    def linearized_dynamics(
        self, x: torch.Tensor, u: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Numerical evaluation of linearized dynamics at point (x, u)

        Args:
            x: State tensor
            u: Control tensor

        Returns:
            (A, B): Linearized dynamics matrices as PyTorch tensors
        """
        import time

        start_time = time.time()

        # Handle batched input
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            u = u.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size = x.shape[0]
        device = x.device
        dtype = x.dtype

        # Allocate output tensors
        A_batch = torch.zeros(batch_size, self.nx, self.nx, dtype=dtype, device=device)
        B_batch = torch.zeros(batch_size, self.nx, self.nu, dtype=dtype, device=device)

        # Evaluate for each sample
        for i in range(batch_size):
            # Convert to numpy - handle both 1D and potential 0D cases
            x_i = x[i] if batch_size > 1 else x.squeeze(0)
            u_i = u[i] if batch_size > 1 else u.squeeze(0)

            x_np = x_i.detach().cpu().numpy()
            u_np = u_i.detach().cpu().numpy()

            # Ensure arrays are at least 1D for SymPy Matrix
            x_np = np.atleast_1d(x_np)
            u_np = np.atleast_1d(u_np)

            A_sym, B_sym = self.linearized_dynamics_symbolic(
                sp.Matrix(x_np), sp.Matrix(u_np)
            )
            A_batch[i] = torch.tensor(
                np.array(A_sym, dtype=np.float64), dtype=dtype, device=device
            )
            B_batch[i] = torch.tensor(
                np.array(B_sym, dtype=np.float64), dtype=dtype, device=device
            )

        if squeeze_output:
            A_batch = A_batch.squeeze(0)
            B_batch = B_batch.squeeze(0)

        # Update performance stats
        self._perf_stats["linearization_calls"] += 1
        self._perf_stats["linearization_time"] += time.time() - start_time

        return A_batch, B_batch

    def linearized_observation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Numerical evaluation of output linearization C = dh/dx

        Args:
            x: State tensor

        Returns:
            C: Linearized observation matrix as PyTorch tensor
        """
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size = x.shape[0]
        device = x.device
        dtype = x.dtype

        C_batch = torch.zeros(batch_size, self.ny, self.nx, dtype=dtype, device=device)

        for i in range(batch_size):
            # Handle indexing properly
            x_i = x[i] if batch_size > 1 else x.squeeze(0)
            x_np = x_i.detach().cpu().numpy()

            # Ensure at least 1D
            x_np = np.atleast_1d(x_np)

            C_sym = self.linearized_observation_symbolic(sp.Matrix(x_np))
            C_batch[i] = torch.tensor(
                np.array(C_sym, dtype=np.float64), dtype=dtype, device=device
            )

        if squeeze_output:
            C_batch = C_batch.squeeze(0)

        return C_batch

    def h(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate output equation: y = h(x)
        
        Args:
            x: State tensor
            
        Returns:
            Output tensor
        """
        if self._h_sym is None:
            return x
        
        # Generate torch function for h if not cached
        if self._h_torch is None:
            h_with_params = self.substitute_parameters(self._h_sym)
            # Use ONLY our custom namespace (don't add 'torch' as fallback to avoid conflicts)
            self._h_torch = sp.lambdify(
                self.state_vars, h_with_params, 
                modules=[SYMPY_TO_TORCH]
            )
        
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        x_list = [x[:, i] for i in range(self.nx)]
        result = self._h_torch(*x_list)
        
        # Handle various return types from lambdify
        def flatten_result(r):
            """Recursively flatten nested lists/tuples to get tensors"""
            if isinstance(r, torch.Tensor):
                return [r]
            elif isinstance(r, (list, tuple)):
                flat = []
                for item in r:
                    flat.extend(flatten_result(item))
                return flat
            else:
                # Scalar - convert to tensor
                return [torch.as_tensor(r)]
        
        if isinstance(result, (list, tuple)):
            flat_tensors = flatten_result(result)
            result = torch.stack(flat_tensors, dim=-1)
        elif not isinstance(result, torch.Tensor):
            result = torch.as_tensor(result).unsqueeze(-1)
        
        if squeeze_output:
            result = result.squeeze(0)
        
        return result

    def print_equations(self, simplify: bool = True):
        """
        Print symbolic equations in human-readable format

        Args:
            simplify: Whether to simplify expressions before printing
        """
        print("=" * 70)
        print(f"{self.__class__.__name__}")
        print("=" * 70)
        print(f"State Variables: {self.state_vars}")
        print(f"Control Variables: {self.control_vars}")
        print(f"System Order: {self.order}")
        print(f"Dimensions: nx={self.nx}, nu={self.nu}, ny={self.ny}")

        print("\nDynamics: dx/dt = f(x, u)")
        for i, (var, expr) in enumerate(zip(self.state_vars, self._f_sym)):
            expr_sub = self.substitute_parameters(expr)
            if simplify:
                expr_sub = sp.simplify(expr_sub)
            print(f"  d{var}/dt = {expr_sub}")

        if self._h_sym is not None:
            print("\nOutput: y = h(x)")
            for i, expr in enumerate(self._h_sym):
                expr_sub = self.substitute_parameters(expr)
                if simplify:
                    expr_sub = sp.simplify(expr_sub)
                print(f"  y[{i}] = {expr_sub}")

        print("=" * 70)

    def check_equilibrium(
        self, x_eq: torch.Tensor, u_eq: torch.Tensor, tol: float = 1e-6
    ) -> Tuple[bool, float]:
        """
        Check if (x_eq, u_eq) is an equilibrium point

        Args:
            x_eq: Candidate equilibrium state
            u_eq: Candidate equilibrium control
            tol: Tolerance for considering derivative as zero

        Returns:
            (is_equilibrium, max_derivative): Boolean and max derivative magnitude
        """
        with torch.no_grad():
            dx = self.forward(
                x_eq.unsqueeze(0) if len(x_eq.shape) == 1 else x_eq,
                u_eq.unsqueeze(0) if len(u_eq.shape) == 1 else u_eq,
            )
            max_deriv = torch.abs(dx).max().item()
            is_eq = max_deriv < tol
        return is_eq, max_deriv

    def eigenvalues_at_equilibrium(self) -> np.ndarray:
        """
        Compute eigenvalues of linearization at equilibrium

        Returns:
            Eigenvalues as complex numpy array
        """
        x_eq = self.x_equilibrium.unsqueeze(0)
        u_eq = self.u_equilibrium.unsqueeze(0)
        A, _ = self.linearized_dynamics(x_eq, u_eq)
        A_np = A.squeeze().detach().cpu().numpy()
        eigenvalues = np.linalg.eigvals(A_np)
        return eigenvalues

    def is_stable_equilibrium(self, discrete_time: bool = False) -> bool:
        """
        Check if equilibrium is stable based on eigenvalues

        Args:
            discrete_time: If True, check |λ| < 1; if False, check Re(λ) < 0

        Returns:
            True if equilibrium is stable
        """
        eigs = self.eigenvalues_at_equilibrium()
        if discrete_time:
            return bool(np.all(np.abs(eigs) < 1.0))
        else:
            return bool(np.all(np.real(eigs) < 0.0))

    def clone(self):
        """Create a deep copy of the system"""
        import copy

        return copy.deepcopy(self)

    def to_device(self, device: Union[str, torch.device]):
        """
        Move system to specified device

        Args:
            device: Target device ('cpu', 'cuda', or torch.device)

        Returns:
            Self for chaining
        """
        if isinstance(device, str):
            device = torch.device(device)

        # Move equilibrium points
        if hasattr(self, "_x_eq_cached"):
            self._x_eq_cached = self._x_eq_cached.to(device)
        if hasattr(self, "_u_eq_cached"):
            self._u_eq_cached = self._u_eq_cached.to(device)

        return self

    def verify_jacobians(
        self, x: torch.Tensor, u: torch.Tensor, epsilon: float = 1e-5, tol: float = 1e-3
    ) -> Dict[str, Union[bool, float]]:
        """
        Verify symbolic Jacobians against numerical finite differences

        Args:
            x: State at which to verify (can be 1D or 2D)
            u: Control at which to verify (can be 1D or 2D)
            epsilon: Not used (kept for API compatibility)
            tol: Tolerance for considering Jacobians equal

        Returns:
            Dict with 'A_match', 'B_match' booleans and error magnitudes
        """
        # Ensure proper 2D shape (batch_size=1, dim)
        x_2d = x.reshape(1, -1) if len(x.shape) <= 1 else x
        u_2d = u.reshape(1, -1) if len(u.shape) <= 1 else u

        # Clone for autograd - keep 2D shape
        x_grad = x_2d.clone().requires_grad_(True)
        u_grad = u_2d.clone().requires_grad_(True)

        # Compute symbolic Jacobians
        A_sym, B_sym = self.linearized_dynamics(x_2d.detach(), u_2d.detach())

        # Ensure 3D shape for batch processing
        if len(A_sym.shape) == 2:
            A_sym = A_sym.unsqueeze(0)
            B_sym = B_sym.unsqueeze(0)

        # Compute numerical Jacobians via autograd
        fx = self.forward(x_grad, u_grad)  # fx shape: (1, nx)
        A_num = torch.zeros_like(A_sym)
        B_num = torch.zeros_like(B_sym)

        for i in range(self.nx):
            if fx[0, i].requires_grad:
                grad_x = torch.autograd.grad(
                    fx[0, i], x_grad, retain_graph=True, create_graph=False
                )[0]
                grad_u = torch.autograd.grad(
                    fx[0, i], u_grad, retain_graph=True, create_graph=False
                )[0]
                A_num[0, i] = grad_x[0]  # grad_x shape: (1, nx)
                B_num[0, i] = grad_u[0]  # grad_u shape: (1, nu)

        A_error = (A_sym - A_num).abs().max().item()
        B_error = (B_sym - B_num).abs().max().item()
        A_match = A_error < tol
        B_match = B_error < tol

        return {
            "A_match": bool(A_match),
            "B_match": bool(B_match),
            "A_error": float(A_error),
            "B_error": float(B_error),
        }

    def get_performance_stats(self) -> Dict[str, float]:
        """
        Get performance statistics

        Returns:
            Dict with timing and call count statistics
        """
        return {
            **self._perf_stats,
            "avg_forward_time": self._perf_stats["forward_time"]
            / max(1, self._perf_stats["forward_calls"]),
            "avg_linearization_time": self._perf_stats["linearization_time"]
            / max(1, self._perf_stats["linearization_calls"]),
        }

    def reset_performance_stats(self):
        """Reset performance counters"""
        for key in self._perf_stats:
            self._perf_stats[key] = 0.0 if "time" in key else 0

    def check_numerical_stability(
        self, x: torch.Tensor, u: torch.Tensor
    ) -> Dict[str, Union[bool, float]]:
        """
        Check for numerical issues (NaN, Inf, extreme values)

        Args:
            x: State to check (any shape)
            u: Control to check (any shape)

        Returns:
            Dict with stability indicators
        """
        # Ensure proper shape
        x_2d = x.reshape(1, -1) if len(x.shape) <= 1 else x
        u_2d = u.reshape(1, -1) if len(u.shape) <= 1 else u

        with torch.no_grad():
            dx = self.forward(x_2d, u_2d)
            return {
                "has_nan": bool(torch.isnan(dx).any().item()),
                "has_inf": bool(torch.isinf(dx).any().item()),
                "max_derivative": float(dx.abs().max().item()),
                "is_stable": bool(not (torch.isnan(dx).any() or torch.isinf(dx).any())),
            }

    def __repr__(self) -> str:
        """String representation for debugging"""
        return (
            f"{self.__class__.__name__}("
            f"nx={self.nx}, nu={self.nu}, ny={self.ny}, order={self.order})"
        )

    def __str__(self) -> str:
        """Human-readable string representation"""
        return (
            f"{self.__class__.__name__}(nx={self.nx}, nu={self.nu}, order={self.order})"
        )

    def save_config(self, filename: str):
        """
        Save system configuration to file

        Args:
            filename: Path to save configuration (supports .json, .yaml, .pt)
        """
        import json

        config = {
            "class_name": self.__class__.__name__,
            "parameters": {str(k): float(v) for k, v in self.parameters.items()},
            "order": self.order,
            "nx": self.nx,
            "nu": self.nu,
            "ny": self.ny,
        }

        if filename.endswith(".json"):
            with open(filename, "w") as f:
                json.dump(config, f, indent=2)
        elif filename.endswith(".pt"):
            torch.save(config, filename)
        else:
            raise ValueError(f"Unsupported file format: {filename}. Use .json or .pt")

        print(f"Configuration saved to {filename}")

    def get_config_dict(self) -> Dict:
        """
        Get configuration as dictionary

        Returns:
            Dict with system configuration
        """
        return {
            "class_name": self.__class__.__name__,
            "parameters": {str(k): float(v) for k, v in self.parameters.items()},
            "order": self.order,
            "nx": self.nx,
            "nu": self.nu,
            "ny": self.ny,
        }

    def lqr_control(self, Q: np.ndarray, R: np.ndarray, 
                x_eq: Optional[torch.Tensor] = None,
                u_eq: Optional[torch.Tensor] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute LQR control gain for continuous-time system
        
        The control law is: u = K @ (x - x_eq) + u_eq
        
        Args:
            Q: State cost matrix (nx, nx)
            R: Control cost matrix (nu, nu) or scalar for single input
            x_eq: Equilibrium state (uses self.x_equilibrium if None)
            u_eq: Equilibrium control (uses self.u_equilibrium if None)
        
        Returns:
            (K, S): Control gain matrix and solution to Riccati equation
        """
        if x_eq is None:
            x_eq = self.x_equilibrium
        if u_eq is None:
            u_eq = self.u_equilibrium
        
        # Ensure proper shape
        if len(x_eq.shape) == 1:
            x_eq = x_eq.unsqueeze(0)
        if len(u_eq.shape) == 1:
            u_eq = u_eq.unsqueeze(0)
        
        # Get linearized dynamics at equilibrium
        A, B = self.linearized_dynamics(x_eq, u_eq)
        A = A.squeeze().detach().cpu().numpy()
        B = B.squeeze().detach().cpu().numpy()
        
        # Ensure B is 2D (nx, nu)
        if B.ndim == 1:
            B = B.reshape(-1, 1)
        
        # Ensure R is 2D
        if isinstance(R, (int, float)):
            R = np.array([[R]])
        elif R.ndim == 1:
            R = np.diag(R)
        
        # Validate dimensions
        nx, nu = B.shape
        if A.shape != (nx, nx):
            raise ValueError(f"A must be ({nx}, {nx}), got {A.shape}")
        if Q.shape != (nx, nx):
            raise ValueError(f"Q must be ({nx}, {nx}), got {Q.shape}")
        if R.shape != (nu, nu):
            raise ValueError(f"R must be ({nu}, {nu}), got {R.shape}")
        
        # Solve continuous-time algebraic Riccati equation
        S = scipy.linalg.solve_continuous_are(A, B, Q, R)
        
        # Compute optimal gain
        K = -np.linalg.solve(R, B.T @ S)
        
        return K, S


    def kalman_gain(self, Q_process: Optional[np.ndarray] = None,
                    R_measurement: Optional[np.ndarray] = None,
                    x_eq: Optional[torch.Tensor] = None,
                    u_eq: Optional[torch.Tensor] = None) -> np.ndarray:
        """
        Compute Kalman filter gain for continuous-time system
        
        Observer dynamics: dx̂/dt = f(x̂, u) + L(y - h(x̂))
        
        Args:
            Q_process: Process noise covariance (nx, nx)
            R_measurement: Measurement noise covariance (ny, ny) or scalar
            x_eq: Equilibrium state
            u_eq: Equilibrium control
        
        Returns:
            L: Kalman gain matrix (nx, ny)
        """
        if Q_process is None:
            Q_process = np.eye(self.nx) * 1e-3
        if R_measurement is None:
            R_measurement = np.eye(self.ny) * 1e-3
        if x_eq is None:
            x_eq = self.x_equilibrium
        if u_eq is None:
            u_eq = self.u_equilibrium
        
        # Ensure proper shape
        if len(x_eq.shape) == 1:
            x_eq = x_eq.unsqueeze(0)
        
        # Get linearized dynamics
        A, _ = self.linearized_dynamics(x_eq, u_eq if len(u_eq.shape) > 1 else u_eq.unsqueeze(0))
        A = A.squeeze().detach().cpu().numpy()
        
        C = self.linearized_observation(x_eq)
        C = C.squeeze().detach().cpu().numpy()
        
        # Ensure C is 2D (ny, nx)
        if C.ndim == 1:
            C = C.reshape(1, -1)
        
        # Ensure R_measurement is 2D
        if isinstance(R_measurement, (int, float)):
            R_measurement = np.array([[R_measurement]])
        elif R_measurement.ndim == 1:
            R_measurement = np.diag(R_measurement)
        
        # Validate dimensions
        nx = A.shape[0]
        ny = C.shape[0]
        
        if A.shape != (nx, nx):
            raise ValueError(f"A must be square, got {A.shape}")
        if C.shape[1] != nx:
            raise ValueError(f"C must have {nx} columns, got {C.shape}")
        if Q_process.shape != (nx, nx):
            raise ValueError(f"Q_process must be ({nx}, {nx}), got {Q_process.shape}")
        if R_measurement.shape != (ny, ny):
            raise ValueError(f"R_measurement must be ({ny}, {ny}), got {R_measurement.shape}")
        
        # Solve continuous-time algebraic Riccati equation (dual problem)
        P = scipy.linalg.solve_continuous_are(A.T, C.T, Q_process, R_measurement)
        
        # Compute Kalman gain
        L = P @ C.T @ np.linalg.inv(R_measurement)
        
        return L


    def lqg_control(self, Q_lqr: np.ndarray, R_lqr: np.ndarray,
                    Q_process: Optional[np.ndarray] = None,
                    R_measurement: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute LQG controller (LQR + Kalman filter)
        
        Returns both the control gain K and observer gain L for output feedback control:
        - dx̂/dt = f(x̂, u) + L(y - h(x̂))
        - u = K @ (x̂ - x_eq) + u_eq
        
        Args:
            Q_lqr: State cost for LQR
            R_lqr: Control cost for LQR
            Q_process: Process noise covariance
            R_measurement: Measurement noise covariance
        
        Returns:
            (K, L): Control gain and observer gain
        """
        K, _ = self.lqr_control(Q_lqr, R_lqr)
        L = self.kalman_gain(Q_process, R_measurement)
        return K, L


    def lqg_closed_loop_matrix(self, K: np.ndarray, L: np.ndarray) -> np.ndarray:
        """
        Compute closed-loop system matrix for LQG control
        
        Augmented state: [x, x̂] where x̂ is the estimate
        Closed-loop dynamics:
            dx/dt = f(x, K(x̂ - x_eq) + u_eq)
            dx̂/dt = f(x̂, K(x̂ - x_eq) + u_eq) + L(h(x) - h(x̂))
        
        Linearized around equilibrium:
            d[x, x̂]/dt = A_cl [x, x̂]
        
        Args:
            K: LQR control gain (nu, nx)
            L: Kalman filter gain (nx, ny)
        
        Returns:
            A_cl: Closed-loop system matrix (2*nx, 2*nx)
        """
        x_eq = self.x_equilibrium.unsqueeze(0)
        u_eq = self.u_equilibrium.unsqueeze(0)
        
        A, B = self.linearized_dynamics(x_eq, u_eq)
        A = A.squeeze().detach().cpu().numpy()
        B = B.squeeze().detach().cpu().numpy()
        
        # Ensure B is 2D
        if B.ndim == 1:
            B = B.reshape(-1, 1)
        
        C = self.linearized_observation(x_eq).squeeze().detach().cpu().numpy()
        
        # Ensure C is 2D
        if C.ndim == 1:
            C = C.reshape(1, -1)
        
        # Ensure K is 2D (nu, nx)
        if K.ndim == 1:
            K = K.reshape(1, -1)
        
        # Ensure L is 2D (nx, ny)
        if L.ndim == 1:
            L = L.reshape(-1, 1)
        
        # Closed-loop system: [x, x̂]
        # dx/dt = Ax + B K x̂
        # dx̂/dt = A x̂ + B K x̂ + L(Cx - C x̂) = (A + B K - L C) x̂ + L C x
        A_cl = np.vstack([
            np.hstack([A + B @ K, -B @ K]),              # dx/dt
            np.hstack([L @ C, A + B @ K - L @ C])        # dx̂/dt
        ])
        
        # Clean up near-zero entries
        A_cl[np.abs(A_cl) <= 1e-6] = 0
        
        return A_cl


class GenericDiscreteTimeSystem(nn.Module):
    """
    Generic discrete-time system for arbitrary order continuous systems.

    Automatically handles first-order, second-order, and higher-order systems
    using various numerical integration methods.

    Attributes:
        continuous_time_system: The underlying continuous-time system
        dt: Integration time step
        order: System order (inherited from continuous system)
        integration_method: Method for integrating derivatives
        position_integration: Method for integrating positions (order > 1)
    """

    def __init__(
        self,
        continuous_time_system: SymbolicDynamicalSystem,
        dt: float,
        integration_method: IntegrationMethod = IntegrationMethod.ExplicitEuler,
        position_integration: Optional[IntegrationMethod] = None,
    ):
        """
        Initialize discrete-time system wrapper

        Args:
            continuous_time_system: Symbolic dynamical system
            dt: Time step for discretization
            integration_method: Method for velocity/derivative integration
            position_integration: Method for position integration (order > 1)
        """
        super().__init__()

        # Validate continuous system
        if (
            not hasattr(continuous_time_system, "_initialized")
            or not continuous_time_system._initialized
        ):
            continuous_time_system._validate_system()

        self.continuous_time_system = continuous_time_system
        self.nx = continuous_time_system.nx
        self.nu = continuous_time_system.nu
        self.dt = float(dt)
        self.order = continuous_time_system.order
        self.integration_method = integration_method
        self.position_integration = position_integration or integration_method
        self.Ix = torch.eye(self.nx)

        if dt <= 0:
            raise ValueError(f"Time step dt must be positive, got {dt}")

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Compute next state: x_next = discrete_dynamics(x, u)

        Args:
            x: Current state
            u: Control input

        Returns:
            Next state after one time step
        """
        if self.order == 1:
            return self._integrate_first_order(x, u)
        elif self.order == 2:
            return self._integrate_second_order(x, u)
        else:
            return self._integrate_arbitrary_order(x, u)

    def __call__(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Make the system callable like a function"""
        return self.forward(x, u)

    def simulate(
        self, x0: torch.Tensor, u_sequence: torch.Tensor, return_all: bool = True
    ) -> Union[torch.Tensor, torch.Tensor]:
        """
        Simulate trajectory from initial state

        Args:
            x0: Initial state (nx,) or (batch, nx)
            u_sequence: Control sequence (T, nu) or (batch, T, nu)
            return_all: If True, return all states; if False, only final state

        Returns:
            Trajectory: (T+1, nx) or (batch, T+1, nx) if return_all=True
                       (nx,) or (batch, nx) if return_all=False
        """
        # Handle dimensionality
        if len(x0.shape) == 1:
            x0 = x0.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False

        if len(u_sequence.shape) == 2:
            u_sequence = u_sequence.unsqueeze(0)

        batch_size, T, _ = u_sequence.shape

        if return_all:
            trajectory = [x0]

        x = x0
        for t in range(T):
            x = self.forward(x, u_sequence[:, t, :])
            if return_all:
                trajectory.append(x)

        if return_all:
            result = torch.stack(trajectory, dim=1)  # (batch, T+1, nx)
            if squeeze_batch:
                result = result.squeeze(0)
            return result
        else:
            if squeeze_batch:
                x = x.squeeze(0)
            return x

    def _integrate_first_order(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Integrate first-order system: dx/dt = f(x, u)"""
        xdot = self.continuous_time_system.forward(x, u)

        if self.integration_method == IntegrationMethod.ExplicitEuler:
            x_next = x + xdot * self.dt
        elif self.integration_method == IntegrationMethod.MidPoint:
            k1 = xdot
            x_mid = x + 0.5 * self.dt * k1
            k2 = self.continuous_time_system.forward(x_mid, u)
            x_next = x + self.dt * k2
        elif self.integration_method == IntegrationMethod.RK4:
            k1 = xdot
            k2 = self.continuous_time_system.forward(x + 0.5 * self.dt * k1, u)
            k3 = self.continuous_time_system.forward(x + 0.5 * self.dt * k2, u)
            k4 = self.continuous_time_system.forward(x + self.dt * k3, u)
            x_next = x + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            raise NotImplementedError(
                f"Integration method {self.integration_method} not implemented"
            )

        return x_next

    def _integrate_second_order(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Integrate second-order system: x = [q, qdot], qddot = f(x, u)"""
        nq = self.continuous_time_system.nq
        q = x[:, :nq]
        qdot = x[:, nq:]

        qddot = self.continuous_time_system.forward(x, u)

        # Validate acceleration shape
        if qddot.shape[1] != nq:
            if qddot.shape[1] == 1 and nq == 1:
                pass  # Correct
            else:
                raise ValueError(f"Expected qddot shape (*, {nq}), got {qddot.shape}")

        # Integrate velocity
        if self.integration_method == IntegrationMethod.ExplicitEuler:
            qdot_next = qdot + qddot * self.dt
        elif self.integration_method == IntegrationMethod.MidPoint:
            qdot_mid = qdot + 0.5 * self.dt * qddot
            x_mid = torch.cat([q, qdot_mid], dim=1)
            qddot_mid = self.continuous_time_system.forward(x_mid, u)
            qdot_next = qdot + self.dt * qddot_mid
        elif self.integration_method == IntegrationMethod.RK4:
            k1 = qddot
            x_mid = torch.cat([q, qdot + 0.5 * self.dt * k1], dim=1)
            k2 = self.continuous_time_system.forward(x_mid, u)
            x_mid = torch.cat([q, qdot + 0.5 * self.dt * k2], dim=1)
            k3 = self.continuous_time_system.forward(x_mid, u)
            x_mid = torch.cat([q, qdot + self.dt * k3], dim=1)
            k4 = self.continuous_time_system.forward(x_mid, u)
            qdot_next = qdot + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            raise NotImplementedError(
                f"Integration method {self.integration_method} not implemented for 2nd order"
            )

        # Integrate position
        if self.position_integration == IntegrationMethod.ExplicitEuler:
            q_next = q + qdot * self.dt
        elif self.position_integration == IntegrationMethod.MidPoint:
            q_next = q + (qdot_next + qdot) / 2 * self.dt
        elif self.position_integration == IntegrationMethod.RK4:
            # Use midpoint for now (fully consistent RK4 would need intermediate velocities)
            q_next = q + (qdot_next + qdot) / 2 * self.dt
        else:
            raise NotImplementedError(
                f"Position integration {self.position_integration} not implemented"
            )

        return torch.cat([q_next, qdot_next], dim=1)

    def _integrate_arbitrary_order(
        self, x: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        """
        Integrate arbitrary order system: x = [q, q', ..., q^(n-1)], q^(n) = f(x, u)
        """
        order = self.order
        nq = self.nx // order

        # Split state into derivative levels
        derivatives = [x[:, i * nq : (i + 1) * nq] for i in range(order)]
        highest_deriv = self.continuous_time_system.forward(x, u)

        derivatives_next = []

        if self.integration_method == IntegrationMethod.ExplicitEuler:
            for i in range(order - 1):
                derivatives_next.append(derivatives[i] + derivatives[i + 1] * self.dt)
            derivatives_next.append(derivatives[-1] + highest_deriv * self.dt)

        elif self.integration_method == IntegrationMethod.MidPoint:
            for i in range(order - 1):
                derivatives_next.append(derivatives[i] + self.dt * derivatives[i + 1])

            q_highest_mid = derivatives[-1] + 0.5 * self.dt * highest_deriv
            x_mid = torch.cat(derivatives[:-1] + [q_highest_mid], dim=1)
            highest_deriv_mid = self.continuous_time_system.forward(x_mid, u)
            derivatives_next.append(derivatives[-1] + self.dt * highest_deriv_mid)

        elif self.integration_method == IntegrationMethod.RK4:
            # Stage 1
            k1_derivs = derivatives[1:] + [highest_deriv]

            # Stage 2
            x_stage2 = [
                derivatives[i] + 0.5 * self.dt * k1_derivs[i] for i in range(order)
            ]
            x_mid_2 = torch.cat(x_stage2, dim=1)
            highest_deriv_2 = self.continuous_time_system.forward(x_mid_2, u)
            k2_derivs = x_stage2[1:] + [highest_deriv_2]

            # Stage 3
            x_stage3 = [
                derivatives[i] + 0.5 * self.dt * k2_derivs[i] for i in range(order)
            ]
            x_mid_3 = torch.cat(x_stage3, dim=1)
            highest_deriv_3 = self.continuous_time_system.forward(x_mid_3, u)
            k3_derivs = x_stage3[1:] + [highest_deriv_3]

            # Stage 4
            x_stage4 = [derivatives[i] + self.dt * k3_derivs[i] for i in range(order)]
            x_end = torch.cat(x_stage4, dim=1)
            highest_deriv_4 = self.continuous_time_system.forward(x_end, u)
            k4_derivs = x_stage4[1:] + [highest_deriv_4]

            # Combine
            for i in range(order):
                weighted = (
                    k1_derivs[i] + 2 * k2_derivs[i] + 2 * k3_derivs[i] + k4_derivs[i]
                ) / 6.0
                derivatives_next.append(derivatives[i] + self.dt * weighted)

        else:
            raise NotImplementedError(
                f"Integration method {self.integration_method} not implemented for order {order}"
            )

        return torch.cat(derivatives_next, dim=1)

    def linearized_dynamics(
        self, x: torch.Tensor, u: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute linearized discrete dynamics using Euler approximation

        Returns:
            (Ad, Bd): Discrete-time linearized dynamics
        """
        Ac, Bc = self.continuous_time_system.linearized_dynamics(x, u)
        Ad = self.dt * Ac + self.Ix.to(x.device)
        Bd = self.dt * Bc
        return Ad, Bd

    def linearized_observation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute linearized observation matrix C = dh/dx
        
        For discrete-time systems, the observation is the same as continuous-time
        since h(x) doesn't depend on the discretization. The observation is with respect to
        state x and not time t.
        
        Args:
            x: State tensor (batch, nx) or (nx,)
        
        Returns:
            C: Observation Jacobian (batch, ny, nx) or (ny, nx)
        """
        return self.continuous_time_system.linearized_observation(x)
    
    def h(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate output equation: y = h(x)
        
        For discrete-time systems, the observation is the same as continuous-time
        since h(x) doesn't depend on the discretization. The observation is with respect to
        state x and not time t.
        
        Args:
            x: State tensor
        
        Returns:
            Output tensor
        """
        return self.continuous_time_system.h(x)

    @property
    def x_equilibrium(self) -> torch.Tensor:
        return self.continuous_time_system.x_equilibrium

    @property
    def u_equilibrium(self) -> torch.Tensor:
        return self.continuous_time_system.u_equilibrium

    def __repr__(self) -> str:
        return (
            f"GenericDiscreteTimeSystem({self.continuous_time_system.__class__.__name__}, "
            f"dt={self.dt}, method={self.integration_method.name})"
        )

    def __str__(self) -> str:
        """Human-readable string representation"""
        return (
            f"Discrete {self.continuous_time_system.__class__.__name__} "
            f"(dt={self.dt:.4f}, {self.integration_method.name})"
        )

    def dlqr_control(self, Q: np.ndarray, R: np.ndarray,
                 x_eq: Optional[torch.Tensor] = None,
                 u_eq: Optional[torch.Tensor] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute discrete-time LQR control gain
        
        The control law is: u = K @ (x - x_eq) + u_eq
        
        Args:
            Q: State cost matrix (nx, nx)
            R: Control cost matrix (nu, nu) or scalar for single input
            x_eq: Equilibrium state
            u_eq: Equilibrium control
        
        Returns:
            (K, S): Control gain matrix and solution to discrete Riccati equation
        """
        if x_eq is None:
            x_eq = self.x_equilibrium
        if u_eq is None:
            u_eq = self.u_equilibrium
        
        # Ensure proper shape
        if len(x_eq.shape) == 1:
            x_eq = x_eq.unsqueeze(0)
        if len(u_eq.shape) == 1:
            u_eq = u_eq.unsqueeze(0)
        
        # Get discrete linearized dynamics at equilibrium
        Ad, Bd = self.linearized_dynamics(x_eq, u_eq)
        Ad = Ad.squeeze().detach().cpu().numpy()
        Bd = Bd.squeeze().detach().cpu().numpy()
        
        # Ensure Bd is 2D
        if Bd.ndim == 1:
            Bd = Bd.reshape(-1, 1)
        
        # Ensure R is 2D
        if isinstance(R, (int, float)):
            R = np.array([[R]])
        elif R.ndim == 1:
            R = np.diag(R)
        
        # Solve discrete-time algebraic Riccati equation
        S = scipy.linalg.solve_discrete_are(Ad, Bd, Q, R)
        
        # Compute optimal gain
        K = -np.linalg.solve(R + Bd.T @ S @ Bd, Bd.T @ S @ Ad)
        
        return K, S


    def discrete_kalman_gain(self, Q_process: Optional[np.ndarray] = None,
                            R_measurement: Optional[np.ndarray] = None,
                            x_eq: Optional[torch.Tensor] = None) -> np.ndarray:
        """
        Compute discrete-time Kalman filter gain
        
        Observer update: x̂[k+1] = f_discrete(x̂[k], u[k]) + L(y[k+1] - h(f_discrete(x̂[k], u[k])))
        
        Args:
            Q_process: Process noise covariance (nx, nx)
            R_measurement: Measurement noise covariance (ny, ny) or scalar
            x_eq: Equilibrium state
        
        Returns:
            L: Kalman gain matrix (nx, ny)
        """
        if Q_process is None:
            Q_process = np.eye(self.nx) * 1e-3
        if R_measurement is None:
            R_measurement = np.eye(self.continuous_time_system.ny) * 1e-3
        if x_eq is None:
            x_eq = self.x_equilibrium
        
        # Ensure proper shape
        if len(x_eq.shape) == 1:
            x_eq = x_eq.unsqueeze(0)
        
        # Get discrete linearized dynamics
        u_eq = self.u_equilibrium
        if len(u_eq.shape) == 1:
            u_eq = u_eq.unsqueeze(0)
        
        Ad, _ = self.linearized_dynamics(x_eq, u_eq)
        Ad = Ad.squeeze().detach().cpu().numpy()
        
        C = self.continuous_time_system.linearized_observation(x_eq)
        C = C.squeeze().detach().cpu().numpy()
        
        # Ensure C is 2D
        if C.ndim == 1:
            C = C.reshape(1, -1)
        
        # Ensure R_measurement is 2D
        if isinstance(R_measurement, (int, float)):
            R_measurement = np.array([[R_measurement]])
        elif R_measurement.ndim == 1:
            R_measurement = np.diag(R_measurement)
        
        # Solve discrete-time algebraic Riccati equation (dual problem)
        P = scipy.linalg.solve_discrete_are(Ad.T, C.T, Q_process, R_measurement)
        
        # Compute Kalman gain
        L = P @ C.T @ np.linalg.inv(C @ P @ C.T + R_measurement)
        
        return L


    def dlqg_control(self, Q_lqr: np.ndarray, R_lqr: np.ndarray,
                    Q_process: Optional[np.ndarray] = None,
                    R_measurement: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute discrete-time LQG controller
        
        Returns:
            (K, L): Control gain and observer gain
        """
        K, _ = self.dlqr_control(Q_lqr, R_lqr)
        L = self.discrete_kalman_gain(Q_process, R_measurement)
        return K, L


    def dlqg_closed_loop_matrix(self, K: np.ndarray, L: np.ndarray) -> np.ndarray:
        """
        Compute closed-loop discrete system matrix for LQG control
        
        Args:
            K: Discrete LQR control gain (nu, nx)
            L: Discrete Kalman filter gain (nx, ny)
        
        Returns:
            A_cl: Closed-loop system matrix (2*nx, 2*nx)
        """
        x_eq = self.x_equilibrium.unsqueeze(0)
        u_eq = self.u_equilibrium.unsqueeze(0)
        
        Ad, Bd = self.linearized_dynamics(x_eq, u_eq)
        Ad = Ad.squeeze().detach().cpu().numpy()
        Bd = Bd.squeeze().detach().cpu().numpy()
        
        # Ensure Bd is 2D (nx, nu)
        if Bd.ndim == 1:
            Bd = Bd.reshape(-1, 1)
        
        C = self.continuous_time_system.linearized_observation(x_eq)
        C = C.squeeze().detach().cpu().numpy()
        
        # Ensure C is 2D (ny, nx)
        if C.ndim == 1:
            C = C.reshape(1, -1)
        
        # Ensure K is 2D (nu, nx)
        if K.ndim == 1:
            K = K.reshape(1, -1)
        
        # Ensure L is 2D (nx, ny)
        if L.ndim == 1:
            L = L.reshape(-1, 1)
        
        # Closed-loop discrete system: [x[k], x̂[k]]
        # x[k+1] = Ad @ x[k] + Bd @ K @ x̂[k]
        # x̂[k+1] = Ad @ x̂[k] + Bd @ K @ x̂[k] + L @ (C @ x[k] - C @ x̂[k])
        #         = (Ad + Bd @ K - L @ C) @ x̂[k] + L @ C @ x[k]
        A_cl = np.vstack([
            np.hstack([Ad + Bd @ K, -Bd @ K]),                    # x[k+1]
            np.hstack([L @ C, Ad + Bd @ K - L @ C])              # x̂[k+1]
        ])
        
        # Clean up near-zero entries
        A_cl[np.abs(A_cl) <= 1e-6] = 0
        
        return A_cl


    def output_feedback_lyapunov(self, K: np.ndarray, L: np.ndarray) -> np.ndarray:
        """
        Solve discrete-time Lyapunov equation for output feedback system
        
        For verifying stability of the closed-loop system
        
        Args:
            K: Control gain
            L: Observer gain
        
        Returns:
            S: Solution to discrete Lyapunov equation
        """
        import control
        
        A_cl = self.dlqg_closed_loop_matrix(K, L)
        S = control.dlyap(A_cl, np.eye(2 * self.nx))
        
        return S

    def plot_trajectory(
        self,
        trajectory: torch.Tensor,
        state_names: Optional[List[str]] = None,
        control_sequence: Optional[torch.Tensor] = None,
        title: Optional[str] = None,
        save_html: Optional[str] = None,
        show: bool = True,
    ):
        """
        Plot trajectory using Plotly (interactive visualization)

        Args:
            trajectory: State trajectory (T, nx) or (batch, T, nx)
            state_names: Names for state variables (uses x0, x1, ... if None)
            control_sequence: Optional control inputs (T, nu) or (batch, T, nu)
            title: Plot title
            save_html: If provided, save interactive plot to this HTML file
            show: If True, display the plot
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            print("Error: plotly not installed. Install with: pip install plotly")
            return

        # Handle batched trajectories
        if len(trajectory.shape) == 3:
            batch_size = trajectory.shape[0]
            print(f"Plotting {batch_size} trajectories...")
        else:
            trajectory = trajectory.unsqueeze(0)
            batch_size = 1
            if control_sequence is not None:
                control_sequence = control_sequence.unsqueeze(0)

        # Convert to numpy
        traj_np = trajectory.detach().cpu().numpy()

        # Determine subplot layout
        has_control = control_sequence is not None
        num_plots = self.nx + (self.nu if has_control else 0)

        # Create subplots
        if num_plots <= 3:
            rows, cols = 1, num_plots
        elif num_plots <= 6:
            rows, cols = 2, (num_plots + 1) // 2
        else:
            rows = (num_plots + 2) // 3
            cols = 3

        # State names
        if state_names is None:
            state_names = [f"x{i}" for i in range(self.nx)]

        subplot_titles = state_names.copy()
        if has_control:
            control_names = [f"u{i}" for i in range(self.nu)]
            subplot_titles.extend(control_names)

        fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles)

        # Time axis
        T = traj_np.shape[1]
        time_steps = np.arange(T) * self.dt

        # Plot states
        for i in range(self.nx):
            row = i // cols + 1
            col = i % cols + 1

            for b in range(batch_size):
                fig.add_trace(
                    go.Scatter(
                        x=time_steps,
                        y=traj_np[b, :, i],
                        mode="lines",
                        name=f"{state_names[i]}"
                        + (f" (batch {b})" if batch_size > 1 else ""),
                        showlegend=(batch_size > 1 or self.nx > 1),
                        line=dict(width=2),
                    ),
                    row=row,
                    col=col,
                )

            fig.update_xaxes(title_text="Time (s)", row=row, col=col)
            fig.update_yaxes(title_text=state_names[i], row=row, col=col)

        # Plot controls
        if has_control:
            control_np = control_sequence.detach().cpu().numpy()
            control_time = np.arange(control_np.shape[1]) * self.dt

            for i in range(self.nu):
                plot_idx = self.nx + i
                row = plot_idx // cols + 1
                col = plot_idx % cols + 1

                for b in range(batch_size):
                    fig.add_trace(
                        go.Scatter(
                            x=control_time,
                            y=control_np[b, :, i],
                            mode="lines",
                            name=f"u{i}" + (f" (batch {b})" if batch_size > 1 else ""),
                            showlegend=(batch_size > 1 or self.nu > 1),
                            line=dict(width=2, dash="dash"),
                        ),
                        row=row,
                        col=col,
                    )

                fig.update_xaxes(title_text="Time (s)", row=row, col=col)
                fig.update_yaxes(title_text=f"u{i}", row=row, col=col)

        # Update layout
        if title is None:
            title = f"{self.continuous_time_system.__class__.__name__} Trajectory"

        fig.update_layout(
            title=title, height=300 * rows, showlegend=True, hovermode="x unified"
        )

        # Save if requested
        if save_html:
            fig.write_html(save_html)
            print(f"Interactive plot saved to {save_html}")

        # Show if requested
        if show:
            fig.show()

        return fig

    def plot_phase_portrait_2d(
        self,
        trajectory: torch.Tensor,
        state_indices: Tuple[int, int] = (0, 1),
        state_names: Optional[Tuple[str, str]] = None,
        title: Optional[str] = None,
        save_html: Optional[str] = None,
        show: bool = True,
    ):
        """
        Plot 2D phase portrait

        Args:
            trajectory: State trajectory (T, nx) or (batch, T, nx)
            state_indices: Which two states to plot (default: first two)
            state_names: Names for the two states
            title: Plot title
            save_html: If provided, save to this HTML file
            show: If True, display the plot
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            print("Error: plotly not installed. Install with: pip install plotly")
            return

        # Handle batched trajectories
        if len(trajectory.shape) == 3:
            batch_size = trajectory.shape[0]
        else:
            trajectory = trajectory.unsqueeze(0)
            batch_size = 1

        traj_np = trajectory.detach().cpu().numpy()

        idx0, idx1 = state_indices
        if state_names is None:
            state_names = (f"x{idx0}", f"x{idx1}")

        fig = go.Figure()

        # Plot trajectories
        for b in range(batch_size):
            fig.add_trace(
                go.Scatter(
                    x=traj_np[b, :, idx0],
                    y=traj_np[b, :, idx1],
                    mode="lines+markers",
                    name=f"Trajectory {b}" if batch_size > 1 else "Trajectory",
                    line=dict(width=2),
                    marker=dict(size=4),
                )
            )

            # Mark start and end points
            fig.add_trace(
                go.Scatter(
                    x=[traj_np[b, 0, idx0]],
                    y=[traj_np[b, 0, idx1]],
                    mode="markers",
                    name="Start" if b == 0 else None,
                    marker=dict(size=12, color="green", symbol="star"),
                    showlegend=(b == 0),
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=[traj_np[b, -1, idx0]],
                    y=[traj_np[b, -1, idx1]],
                    mode="markers",
                    name="End" if b == 0 else None,
                    marker=dict(size=12, color="red", symbol="x"),
                    showlegend=(b == 0),
                )
            )

        # Mark equilibrium if it's 2D
        if self.nx >= 2:
            x_eq = self.continuous_time_system.x_equilibrium.detach().cpu().numpy()
            fig.add_trace(
                go.Scatter(
                    x=[x_eq[idx0]],
                    y=[x_eq[idx1]],
                    mode="markers",
                    name="Equilibrium",
                    marker=dict(size=15, color="black", symbol="diamond"),
                )
            )

        if title is None:
            title = f"{self.continuous_time_system.__class__.__name__} Phase Portrait"

        fig.update_layout(
            title=title,
            xaxis_title=state_names[0],
            yaxis_title=state_names[1],
            hovermode="closest",
            width=800,
            height=600,
        )

        if save_html:
            fig.write_html(save_html)
            print(f"Phase portrait saved to {save_html}")

        if show:
            fig.show()

        return fig

class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for nonlinear systems
    
    Works with both SymbolicDynamicalSystem (continuous) and 
    GenericDiscreteTimeSystem (discrete)
    """
    
    def __init__(self, system, Q_process: np.ndarray, R_measurement: np.ndarray):
        """
        Initialize EKF
        
        Args:
            system: SymbolicDynamicalSystem or GenericDiscreteTimeSystem
            Q_process: Process noise covariance (nx, nx)
            R_measurement: Measurement noise covariance (ny, ny)
        """
        self.system = system
        self.Q = Q_process
        self.R = R_measurement
        
        # State estimate and covariance
        self.x_hat = system.x_equilibrium.clone()
        self.P = torch.eye(system.nx) * 0.1
        
        self.is_discrete = hasattr(system, 'continuous_time_system')
    
    def predict(self, u: torch.Tensor, dt: Optional[float] = None):
        """
        Prediction step
        
        Args:
            u: Control input
            dt: Time step (required for continuous systems)
        """
        if self.is_discrete:
            # Discrete system: x̂[k+1|k] = f(x̂[k|k], u[k])
            with torch.no_grad():
                self.x_hat = self.system(self.x_hat, u)
            
            # Propagate covariance: P[k+1|k] = A P[k|k] A^T + Q
            A, _ = self.system.linearized_dynamics(
                self.x_hat.unsqueeze(0), u.unsqueeze(0)
            )
            A = A.squeeze()
        else:
            # Continuous system: integrate forward
            if dt is None:
                raise ValueError("dt required for continuous systems")
            
            with torch.no_grad():
                dx = self.system.forward(self.x_hat, u)
                self.x_hat = self.x_hat + dx * dt
            
            A, _ = self.system.linearized_dynamics(
                self.x_hat.unsqueeze(0), u.unsqueeze(0)
            )
            A = A.squeeze()
            A = torch.eye(self.system.nx) + A * dt  # Euler discretization
        
        Q_tensor = torch.tensor(self.Q, dtype=self.P.dtype, device=self.P.device)
        self.P = A @ self.P @ A.T + Q_tensor
    
    def update(self, y_measurement: torch.Tensor):
        """
        Update step (correction)
        
        Args:
            y_measurement: Actual measurement
        """
        # Ensure y_measurement is 1D
        if len(y_measurement.shape) == 0:
            y_measurement = y_measurement.unsqueeze(0)
        
        # Predicted measurement
        with torch.no_grad():
            if self.is_discrete:
                y_pred = self.system.continuous_time_system.h(self.x_hat.unsqueeze(0)).squeeze()
            else:
                y_pred = self.system.h(self.x_hat.unsqueeze(0)).squeeze()
        
        # Ensure y_pred is 1D
        if len(y_pred.shape) == 0:
            y_pred = y_pred.unsqueeze(0)
        
        # Measurement residual (innovation)
        innovation = y_measurement - y_pred
        if len(innovation.shape) == 0:
            innovation = innovation.unsqueeze(0)
        
        # Get measurement Jacobian
        if self.is_discrete:
            C = self.system.continuous_time_system.linearized_observation(
                self.x_hat.unsqueeze(0)
            ).squeeze()
        else:
            C = self.system.linearized_observation(self.x_hat.unsqueeze(0)).squeeze()
        
        # Ensure C is 2D (ny, nx)
        if len(C.shape) == 1:
            C = C.unsqueeze(0)  # (ny, nx)
        
        # Innovation covariance: S = C P C^T + R
        R_tensor = torch.tensor(self.R, dtype=self.P.dtype, device=self.P.device)
        S = C @ self.P @ C.mT + R_tensor  # Use .mT for matrix transpose
        
        # Ensure S is 2D
        if len(S.shape) == 0:
            S = S.unsqueeze(0).unsqueeze(0)
        elif len(S.shape) == 1:
            S = S.unsqueeze(0)
        
        # Kalman gain: K = P C^T S^{-1}
        Kt = self.P @ C.mT @ torch.inverse(S)  # (nx, ny)
        
        # Update state estimate: x̂ = x̂ + K * innovation
        correction = (Kt @ innovation.unsqueeze(-1)).squeeze(-1)
        self.x_hat = self.x_hat + correction
        
        # Update covariance: P = (I - K C) P
        nx = self.system.nx if not self.is_discrete else self.system.continuous_time_system.nx
        I = torch.eye(nx, device=self.P.device, dtype=self.P.dtype)
        self.P = (I - Kt @ C) @ self.P
    
    def reset(self, x0: Optional[torch.Tensor] = None, P0: Optional[torch.Tensor] = None):
        """Reset filter state"""
        if x0 is not None:
            self.x_hat = x0.clone()
        else:
            if self.is_discrete:
                self.x_hat = self.system.x_equilibrium.clone()
            else:
                self.x_hat = self.system.x_equilibrium.clone()
        
        if P0 is not None:
            self.P = P0.clone()
        else:
            nx = self.system.nx if not self.is_discrete else self.system.continuous_time_system.nx
            self.P = torch.eye(nx) * 0.1

class LinearController:
    """
    Linear state feedback controller: u = K @ (x - x_eq) + u_eq
    """
    
    def __init__(self, K: np.ndarray, x_eq: torch.Tensor, u_eq: torch.Tensor):
        """
        Args:
            K: Control gain matrix
            x_eq: Equilibrium state
            u_eq: Equilibrium control
        """
        self.K = torch.tensor(K, dtype=torch.float32)
        self.x_eq = x_eq
        self.u_eq = u_eq
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Compute control input"""
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        
        u = self.u_eq + (self.K @ (x - self.x_eq).T).T
        
        if squeeze:
            u = u.squeeze(0)
        
        return u
    
    def to(self, device):
        """Move to device"""
        self.K = self.K.to(device)
        self.x_eq = self.x_eq.to(device)
        self.u_eq = self.u_eq.to(device)
        return self


class LinearObserver:
    """
    Linear observer: dx̂/dt = A x̂ + B u + L(y - C x̂)
    """
    
    def __init__(self, system, L: np.ndarray):
        """
        Args:
            system: Dynamical system
            L: Observer gain matrix
        """
        self.system = system
        self.L = torch.tensor(L, dtype=torch.float32)
        self.x_hat = system.x_equilibrium.clone()
    
    def update(self, u: torch.Tensor, y: torch.Tensor, dt: float):
        """
        Update observer state
        
        Args:
            u: Control input
            y: Measurement
            dt: Time step (for continuous systems)
        """
        # Predict
        with torch.no_grad():
            if hasattr(self.system, 'continuous_time_system'):
                # Discrete system
                x_pred = self.system(self.x_hat.unsqueeze(0), u.unsqueeze(0)).squeeze(0)
                y_pred = self.system.continuous_time_system.h(x_pred.unsqueeze(0)).squeeze(0)
            else:
                # Continuous system
                dx = self.system.forward(self.x_hat.unsqueeze(0), u.unsqueeze(0)).squeeze(0)
                x_pred = self.x_hat + dx * dt
                y_pred = self.system.h(x_pred.unsqueeze(0)).squeeze(0)
            
            # Correct
            innovation = y - y_pred
            self.x_hat = x_pred + (self.L @ innovation.unsqueeze(-1)).squeeze(-1)
    
    def reset(self, x0: Optional[torch.Tensor] = None):
        """Reset observer state"""
        if x0 is not None:
            self.x_hat = x0.clone()
        else:
            self.x_hat = self.system.x_equilibrium.clone()
    
    def to(self, device):
        """Move to device"""
        self.L = self.L.to(device)
        self.x_hat = self.x_hat.to(device)
        return self