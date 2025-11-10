import sympy as sp
import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union, Optional, Callable
from enum import Enum
import control
import warnings
from neural_lyapunov_training.symbolic_dynamics import SymbolicDynamicalSystem


class SymbolicPendulum(SymbolicDynamicalSystem):
    """
    Inverted pendulum system (first-order formulation)

    State: [theta, theta_dot]
    Control: [u] (torque)

    Dynamics: Standard inverted pendulum with damping
    """

    def __init__(
        self, m: float = 1.0, l: float = 1.0, beta: float = 1.0, g: float = 9.81
    ):
        super().__init__()
        self.order = 1
        # Store values for backward compatibility
        self.m_val = m
        self.l_val = l
        self.beta_val = beta
        self.g_val = g
        self.inertia_val = m * l**2
        self.define_system(m, l, beta, g)

    def define_system(self, m_val, l_val, beta_val, g_val):
        theta, theta_dot = sp.symbols("theta theta_dot", real=True)
        u = sp.symbols("u", real=True)
        m, l, beta, g = sp.symbols("m l beta g", real=True, positive=True)

        self.parameters = {m: m_val, l: l_val, beta: beta_val, g: g_val}
        self.state_vars = [theta, theta_dot]
        self.control_vars = [u]
        self.output_vars = [theta]

        ml2 = m * l * l
        self._f_sym = sp.Matrix(
            [theta_dot, (-beta / ml2) * theta_dot + (g / l) * sp.sin(theta) + u / ml2]
        )
        self._h_sym = sp.Matrix([theta])

    @property
    def inertia(self):
        """For backward compatibility"""
        return self.inertia_val


class SymbolicQuadrotor2D(SymbolicDynamicalSystem):
    """
    2D Quadrotor system (second-order formulation)

    State: [x, y, theta, x_dot, y_dot, theta_dot]
    Control: [u1, u2] (thrust forces)

    Dynamics: Planar quadrotor with two rotors
    """

    def __init__(
        self,
        length: float = 0.25,
        mass: float = 0.486,
        inertia: float = 0.00383,
        gravity: float = 9.81,
    ):
        super().__init__()
        self.order = 2
        # Store values for backward compatibility
        self.length_val = length
        self.mass_val = mass
        self.inertia_val = inertia
        self.gravity_val = gravity
        self.define_system(length, mass, inertia, gravity)

    def define_system(self, length_val, mass_val, inertia_val, gravity_val):
        x, y, theta, x_dot, y_dot, theta_dot = sp.symbols(
            "x y theta x_dot y_dot theta_dot", real=True
        )
        u1, u2 = sp.symbols("u1 u2", real=True)
        L, m, I, g = sp.symbols("L m I g", real=True, positive=True)

        self.parameters = {L: length_val, m: mass_val, I: inertia_val, g: gravity_val}
        self.state_vars = [x, y, theta, x_dot, y_dot, theta_dot]
        self.control_vars = [u1, u2]
        self.output_vars = [x, y, theta]

        # For second-order system, forward() returns acceleration
        dx_dot = (-1 / m) * sp.sin(theta) * (u1 + u2)
        dy_dot = (1 / m) * sp.cos(theta) * (u1 + u2) - g
        dtheta_dot = (L / I) * (u1 - u2)

        self._f_sym = sp.Matrix([dx_dot, dy_dot, dtheta_dot])
        self._h_sym = sp.Matrix([x, y, theta])

    @property
    def u_equilibrium(self) -> torch.Tensor:
        mg = self.mass_val * self.gravity_val
        return torch.tensor([mg / 2, mg / 2])

    @property
    def length(self):
        """For backward compatibility"""
        return self.length_val

    @property
    def mass(self):
        """For backward compatibility"""
        return self.mass_val

    @property
    def inertia(self):
        """For backward compatibility"""
        return self.inertia_val

    @property
    def gravity(self):
        """For backward compatibility"""
        return self.gravity_val


class FifthOrderMechanicalSystem(SymbolicDynamicalSystem):
    """
    A fifth-order mechanical system: q^(5) = f(q, q', q'', q''', q'''', u)

    State: [q, q', q'', q''', q'''']
    where:
    - q: position
    - q': velocity
    - q'': acceleration
    - q''': jerk
    - q'''': snap

    The forward() method returns q^(5) (the fifth derivative).

    This could model a system with very complex dynamics, such as:
    - A flexible manipulator with high-order modes
    - A system with nested actuator dynamics
    """

    def __init__(
        self,
        m: float = 1.0,
        k: float = 1.0,
        c1: float = 0.1,
        c2: float = 0.05,
        c3: float = 0.01,
        g: float = 9.81,
    ):
        super().__init__()
        self.order = 5
        # Store values for backward compatibility
        self.m_val = m
        self.k_val = k
        self.c1_val = c1
        self.c2_val = c2
        self.c3_val = c3
        self.g_val = g
        self.define_system(m, k, c1, c2, c3, g)

    def define_system(self, m_val, k_val, c1_val, c2_val, c3_val, g_val):
        q, q1, q2, q3, q4 = sp.symbols("q q1 q2 q3 q4", real=True)
        u = sp.symbols("u", real=True)
        m, k, c1, c2, c3, g = sp.symbols("m k c1 c2 c3 g", real=True, positive=True)

        self.parameters = {
            m: m_val,
            k: k_val,
            c1: c1_val,
            c2: c2_val,
            c3: c3_val,
            g: g_val,
        }

        self.state_vars = [q, q1, q2, q3, q4]
        self.control_vars = [u]
        self.output_vars = [q, q1]

        # Fifth derivative: complex dynamics with multiple damping terms
        q5 = -k / m * q - c1 * q1 - c2 * q2 - c3 * q3 - 0.01 * q4 - g + u / m

        self._f_sym = sp.Matrix([q5])
        self._h_sym = sp.Matrix([q, q1])

    @property
    def x_equilibrium(self) -> torch.Tensor:
        q_eq = -self.g_val * self.m_val / self.k_val
        return torch.tensor([q_eq, 0.0, 0.0, 0.0, 0.0])

    @property
    def u_equilibrium(self) -> torch.Tensor:
        return torch.tensor([self.m_val * self.g_val])


class CoupledOscillatorSystem(SymbolicDynamicalSystem):
    """
    A first-order system with 5 state variables: coupled oscillators.

    State: [x1, x2, v1, v2, theta]
    - Two masses coupled with springs
    - One rotational degree of freedom
    """

    def __init__(
        self,
        m1: float = 1.0,
        m2: float = 0.5,
        k1: float = 2.0,
        k2: float = 1.0,
        k_coupling: float = 0.5,
        c: float = 0.1,
        J: float = 0.1,
    ):
        super().__init__()
        self.order = 1
        # Store values
        self.m1_val = m1
        self.m2_val = m2
        self.k1_val = k1
        self.k2_val = k2
        self.k_coupling_val = k_coupling
        self.c_val = c
        self.J_val = J
        self.define_system(m1, m2, k1, k2, k_coupling, c, J)

    def define_system(
        self, m1_val, m2_val, k1_val, k2_val, k_coupling_val, c_val, J_val
    ):
        x1, x2, v1, v2, theta = sp.symbols("x1 x2 v1 v2 theta", real=True)
        u1, u2 = sp.symbols("u1 u2", real=True)
        m1, m2, k1, k2, k_c, c, J = sp.symbols(
            "m1 m2 k1 k2 k_c c J", real=True, positive=True
        )

        self.parameters = {
            m1: m1_val,
            m2: m2_val,
            k1: k1_val,
            k2: k2_val,
            k_c: k_coupling_val,
            c: c_val,
            J: J_val,
        }

        self.state_vars = [x1, x2, v1, v2, theta]
        self.control_vars = [u1, u2]
        self.output_vars = [x1, x2, theta]

        # Coupled dynamics
        dx1 = v1
        dv1 = -k1 / m1 * x1 - k_c / m1 * (x1 - x2) - c / m1 * v1 + u1 / m1
        dx2 = v2
        dv2 = (
            -k2 / m2 * x2
            - k_c / m2 * (x2 - x1)
            - c / m2 * v2
            + sp.sin(theta) / m2
            + u2 / m2
        )
        dtheta = -theta / J - x2 / J + u2 / (2 * J)

        self._f_sym = sp.Matrix([dx1, dx2, dv1, dv2, dtheta])
        self._h_sym = sp.Matrix([x1, x2, theta])

    @property
    def x_equilibrium(self) -> torch.Tensor:
        return torch.zeros(5)

    @property
    def u_equilibrium(self) -> torch.Tensor:
        return torch.zeros(2)


class NonlinearChainSystem(SymbolicDynamicalSystem):
    """
    A chain of 5 coupled nonlinear oscillators (first-order system).
    Each oscillator influences its neighbors.

    State: [x1, x2, x3, x4, x5]
    """

    def __init__(self, k: float = 1.0, c: float = 0.1, alpha: float = 0.1):
        super().__init__()
        self.order = 1
        # Store values
        self.k_val = k
        self.c_val = c
        self.alpha_val = alpha
        self.define_system(k, c, alpha)

    def define_system(self, k_val, c_val, alpha_val):
        x1, x2, x3, x4, x5 = sp.symbols("x1 x2 x3 x4 x5", real=True)
        u = sp.symbols("u", real=True)
        k, c, alpha = sp.symbols("k c alpha", real=True, positive=True)

        self.parameters = {k: k_val, c: c_val, alpha: alpha_val}

        self.state_vars = [x1, x2, x3, x4, x5]
        self.control_vars = [u]
        self.output_vars = [x1, x3, x5]

        # Chain dynamics with nonlinear coupling
        dx1 = -k * x1 - c * x1 + alpha * sp.sin(x2 - x1) + u
        dx2 = -k * x2 - c * x2 + alpha * sp.sin(x1 - x2) + alpha * sp.sin(x3 - x2)
        dx3 = -k * x3 - c * x3 + alpha * sp.sin(x2 - x3) + alpha * sp.sin(x4 - x3)
        dx4 = -k * x4 - c * x4 + alpha * sp.sin(x3 - x4) + alpha * sp.sin(x5 - x4)
        dx5 = -k * x5 - c * x5 + alpha * sp.sin(x4 - x5)

        self._f_sym = sp.Matrix([dx1, dx2, dx3, dx4, dx5])
        self._h_sym = sp.Matrix([x1, x3, x5])

    @property
    def x_equilibrium(self) -> torch.Tensor:
        return torch.zeros(5)

    @property
    def u_equilibrium(self) -> torch.Tensor:
        return torch.zeros(1)
