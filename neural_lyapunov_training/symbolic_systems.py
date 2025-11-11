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


class SymbolicPendulumState(SymbolicDynamicalSystem):
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

        ml2 = m * l * l
        self._f_sym = sp.Matrix(
            [theta_dot, (-beta / ml2) * theta_dot + (g / l) * sp.sin(theta) + u / ml2]
        )

        @property
        def inertia(self):
            """For backward compatibility"""
            return self.inertia_val

class SymbolicPendulum2ndOrder(SymbolicDynamicalSystem):
    """Second-order formulation (returns only acceleration)"""
    
    def __init__(self, m=1.0, l=1.0, beta=1.0, g=9.81):
        super().__init__()
        self.order = 2  # ← SECOND-ORDER
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
        
        ml2 = m * l * l
        # For 2nd order, return ONLY acceleration
        theta_ddot = (-beta / ml2) * theta_dot + (g / l) * sp.sin(theta) + u / ml2
        
        self._f_sym = sp.Matrix([theta_ddot])  # ← Only acceleration


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


class SymbolicQuadrotor2DState(SymbolicDynamicalSystem):
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

        # For second-order system, forward() returns acceleration
        dx_dot = (-1 / m) * sp.sin(theta) * (u1 + u2)
        dy_dot = (1 / m) * sp.cos(theta) * (u1 + u2) - g
        dtheta_dot = (L / I) * (u1 - u2)

        self._f_sym = sp.Matrix([dx_dot, dy_dot, dtheta_dot])

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

class CartPole(SymbolicDynamicalSystem):
    """
    Cart-pole system (inverted pendulum on cart) - second-order formulation
    
    State: [x, theta, x_dot, theta_dot]
    where:
    - x: cart position
    - theta: pole angle from vertical (0 = upright)
    - x_dot: cart velocity
    - theta_dot: pole angular velocity
    
    Control: [F] (horizontal force on cart)
    
    Classic underactuated control problem
    """
    
    def __init__(
        self,
        m_cart: float = 1.0,
        m_pole: float = 0.1,
        length: float = 0.5,
        gravity: float = 9.81,
        friction: float = 0.1,
    ):
        super().__init__()
        self.order = 2
        # Store values
        self.m_cart_val = m_cart
        self.m_pole_val = m_pole
        self.length_val = length
        self.gravity_val = gravity
        self.friction_val = friction
        self.define_system(m_cart, m_pole, length, gravity, friction)
    
    def define_system(self, m_cart_val, m_pole_val, length_val, gravity_val, friction_val):
        # State variables
        x, theta, x_dot, theta_dot = sp.symbols('x theta x_dot theta_dot', real=True)
        F = sp.symbols('F', real=True)
        
        # Parameters
        mc, mp, l, g, b = sp.symbols('mc mp l g b', real=True, positive=True)
        
        self.parameters = {
            mc: m_cart_val,
            mp: m_pole_val,
            l: length_val,
            g: gravity_val,
            b: friction_val
        }
        
        self.state_vars = [x, theta, x_dot, theta_dot]
        self.control_vars = [F]
        self.output_vars = [x, theta]
        
        # Dynamics (derived from Euler-Lagrange equations)
        # Total mass
        M = mc + mp
        
        # Sin and cos of theta
        sin_theta = sp.sin(theta)
        cos_theta = sp.cos(theta)
        
        # Denominator for both equations
        denom = M - mp * cos_theta**2
        
        # Cart acceleration
        x_ddot = (F - b*x_dot + mp*l*theta_dot**2*sin_theta - mp*g*sin_theta*cos_theta) / denom
        
        # Pole angular acceleration  
        theta_ddot = (F*cos_theta - b*x_dot*cos_theta + mp*l*theta_dot**2*sin_theta*cos_theta 
                      - M*g*sin_theta) / (l * denom)
        
        # Second-order system: forward() returns accelerations
        self._f_sym = sp.Matrix([x_ddot, theta_ddot])
        self._h_sym = sp.Matrix([x, theta])
    
    @property
    def x_equilibrium(self) -> torch.Tensor:
        """Upright equilibrium at origin"""
        return torch.zeros(4)
    
    @property
    def u_equilibrium(self) -> torch.Tensor:
        """No force needed at equilibrium"""
        return torch.zeros(1)


class VanDerPolOscillator(SymbolicDynamicalSystem):
    """
    Van der Pol oscillator - classic nonlinear oscillator with limit cycle
    
    State: [x, y] where y = dx/dt
    Control: [u] (forcing term)
    
    Exhibits self-sustained oscillations and limit cycle behavior
    """
    
    def __init__(self, mu: float = 1.0):
        super().__init__()
        self.order = 1
        self.mu_val = mu
        self.define_system(mu)
    
    def define_system(self, mu_val):
        x, y = sp.symbols('x y', real=True)
        u = sp.symbols('u', real=True)
        mu = sp.symbols('mu', real=True, positive=True)
        
        self.parameters = {mu: mu_val}
        self.state_vars = [x, y]
        self.control_vars = [u]
        self.output_vars = [x]
        
        # Van der Pol dynamics: d²x/dt² - μ(1-x²)dx/dt + x = u
        # Rewritten as first-order system
        dx = y
        dy = mu * (1 - x**2) * y - x + u
        
        self._f_sym = sp.Matrix([dx, dy])
        self._h_sym = sp.Matrix([x])
    
    @property
    def x_equilibrium(self) -> torch.Tensor:
        return torch.zeros(2)
    
    @property  
    def u_equilibrium(self) -> torch.Tensor:
        return torch.zeros(1)


class DubinsVehicle(SymbolicDynamicalSystem):
    """
    Dubins vehicle (kinematic car model)
    
    State: [x, y, theta]
    - (x, y): position
    - theta: heading angle
    
    Control: [v, omega]
    - v: forward velocity
    - omega: angular velocity
    
    Simple kinematic model for wheeled robots
    """
    
    def __init__(self):
        super().__init__()
        self.order = 1
        self.define_system()
    
    def define_system(self):
        x, y, theta = sp.symbols('x y theta', real=True)
        v, omega = sp.symbols('v omega', real=True)
        
        self.parameters = {}
        self.state_vars = [x, y, theta]
        self.control_vars = [v, omega]
        self.output_vars = [x, y, theta]
        
        # Kinematic equations
        dx = v * sp.cos(theta)
        dy = v * sp.sin(theta)
        dtheta = omega
        
        self._f_sym = sp.Matrix([dx, dy, dtheta])
        self._h_sym = sp.Matrix([x, y, theta])
    
    @property
    def x_equilibrium(self) -> torch.Tensor:
        return torch.zeros(3)
    
    @property
    def u_equilibrium(self) -> torch.Tensor:
        return torch.zeros(2)


class Manipulator2Link(SymbolicDynamicalSystem):
    """
    2-link planar manipulator - second-order formulation
    
    State: [q1, q2, q1_dot, q2_dot]
    - q1, q2: joint angles
    - q1_dot, q2_dot: joint velocities
    
    Control: [tau1, tau2] (joint torques)
    
    Classic robotics system with coupling dynamics
    """
    
    def __init__(
        self,
        m1: float = 1.0,
        m2: float = 1.0, 
        l1: float = 1.0,
        l2: float = 1.0,
        lc1: float = 0.5,
        lc2: float = 0.5,
        I1: float = 0.1,
        I2: float = 0.1,
        gravity: float = 9.81,
        friction1: float = 0.1,
        friction2: float = 0.1,
    ):
        super().__init__()
        self.order = 2
        # Store values
        self.m1_val = m1
        self.m2_val = m2
        self.l1_val = l1
        self.l2_val = l2
        self.lc1_val = lc1
        self.lc2_val = lc2
        self.I1_val = I1
        self.I2_val = I2
        self.gravity_val = gravity
        self.friction1_val = friction1
        self.friction2_val = friction2
        self.define_system(m1, m2, l1, l2, lc1, lc2, I1, I2, gravity, friction1, friction2)
    
    def define_system(self, m1_val, m2_val, l1_val, l2_val, lc1_val, lc2_val, 
                      I1_val, I2_val, gravity_val, friction1_val, friction2_val):
        # State variables
        q1, q2, q1_dot, q2_dot = sp.symbols('q1 q2 q1_dot q2_dot', real=True)
        tau1, tau2 = sp.symbols('tau1 tau2', real=True)
        
        # Parameters
        m1, m2, l1, l2, lc1, lc2 = sp.symbols('m1 m2 l1 l2 lc1 lc2', real=True, positive=True)
        I1, I2, g, b1, b2 = sp.symbols('I1 I2 g b1 b2', real=True, positive=True)
        
        self.parameters = {
            m1: m1_val, m2: m2_val,
            l1: l1_val, l2: l2_val,
            lc1: lc1_val, lc2: lc2_val,
            I1: I1_val, I2: I2_val,
            g: gravity_val,
            b1: friction1_val, b2: friction2_val
        }
        
        self.state_vars = [q1, q2, q1_dot, q2_dot]
        self.control_vars = [tau1, tau2]
        self.output_vars = [q1, q2]
        
        # Mass matrix M(q)
        M11 = m1*lc1**2 + m2*(l1**2 + lc2**2 + 2*l1*lc2*sp.cos(q2)) + I1 + I2
        M12 = m2*(lc2**2 + l1*lc2*sp.cos(q2)) + I2
        M21 = M12
        M22 = m2*lc2**2 + I2
        
        # Coriolis and centrifugal terms C(q, q_dot)
        h = -m2*l1*lc2*sp.sin(q2)
        C1 = h * (2*q1_dot*q2_dot + q2_dot**2)
        C2 = -h * q1_dot**2
        
        # Gravity terms G(q)
        G1 = (m1*lc1 + m2*l1)*g*sp.cos(q1) + m2*lc2*g*sp.cos(q1 + q2)
        G2 = m2*lc2*g*sp.cos(q1 + q2)
        
        # Friction
        F1 = b1 * q1_dot
        F2 = b2 * q2_dot
        
        # Solve for accelerations: M * q_ddot = tau - C - G - F
        # q_ddot = M^(-1) * (tau - C - G - F)
        det_M = M11*M22 - M12*M21
        
        # Inverse of M
        M_inv_11 = M22 / det_M
        M_inv_12 = -M12 / det_M
        M_inv_21 = -M21 / det_M
        M_inv_22 = M11 / det_M
        
        # Right-hand side
        rhs1 = tau1 - C1 - G1 - F1
        rhs2 = tau2 - C2 - G2 - F2
        
        # Accelerations
        q1_ddot = M_inv_11 * rhs1 + M_inv_12 * rhs2
        q2_ddot = M_inv_21 * rhs1 + M_inv_22 * rhs2
        
        self._f_sym = sp.Matrix([q1_ddot, q2_ddot])
        self._h_sym = sp.Matrix([q1, q2])
    
    @property
    def x_equilibrium(self) -> torch.Tensor:
        """Hanging down equilibrium"""
        return torch.tensor([sp.pi, 0.0, 0.0, 0.0])
    
    @property
    def u_equilibrium(self) -> torch.Tensor:
        """Zero torque at hanging equilibrium"""
        return torch.zeros(2)


class PathTracking(SymbolicDynamicalSystem):
    """
    Path tracking error dynamics for a vehicle following a circular path
    
    State: [d_e, theta_e]
    - d_e: lateral tracking error
    - theta_e: heading error
    
    Control: [delta] (steering angle)
    
    Converted from hard-coded PathTrackingDynamics
    """
    
    def __init__(self, speed: float = 1.0, length: float = 1.0, radius: float = 10.0):
        super().__init__()
        self.order = 1
        self.speed_val = speed
        self.length_val = length
        self.radius_val = radius
        self.define_system(speed, length, radius)
    
    def define_system(self, speed_val, length_val, radius_val):
        d_e, theta_e = sp.symbols('d_e theta_e', real=True)
        delta = sp.symbols('delta', real=True)
        v, L, R = sp.symbols('v L R', real=True, positive=True)
        
        self.parameters = {
            v: speed_val,
            L: length_val,
            R: radius_val
        }
        
        self.state_vars = [d_e, theta_e]
        self.control_vars = [delta]
        self.output_vars = [d_e, theta_e]
        
        # Error dynamics
        sin_theta_e = sp.sin(theta_e)
        cos_theta_e = sp.cos(theta_e)
        
        # Lateral error rate
        d_e_dot = v * sin_theta_e
        
        # Heading error rate
        coef = R / v
        theta_e_dot = (v * delta / L) - (cos_theta_e / (coef - sin_theta_e))
        
        self._f_sym = sp.Matrix([d_e_dot, theta_e_dot])
        self._h_sym = sp.Matrix([d_e, theta_e])
    
    @property
    def x_equilibrium(self) -> torch.Tensor:
        return torch.zeros(2)
    
    @property
    def u_equilibrium(self) -> torch.Tensor:
        return torch.tensor([self.length_val / self.radius_val])


class PVTOL(SymbolicDynamicalSystem):
    """
    Planar Vertical Take-Off and Landing (PVTOL) aircraft - second-order formulation
    
    State: [x, y, theta, x_dot, y_dot, theta_dot]
    - (x, y): position in body frame
    - theta: pitch angle
    - velocities in body frame
    
    Control: [u1, u2] (thrust forces from two rotors)
    
    Converted from hard-coded PvtolDynamics
    """
    
    def __init__(
        self,
        length: float = 0.25,
        mass: float = 4.0,
        inertia: float = 0.0475,
        gravity: float = 9.8,
        dist: float = 0.25,
    ):
        super().__init__()
        self.order = 2
        # Store values
        self.length_val = length
        self.mass_val = mass
        self.inertia_val = inertia
        self.gravity_val = gravity
        self.dist_val = dist
        self.define_system(length, mass, inertia, gravity, dist)
    
    def define_system(self, length_val, mass_val, inertia_val, gravity_val, dist_val):
        # State variables (position and velocity in body frame)
        x, y, theta, x_dot, y_dot, theta_dot = sp.symbols(
            'x y theta x_dot y_dot theta_dot', real=True
        )
        u1, u2 = sp.symbols('u1 u2', real=True)
        
        # Parameters
        L, m, I, g, d = sp.symbols('L m I g d', real=True, positive=True)
        
        self.parameters = {
            L: length_val,
            m: mass_val,
            I: inertia_val,
            g: gravity_val,
            d: dist_val
        }
        
        self.state_vars = [x, y, theta, x_dot, y_dot, theta_dot]
        self.control_vars = [u1, u2]
        self.output_vars = [x, y, theta]
        
        # Rotation from body to world frame
        sin_theta = sp.sin(theta)
        cos_theta = sp.cos(theta)
        
        # The original code has velocities in a rotated frame
        # Position derivatives in world frame
        # x_change = x_dot * cos_theta - y_dot * sin_theta
        # y_change = x_dot * sin_theta + y_dot * cos_theta
        
        # Acceleration dynamics in body frame
        x_ddot = y_dot * theta_dot - g * sin_theta
        y_ddot = -x_dot * theta_dot - g * cos_theta + (u1 + u2) / m
        theta_ddot = (u1 - u2) * d / I
        
        # For second-order system, forward() returns accelerations
        self._f_sym = sp.Matrix([x_ddot, y_ddot, theta_ddot])
        self._h_sym = sp.Matrix([x, y, theta])
    
    @property
    def x_equilibrium(self) -> torch.Tensor:
        return torch.zeros(6)
    
    @property
    def u_equilibrium(self) -> torch.Tensor:
        return torch.full((2,), self.mass_val * self.gravity_val / 2)
    
    @property
    def length(self):
        return self.length_val
    
    @property
    def mass(self):
        return self.mass_val
    
    @property
    def inertia(self):
        return self.inertia_val
    
    @property
    def gravity(self):
        return self.gravity_val
    
    @property
    def dist(self):
        return self.dist_val


class Lorenz(SymbolicDynamicalSystem):
    """
    Lorenz system - famous chaotic system
    
    State: [x, y, z]
    Control: [u] (external forcing)
    
    Exhibits chaotic behavior for certain parameter values
    """
    
    def __init__(self, sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0/3.0):
        super().__init__()
        self.order = 1
        self.sigma_val = sigma
        self.rho_val = rho
        self.beta_val = beta
        self.define_system(sigma, rho, beta)
    
    def define_system(self, sigma_val, rho_val, beta_val):
        x, y, z = sp.symbols('x y z', real=True)
        u = sp.symbols('u', real=True)
        sigma, rho, beta = sp.symbols('sigma rho beta', real=True, positive=True)
        
        self.parameters = {
            sigma: sigma_val,
            rho: rho_val,
            beta: beta_val
        }
        
        self.state_vars = [x, y, z]
        self.control_vars = [u]
        self.output_vars = [x, y]
        
        # Lorenz dynamics with control
        dx = sigma * (y - x) + u
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        
        self._f_sym = sp.Matrix([dx, dy, dz])
        self._h_sym = sp.Matrix([x, y])
    
    @property
    def x_equilibrium(self) -> torch.Tensor:
        """Origin (unstable for standard parameters)"""
        return torch.zeros(3)
    
    @property
    def u_equilibrium(self) -> torch.Tensor:
        return torch.zeros(1)


class DuffingOscillator(SymbolicDynamicalSystem):
    """
    Duffing oscillator - nonlinear oscillator with cubic stiffness
    
    State: [x, v]
    Control: [u] (forcing)
    
    Can exhibit chaotic behavior and multiple stable states
    """
    
    def __init__(
        self,
        alpha: float = -1.0,
        beta: float = 1.0,
        delta: float = 0.3,
        gamma: float = 0.0,
        omega: float = 1.0,
    ):
        super().__init__()
        self.order = 1
        self.alpha_val = alpha
        self.beta_val = beta
        self.delta_val = delta
        self.gamma_val = gamma
        self.omega_val = omega
        self.define_system(alpha, beta, delta, gamma, omega)
    
    def define_system(self, alpha_val, beta_val, delta_val, gamma_val, omega_val):
        x, v = sp.symbols('x v', real=True)
        u = sp.symbols('u', real=True)
        alpha, beta, delta, gamma, omega = sp.symbols(
            'alpha beta delta gamma omega', real=True
        )
        
        self.parameters = {
            alpha: alpha_val,
            beta: beta_val,
            delta: delta_val,
            gamma: gamma_val,
            omega: omega_val
        }
        
        self.state_vars = [x, v]
        self.control_vars = [u]
        self.output_vars = [x]
        
        # Duffing equation: d²x/dt² + delta*dx/dt + alpha*x + beta*x³ = gamma*cos(omega*t) + u
        # First-order form
        dx = v
        dv = -delta * v - alpha * x - beta * x**3 + u
        
        self._f_sym = sp.Matrix([dx, dv])
        self._h_sym = sp.Matrix([x])
    
    @property
    def x_equilibrium(self) -> torch.Tensor:
        return torch.zeros(2)
    
    @property
    def u_equilibrium(self) -> torch.Tensor:
        return torch.zeros(1)