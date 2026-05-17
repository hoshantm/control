#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gymnasium as gym  # Mujoco gymnasium version 1.2.3
import numpy as np
import math
from scipy.signal import cont2discrete
from scipy.linalg import solve_discrete_are

def get_dlqr_gain(A, B, Q, R, dt):
    """
    Converts continuous A, B to discrete Ad, Bd and solves the 
    Discrete Algebraic Riccati Equation (DARE).
    
    LQR (Linear Quadratic Regulator) aims to find the optimal control input 
    that minimizes the cost function J = Sum(x^T Q x + u^T R u).
    Because the simulation runs in discrete time steps (dt), we must 
    convert our continuous system equations to discrete ones before solving.
    """
    # 1. Discretize the system (Zero-Order Hold)
    # The C and D matrices define the system outputs, but for LQR state feedback
    # we assume full state observability, so C is an Identity matrix and D is zero.
    C = np.eye(4)
    D = np.zeros((4, 1))
    sys_continuous = (A, B, C, D)
    
    # Converts to discrete time considering the 0.04s timestep
    # 'zoh' (Zero-Order Hold) assumes the control input is kept constant during each timestep dt
    sys_discrete = cont2discrete(sys_continuous, dt, method='zoh')
    Ad, Bd = sys_discrete[0], sys_discrete[1]
    
    # 2. Solve Discrete Riccati Equation
    # P represents the unique positive-definite solution to the DARE. 
    # It represents the optimal "cost-to-go" matrix from any state.
    P = solve_discrete_are(Ad, Bd, Q, R)
    
    # 3. Compute Discrete LQR Gain Matrix K
    # Formula: K = (R + B^T P B)^(-1) B^T P A
    # This matrix K gives us the optimal action for any given state using: u = -Kx
    K = np.linalg.inv(R + Bd.T @ P @ Bd) @ (Bd.T @ P @ Ad)
    return K

# === 1. System Parameters ===
# These parameters correspond to the physical properties extracted 
# from the MuJoCo XML model of the InvertedPendulum-v5 environment.
M = 10.4720       # Mass of cart (kg)
m = 5.0186        # Mass of pole (kg)
g = 9.81          # Gravity (m/s^2)

# Damping coefficients (friction applied to the joints in the simulation)
b_x = 1.00        # Slider track damping (Ns/m) resists cart movement
b_theta = 1.00    # Hinge joint damping (Nms/rad) resists pole rotation

# Pole Geometry (Capsule)
# In MuJoCo, the pole is modeled as a capsule shape (a cylinder with two hemispherical ends).
r = 0.049         # Capsule radius (m)
l_cyl = 0.600     # Inner cylinder length (m)
l = l_cyl / 2.0   # Distance from pivot to Center of Mass (m)

# Environment specifics
dt = 0.04         # Gym Environment Timestep (matches the frame_skip * MuJoCo simulation step)
GEAR = 100.0      # MuJoCo Actuator multiplier (Force = Action * 100)

# === 2. Moment of Inertia (I) ===
# To build an accurate mathematical model, we need the exact moment of inertia of the pole.
# Since it's a capsule, we calculate the inertia of a cylinder + two hemispheres.

# Calculate volumes to distribute mass evenly (assuming uniform density rho)
V_cyl = math.pi * (r**2) * l_cyl
V_sph = (4.0 / 3.0) * math.pi * (r**3) # Volume of both hemispherical caps combined
V_total = V_cyl + V_sph
rho = m / V_total

# Mass contributions of the cylinder and the spherical caps
m_cyl = rho * V_cyl
m_sph = rho * V_sph 

# Moment of inertia for the cylinder around its center of mass
I_cyl = (1.0 / 12.0) * m_cyl * (3 * r**2 + l_cyl**2)

# Moment of inertia for the two hemispherical caps
# First, calculate their inertia around their own centers
I_sph_centers = (2.0 / 5.0) * m_sph * (r**2)
# Next, use the Parallel Axis Theorem (I = I_cm + m*d^2) to shift their inertia 
# to the center of the capsule. The distance 'd' is half the cylinder length.
I_sph_offset = m_sph * ((l_cyl / 2.0)**2)
I_hemis = I_sph_centers + I_sph_offset

# Total Moment of Inertia (I)
I = I_cyl + I_hemis 

# === 3. Continuous State-Space Matrices (A and B) ===
# We are expressing the system as a linear differential equation: dx/dt = Ax + Bu
# Since pendulum dynamics are highly non-linear (involving sin/cos), we linearize
# the equations around the equilibrium point (upright, where theta = 0, so sin(theta) ≈ theta, cos(theta) ≈ 1).

# 'det' is the determinant of the system's mass/inertia matrix. It's a common denominator 
# when solving the coupled differential equations for cart acceleration and pole angular acceleration.
det = (M + m) * (I + m * l**2) - (m * l)**2

# State vector x = [cart_pos (x), pole_angle (theta), cart_vel (x_dot), pole_ang_vel (theta_dot)]
# The A matrix describes how the system states evolve over time without any control input.
A = np.array([
    [0, 0, 1, 0],[0, 0, 0, 1],[0, -(m**2 * g * l**2) / det, -(I + m * l**2) * b_x / det, (m * l * b_theta) / det],[0, (M + m) * m * g * l / det, (m * l * b_x) / det, -(M + m) * b_theta / det]
])

# MuJoCo Actuator multiplier: Action * 100 = Force in Newtons
GEAR = 100.0  

# The B matrix describes how our control input 'u' (the action) affects the state derivatives.
# Here we multiply by the GEAR ratio so the optimal controller natively outputs actions in the 
# normalized [-1.0, 1.0] range expected by Gymnasium, instead of massive raw force numbers.
B = np.array([
    [0],
    [0],[((I + m * l**2) / det) * GEAR],
    [(-(m * l) / det) * GEAR]
])

# === 4. Weighting Matrices ===
# Q and R matrices define what we care about penalizing in the LQR cost function.
# Higher values = we penalize deviations more strictly.
Q = np.diag([
    200.0,   # Position (Keep heavily localized to the center 0.0, strict penalty)
    100.0,   # Angle (Keep perfectly upright)
    1.0,     # Cart Velocity (Small penalty, allows it to move if needed to balance)
    1.0      # Pole Angular Velocity (Small penalty)
])

# The R matrix penalizes actuation effort.
# Setting it to 1.0 balances control energy usage against state errors.
R = np.array([[1.0]])

# Get Discrete Optimal Gain Matrix K
K = get_dlqr_gain(A, B, Q, R, dt)
print(f"Moment of Inertia (I): {I:.6f} kg*m^2")
print("Discrete LQR Gain Matrix K:", K)

# === 5. Simulation Loop ===
# Initialize the MuJoCo simulation via Gymnasium.
env = gym.make("InvertedPendulum-v5", render_mode="human", width=1200, height=800, reset_noise_scale=0.5)
observation, _ = env.reset()

for _ in range(500):
    # Calculate action using optimal LQR Control Law: u = -Kx
    # Since 'observation' is perfectly mapped to our state [x, theta, dx, dtheta], 
    # taking the dot product directly yields our optimal input.
    action = -K @ observation
    
    # Thanks to the GEAR modifier in the B matrix, `action` naturally 
    # hovers in the ~0.1 to ~1.5 range, completely avoiding clipping.
    # We clip here anyway just to ensure strictly valid actions are sent to the environment.
    action = np.clip(action, env.action_space.low, env.action_space.high)
    
    # Apply the action and advance the physics simulation by one timestep (dt)
    observation, reward, terminated, truncated, info = env.step(action)
    
    # If the pole falls too far (terminated) or the time limit is reached (truncated), reset.
    if terminated or truncated:
        observation, _ = env.reset()

# Safely close the MuJoCo visualizer
env.close()