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
    """
    # 1. Discretize the system (Zero-Order Hold)
    C = np.eye(4)
    D = np.zeros((4, 1))
    sys_continuous = (A, B, C, D)
    
    # Converts to discrete time considering the 0.04s timestep
    sys_discrete = cont2discrete(sys_continuous, dt, method='zoh')
    Ad, Bd = sys_discrete[0], sys_discrete[1]
    
    # 2. Solve Discrete Riccati Equation
    P = solve_discrete_are(Ad, Bd, Q, R)
    
    # 3. Compute Discrete LQR Gain Matrix K
    # Formula: K = (R + B^T P B)^(-1) B^T P A
    K = np.linalg.inv(R + Bd.T @ P @ Bd) @ (Bd.T @ P @ Ad)
    return K

# === 1. System Parameters ===
M = 10.4720       # Mass of cart (kg)
m = 5.0186        # Mass of pole (kg)
g = 9.81          # Gravity (m/s^2)

# Damping coefficients
b_x = 1.00        # Slider track damping (Ns/m)
b_theta = 1.00    # Hinge joint damping (Nms/rad)

# Pole Geometry (Capsule)
r = 0.049         # Capsule radius (m)
l_cyl = 0.600     # Inner cylinder length (m)
l = l_cyl / 2.0   # Distance from pivot to Center of Mass (m)

# Environment specifics
dt = 0.04         # Gym Environment Timestep
GEAR = 100.0      # MuJoCo Actuator multiplier (Force = Action * 100)

# === 2. Moment of Inertia (I) ===
V_cyl = math.pi * (r**2) * l_cyl
V_sph = (4.0 / 3.0) * math.pi * (r**3)
V_total = V_cyl + V_sph
rho = m / V_total

m_cyl = rho * V_cyl
m_sph = rho * V_sph 

I_cyl = (1.0 / 12.0) * m_cyl * (3 * r**2 + l_cyl**2)
I_sph_centers = (2.0 / 5.0) * m_sph * (r**2)
I_sph_offset = m_sph * ((l_cyl / 2.0)**2)
I_hemis = I_sph_centers + I_sph_offset
I = I_cyl + I_hemis 

# === 3. Continuous State-Space Matrices (A and B) ===
det = (M + m) * (I + m * l**2) - (m * l)**2

# State x =[cart_pos, pole_angle, cart_vel, pole_ang_vel]
A = np.array([
    [0, 0, 1, 0],[0, 0, 0, 1],[0, -(m**2 * g * l**2) / det, -(I + m * l**2) * b_x / det, (m * l * b_theta) / det],[0, (M + m) * m * g * l / det, (m * l * b_x) / det, -(M + m) * b_theta / det]
])

# MuJoCo Actuator multiplier: Action * 100 = Force in Newtons
GEAR = 100.0  

B = np.array([
    [0],
    [0],[((I + m * l**2) / det) * GEAR],
    [(-(m * l) / det) * GEAR]
])

# === 4. Weighting Matrices ===
Q = np.diag([
    200.0,   # Position (Keep heavily localized to the center 0.0)
    100.0,   # Angle (Keep perfectly upright)
    1.0,    # Cart Velocity
    1.0      # Pole Angular Velocity
])

R = np.array([[1.0]])

# Get Discrete Optimal Gain Matrix K
K = get_dlqr_gain(A, B, Q, R, dt)
print(f"Moment of Inertia (I): {I:.6f} kg*m^2")
print("Discrete LQR Gain Matrix K:", K)

# === 5. Simulation Loop ===
env = gym.make("InvertedPendulum-v5", render_mode="human", width=1200, height=800, reset_noise_scale=0.5)
observation, _ = env.reset()

for _ in range(500):
    # Calculate action
    action = -K @ observation
    
    # Thanks to the GEAR modifier in the B matrix, `action` naturally 
    # hovers in the ~0.1 to ~1.5 range, completely avoiding clipping.
    action = np.clip(action, env.action_space.low, env.action_space.high)
    
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        observation, _ = env.reset()

env.close()