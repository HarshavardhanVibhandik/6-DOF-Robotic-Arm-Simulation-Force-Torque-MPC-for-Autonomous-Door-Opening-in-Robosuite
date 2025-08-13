**6-DOF-Robotic-Arm-Simulation-Force-Torque-MPC-for-Autonomous-Door-Opening-in-Robosuite**

Contact-aware manipulation policy using tactile sensing and force-torque feedback for compliant door-opening with a Panda arm

# **Project Objective:**
Developed a model-predictive controller (MPC) that uses real-time force/torque feedback and tactile contact sensing to autonomously detect, grasp, and pull open a hinged door in simulation. The system mimics real-world compliant manipulation by integrating wrench regulation, geometric control, and a tactile-triggered grasping policy.

# **Key Contributions:**

Implemented a force-torque cost function to penalize excessive contact forces and torques, avoiding hardware-damaging behavior in manipulation.

Designed a tactile-triggered grasp strategy using MuJoCo contact data for accurate timing of handle engagement.

Used orientation-aligned servoing and wrench-aware trajectory sampling to create a compliant yet forceful door-pulling controller.

Developed a simple MPC loop with noise-injected trajectory rollouts for cost-optimized action selection in Robosuite.

# **Methodology Highlights:**

**1. Force-Torque Sensing & Limits:**

Accessed 6-axis wrench data from MuJoCo sensor readings (force_ee, torque_ee).

Tuned soft/hard thresholds to limit undesired force/torque excursions during contact-based interaction.

**2. Tactile Grasp Detection:**

Used MuJoCo collision API to detect finger-handle contacts.

Applied GRASP_LEVEL grip strength only after sustained contact detection.

**3. MPC Rollouts & Cost Function:**

Sampled noisy action trajectories from a servoing baseline using n_samples × horizon.

Evaluated each using weighted cost terms: position error, orientation error, hinge progress, and wrench penalties.

**4. Real-Time Switching Strategy:**

Pre-contact: servo toward handle with increasing grip.

Post-contact: plan forward motion with constant grasp while minimizing wrench cost.

# **Results & Outcomes:**

Achieved stable door opening with hinge displacement > 0.15 m using a low-DOF compliant controller.

Maintained wrench norms below critical thresholds (≤ 60N, ≤ 12Nm) during pulling, emulating compliant interaction.

Demonstrated reliable grasp detection, low drift, and minimal collisions using simple model-free logic.

# **Technologies & Tools:**

Simulation: Robosuite + MuJoCo

Controller: Model Predictive Control (MPC) with contact feedback

Languages: Python, NumPy, SciPy

Sensors Used: 6D Wrench (Force/Torque), Tactile Collision

Robot: Franka Emika Panda Arm

Environment: Door environment in Robosuite

# **Future Work:**

Extend to visual+tactile grasping using camera-based handle segmentation (e.g., SAM + DINO).

Test Sim2Real transfer with real-world force-torque sensors (e.g., Robotiq FT-300).

Integrate constrained iLQR for smoother long-horizon control and grasp refinement.
