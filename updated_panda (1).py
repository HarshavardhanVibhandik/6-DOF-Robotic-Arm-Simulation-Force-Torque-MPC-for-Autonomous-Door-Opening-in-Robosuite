from __future__ import annotations
import os
import copy
import time
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation as Rot
import robosuite as suite
from robosuite.controllers import load_composite_controller_config

os.environ.setdefault("MUJOCO_GL", "glfw")

CTRL_CFG = load_composite_controller_config(controller="BASIC")
ENV = suite.make(
    "Door",
    robots="Panda",
    controller_configs=CTRL_CFG,
    use_latch=False,
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    use_object_obs=True,
    horizon=10_000,
)
MODEL, DATA = ENV.sim.model, ENV.sim.data

sensor_names = MODEL.sensor_names
FORCE_SID  = next(i for i,n in enumerate(sensor_names) if "force_ee" in n.lower())
TORQUE_SID = next(i for i,n in enumerate(sensor_names) if "torque_ee" in n.lower())

ACT_LOW, ACT_HIGH = ENV.action_spec
GRIP_INDEX = ACT_LOW.shape[0] - 1

def orientation_error(quat: np.ndarray, target: np.ndarray) -> np.ndarray:
    z_axis = Rot.from_quat(quat).as_matrix()[:, 2]
    tgt = target / (np.linalg.norm(target) + 1e-8)
    return 0.5 * np.cross(z_axis, tgt)

def get_wrench(sim) -> Tuple[np.ndarray, np.ndarray]:
    sd = sim.data.sensordata
    f = sd[FORCE_SID:FORCE_SID+3].copy()
    t = sd[TORQUE_SID:TORQUE_SID+3].copy()
    return f, t

class TactileSensor:
    def __init__(self, model):
        self.fingers = [
            i for i, n in enumerate(model.geom_names)
            if "finger" in n.lower() and ("tip" in n.lower() or "pad" in n.lower())
        ]
        self.handle = [
            i for i, n in enumerate(model.geom_names)
            if "handle" in n.lower() or "latch" in n.lower()
        ]
        if not self.fingers or not self.handle:
            raise RuntimeError("Missing finger or handle geoms in MJCF.")

    def sense(self, sim) -> bool:
        for k in range(sim.data.ncon):
            c = sim.data.contact[k]
            if ((c.geom1 in self.fingers and c.geom2 in self.handle)
                or (c.geom2 in self.fingers and c.geom1 in self.handle)):
                return True
        return False

@dataclass
class FTLimits:
    f_soft: float = 30.0
    f_hard: float = 60.0
    t_soft: float = 6.0
    t_hard: float = 12.0

class ForceTorqueMPC:
    SERVO_GAIN = 4.0
    ORI_GAIN = 1.5
    SERVO_DIST = 0.04
    APPROACH_DIST = 0.18
    CONTACT_TIMEOUT = 0.4
    PREGRASP_LEVEL = 0.30
    GRASP_LEVEL = 0.60

    def __init__(
        self,
        env,
        horizon: int = 15,
        n_samples: int = 256,
        noise_scale: float = 0.4,
        ft: FTLimits | None = None,
        seed: int | None = None,
    ):
        self.env = env
        self.sim = env.sim
        self.horizon = horizon
        self.n_samples = n_samples
        self.noise = noise_scale
        self.ft = ft or FTLimits()
        self.rng = np.random.default_rng(seed)
        self.act_dim = ACT_LOW.shape[0]
        self.w_pos = 12.0
        self.w_ori = 1.5
        self.w_hinge = -60.0
        self.w_force = 2.0
        self.w_torque = 0.8
        self.w_hard = 1e3
        self.best_seq = None
        self._touch = TactileSensor(self.sim.model)
        self.has_grasp = False
        self.last_contact = -np.inf
        self.grip_cmd = -1.0

    def _wrench_cost(self, f, t):
        fm, tm = np.linalg.norm(f), np.linalg.norm(t)
        soft = self.w_force * max(0, fm - self.ft.f_soft)**2 + \
               self.w_torque * max(0, tm - self.ft.t_soft)**2
        hard = self.w_hard * (fm + tm) if (fm > self.ft.f_hard or tm > self.ft.t_hard) else 0.0
        return soft + hard

    def _stage_cost(self, obs, f, t):
        pos, quat = obs["robot0_eef_pos"], obs["robot0_eef_quat"]
        handle = obs["handle_pos"]
        hinge = obs["hinge_qpos"]
        pe = np.linalg.norm(handle - pos)
        oe = np.linalg.norm(orientation_error(quat, handle - pos))
        grip_factor = 0.15 if self.has_grasp else 1.0
        return (grip_factor * (self.w_pos * pe * 2 + self.w_ori * oe * 2)
                + self.w_hinge * float(hinge)
                + self._wrench_cost(f, t))

    def _rollout(self, state, seq, f0, t0):
        sim = self.sim
        sim.set_state(state)
        sim.forward()
        obs = self.env._get_observations()
        cost = self._stage_cost(obs, f0, t0)
        for a in seq:
            sim.data.ctrl[:self.act_dim] = a
            for _ in range(int(self.env.control_timestep)):
                sim.step()
            obs = self.env._get_observations()
            f, t = get_wrench(sim)
            cost += self._stage_cost(obs, f, t)
            if np.linalg.norm(f) > self.ft.f_hard or np.linalg.norm(t) > self.ft.t_hard:
                cost += self.w_hard * 1e2
                break
        return cost

    def act(self, obs):
        now = time.time()
        if self._touch.sense(self.sim):
            self.last_contact = now
            if not self.has_grasp:
                self.has_grasp = True
                self.grip_cmd = self.GRASP_LEVEL
        elif self.has_grasp and (now - self.last_contact) > self.CONTACT_TIMEOUT:
            self.has_grasp = False
            self.grip_cmd = self.PREGRASP_LEVEL
        eef_pos = obs["robot0_eef_pos"]
        handle = obs["handle_pos"]
        delta = handle - eef_pos
        dist = np.linalg.norm(delta)
        if dist > self.SERVO_DIST and not self.has_grasp:
            cmd = np.zeros(self.act_dim)
            cmd[:3] = np.clip(self.SERVO_GAIN * delta, ACT_LOW[:3], ACT_HIGH[:3])
            cmd[3:6] = np.clip(self.ORI_GAIN * orientation_error(obs["robot0_eef_quat"], delta),
                               ACT_LOW[3:6], ACT_HIGH[3:6])
            cmd[GRIP_INDEX] = -1.0 if dist <= self.APPROACH_DIST else self.PREGRASP_LEVEL
            self.grip_cmd = cmd[GRIP_INDEX]
            return cmd
        f0, t0 = get_wrench(self.sim)
        base = (np.zeros((self.horizon, self.act_dim))
                if self.best_seq is None
                else np.vstack([self.best_seq[1:], np.zeros((1, self.act_dim))]))
        if not self.has_grasp:
            step = (delta / max(dist, 1e-6)) * min(0.03, dist)
            base[:, :3] += step
            self.grip_cmd = self.PREGRASP_LEVEL
        else:
            base[:, :3] += np.array([0, 0.08, 0]) / self.horizon
        base[:, GRIP_INDEX] = self.grip_cmd
        seqs = self.rng.normal(base, self.noise, (self.n_samples, self.horizon, self.act_dim))
        seqs = np.clip(seqs, ACT_LOW, ACT_HIGH)
        st = copy.deepcopy(self.sim.get_state())
        costs = np.array([self._rollout(st, s, f0, t0) for s in seqs])
        idx = int(np.argmin(costs))
        self.best_seq = seqs[idx]
        return self.best_seq[0]

def run_episode(max_steps=None):
    obs = ENV.reset()
    ctrl = ForceTorqueMPC(ENV)
    THRESH = 0.150
    for i in range(max_steps or ENV.horizon):
        action = ctrl.act(obs)
        obs, _, done, _ = ENV.step(action)
        if obs["hinge_qpos"] > THRESH:
            break
        if done:
            break
    ENV.close()

if __name__ == "__main__":
    run_episode()
