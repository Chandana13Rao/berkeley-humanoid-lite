"""Record any quantity from a running MuJoCo simulation and save it for plotting.

Quick start
-----------
    rec = SignalRecorder(robot.mj_model, robot.mj_data)
    rec.add("joint_pos", joint_pos, labels=joint_names(robot.mj_model))
    rec.add("base_height", lambda m, d: d.qpos[2])

    while running:
        robot.step(actions)
        rec.record()

    rec.save("logs/run.npz")

Adding a new signal
-------------------
A signal is any function ``fn(model, data) -> number or array``. ``model`` is the
``mujoco.MjModel`` (fixed robot description) and ``data`` is the ``mujoco.MjData``
(the current state). Write the function, then call ``rec.add(name, fn)``.

Useful places in ``data``:
    data.time          simulation time [s]
    data.qpos          generalized positions (base pos 0:3, base quat 3:7, joints 7:)
    data.qvel          generalized velocities (base lin 0:3, base ang 3:6, joints 6:)
    data.ctrl          actuator commands (torques for this robot)
    data.xpos[i]       world position of body i
    data.cfrc_ext[i]   external (contact) wrench on body i: [torque(3), force(3)]
    data.sensor(name)  a named sensor from the MJCF, e.g. data.sensor("imu_acc").data

Examples:
    # Contact force on the left foot
    foot = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "leg_left_ankle_roll")
    rec.add("left_foot_force", lambda m, d: d.cfrc_ext[foot, 3:6], labels=["fx", "fy", "fz"])

    # IMU accelerometer sensor
    rec.add("imu_acc", lambda m, d: d.sensor("imu_acc").data, labels=["ax", "ay", "az"])
"""

from pathlib import Path

import mujoco
import numpy as np


# ---------------------------------------------------------------------------
# Name helpers (used for plot legends)
# ---------------------------------------------------------------------------

def joint_names(model: mujoco.MjModel) -> list[str]:
    """Names of the actuated joints, in actuator order."""
    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in model.actuator_trnid[:, 0]]


def _actuated_qpos_adr(model):
    return model.jnt_qposadr[model.actuator_trnid[:, 0]]


def _actuated_dof_adr(model):
    return model.jnt_dofadr[model.actuator_trnid[:, 0]]


# ---------------------------------------------------------------------------
# Ready-made signals. Each is a plain fn(model, data); copy one to make your own.
# ---------------------------------------------------------------------------

def joint_pos(model, data):
    """Actuated joint angles [rad]."""
    return data.qpos[_actuated_qpos_adr(model)]


def joint_vel(model, data):
    """Actuated joint velocities [rad/s]."""
    return data.qvel[_actuated_dof_adr(model)]


def joint_torque(model, data):
    """Commanded actuator torques [Nm] (after clipping)."""
    return data.ctrl


def base_pos(model, data):
    """Base position in world frame [m]."""
    return data.qpos[0:3]


def base_quat(model, data):
    """Base orientation quaternion [w, x, y, z]."""
    return data.qpos[3:7]


def base_lin_vel(model, data):
    """Base linear velocity in world frame [m/s]."""
    return data.qvel[0:3]


def base_ang_vel(model, data):
    """Base angular velocity in body frame [rad/s]."""
    return data.qvel[3:6]


# name -> (function, labels or a function that builds labels from the model)
BUILTIN_SIGNALS = {
    "joint_pos": (joint_pos, joint_names),
    "joint_vel": (joint_vel, joint_names),
    "joint_torque": (joint_torque, joint_names),
    "base_pos": (base_pos, ["x", "y", "z"]),
    "base_quat": (base_quat, ["w", "x", "y", "z"]),
    "base_lin_vel": (base_lin_vel, ["vx", "vy", "vz"]),
    "base_ang_vel": (base_ang_vel, ["wx", "wy", "wz"]),
}


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------

class SignalRecorder:
    """Collects named signals from a MuJoCo model/data pair over time."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self._fns = {}
        self._labels = {}
        self._time = []
        self._values = {}

    def add(self, name: str, fn, labels=None):
        """Register a signal. ``labels`` names each channel (used in plot legends)."""
        if callable(labels):
            labels = labels(self.model)
        self._fns[name] = fn
        self._labels[name] = list(labels) if labels is not None else None
        self._values[name] = []

    def add_builtin(self, name: str):
        """Register one of the signals in ``BUILTIN_SIGNALS`` by name."""
        if name not in BUILTIN_SIGNALS:
            raise KeyError(f"Unknown signal '{name}'. Available: {', '.join(BUILTIN_SIGNALS)}")
        fn, labels = BUILTIN_SIGNALS[name]
        self.add(name, fn, labels)

    def record(self):
        """Sample every registered signal at the current simulation time."""
        self._time.append(self.data.time)
        for name, fn in self._fns.items():
            # np.array copies, so later sim steps don't overwrite stored values
            self._values[name].append(np.array(fn(self.model, self.data), dtype=np.float64))

    def __len__(self):
        return len(self._time)

    def save(self, path):
        """Write everything to a .npz file. Each signal is stored as a (T, channels) array."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"time": np.array(self._time)}
        for name, values in self._values.items():
            arr = np.array(values)
            if arr.ndim == 1:
                arr = arr[:, None]
            arrays[name] = arr
            labels = self._labels[name] or [f"{name}[{i}]" for i in range(arr.shape[1])]
            arrays[f"{name}__labels"] = np.array(labels)
        np.savez(path, **arrays)
        print(f"Saved {len(self)} samples of {list(self._values)} to {path}")
