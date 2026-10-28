# Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.

"""Same as play_mujoco.py, but records signals to a .npz file for later plotting.

Usage:
    python scripts/sim2sim/play_mujoco_record.py --config configs/policy_biped_50hz.yaml \
        --signals joint_pos,joint_vel,base_pos --out logs/sim/walk.npz

Stop with Ctrl+C or by closing the viewer window; the data is saved on exit. Plot with plot_signals.py.
To record something not in the built-in list, add it in `add_custom_signals` below.
"""

import argparse
import signal
import sys
import threading
import time

import numpy as np
import torch

from berkeley_humanoid_lite_lowlevel.policy.rl_controller import RlController
from berkeley_humanoid_lite.environments import MujocoSimulator, Cfg

from signal_recorder import BUILTIN_SIGNALS, SignalRecorder


# Parse recorder options, then leave the remaining args (e.g. --config) for Cfg
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--signals", type=str, default="joint_pos,joint_vel,joint_torque,base_pos",
                    help=f"Comma-separated built-in signals. Available: {','.join(BUILTIN_SIGNALS)}")
parser.add_argument("--out", type=str, default=None,
                    help="Output .npz path (default: logs/sim/run_<timestamp>.npz)")
parser.add_argument("--every", type=int, default=1, help="Record every N policy steps")
rec_args, remaining = parser.parse_known_args()
sys.argv = [sys.argv[0]] + remaining

cfg = Cfg.from_arguments()

if not cfg:
    raise ValueError("Failed to load config.")


def add_custom_signals(rec: SignalRecorder):
    """Add your own signals here. Each is fn(model, data) -> number or array."""
    rec.add("base_height", lambda m, d: d.qpos[2], labels=["z"])


def main():
    """Main execution function for the MuJoCo simulation environment."""
    robot = MujocoSimulator(cfg)
    obs = robot.reset()

    controller = RlController(cfg)
    controller.load_policy()

    default_actions = np.array(cfg.default_joint_positions, dtype=np.float32)[robot.cfg.action_indices]

    rec = SignalRecorder(robot.mj_model, robot.mj_data)
    for name in filter(None, rec_args.signals.split(",")):
        rec.add_builtin(name.strip())
    add_custom_signals(rec)

    out_path = rec_args.out or f"logs/sim/run_{time.strftime('%Y%m%d_%H%M%S')}.npz"
    print(f"Recording to {out_path} (Ctrl+C or close the window to stop and save)")

    # Ctrl+C only sets a flag, so the loop stops between steps instead of
    # being interrupted halfway through one
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())

    try:
        # Also stops when the viewer window is closed
        while not stop.is_set() and robot.mj_viewer.is_running():
            actions = controller.update(obs.numpy())

            if actions is None:
                actions = default_actions

            actions = torch.tensor(actions)
            obs = robot.step(actions)

            if robot.n_steps % rec_args.every == 0:
                rec.record()
    finally:
        rec.save(out_path)
        # Close the viewer and wait for its render thread to finish; exiting
        # while it is still drawing crashes (segfault / GLXBadDrawable)
        robot.mj_viewer.close()
        deadline = time.time() + 3.0
        while robot.mj_viewer.is_running() and time.time() < deadline:
            time.sleep(0.05)


if __name__ == "__main__":
    main()
