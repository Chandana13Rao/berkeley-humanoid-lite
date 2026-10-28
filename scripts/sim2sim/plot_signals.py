"""Plot signals recorded by play_mujoco_record.py.

Usage:
    python scripts/sim2sim/plot_signals.py logs/sim/walk.npz
    python scripts/sim2sim/plot_signals.py logs/sim/walk.npz --signals joint_pos --channels knee
    python scripts/sim2sim/plot_signals.py logs/sim/walk.npz --csv logs/sim/walk.csv
"""

import argparse
from pathlib import Path

import numpy as np


def load(path):
    """Return (time, {name: (values (T, C), labels)}) from a recorder .npz file."""
    f = np.load(path)
    signals = {}
    for key in f.files:
        if key == "time" or key.endswith("__labels"):
            continue
        signals[key] = (f[key], [str(s) for s in f[f"{key}__labels"]])
    return f["time"], signals


def main():
    parser = argparse.ArgumentParser(description="Plot recorded MuJoCo signals")
    parser.add_argument("file", help="Recorded .npz file")
    parser.add_argument("--signals", type=str, default=None, help="Comma-separated signals to plot (default: all)")
    parser.add_argument("--channels", type=str, default=None,
                        help="Comma-separated substrings; only channels whose label contains one are plotted")
    parser.add_argument("--csv", type=str, default=None, help="Also export all data to this CSV file")
    parser.add_argument("--save", type=str, default=None, help="Save the figure to this image file instead of showing it")
    args = parser.parse_args()

    time, signals = load(args.file)
    print(f"{len(time)} samples, {time[0]:.2f}s to {time[-1]:.2f}s" if len(time) else "No samples")
    for name, (values, labels) in signals.items():
        print(f"  {name}: {values.shape}  {labels}")

    if args.csv:
        columns = [time[:, None]] + [values for values, _ in signals.values()]
        header = ["time"] + [f"{name}/{label}" for name, (_, labels) in signals.items() for label in labels]
        np.savetxt(args.csv, np.hstack(columns), delimiter=",", header=",".join(header), comments="")
        print(f"Wrote {args.csv}")

    names = args.signals.split(",") if args.signals else list(signals)
    filters = args.channels.split(",") if args.channels else None

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(names), 1, sharex=True, squeeze=False, figsize=(10, 3 * len(names)))
    for ax, name in zip(axes[:, 0], names):
        values, labels = signals[name]
        for i, label in enumerate(labels):
            if filters and not any(f in label for f in filters):
                continue
            ax.plot(time, values[:, i], label=label)
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize="small", loc="upper left", bbox_to_anchor=(1.01, 1.0))
    axes[-1, 0].set_xlabel("time [s]")
    fig.tight_layout()

    # Without a GUI toolkit (e.g. no python3-tk), matplotlib can't open a window
    save_path = args.save
    if not save_path and plt.get_backend().lower() == "agg":
        save_path = str(Path(args.file).with_suffix(".png"))
        print("No GUI backend available (try: sudo apt install python3-tk); saving instead.")

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved figure to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
