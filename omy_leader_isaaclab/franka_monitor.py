"""Live monitor of the L100 leader: raw plugin angles and, with --calib, the mapped Franka joints.

Reads the L100 directly (default) or a publisher.py stream (--source tcp):

    omy-leader-monitor --calib omy_calib.json

Refreshes in place at 10 Hz; Ctrl-C to quit. All angles in degrees.
  raw   : what the plugin reports (after the publisher's rad conversion), before sign/zero
  franka: J1..J7 the retargeter would command (J3 is always 0) and gripper open/close
"""

from __future__ import annotations

if __package__ in (None, ""):  # allow `python omy_leader_isaaclab/<file>.py` as well as `python -m omy_leader_isaaclab.<file>`
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    __package__ = "omy_leader_isaaclab"

import argparse
import math
import time

import numpy as np

from .franka_config import DEFAULT_OMY_TO_FRANKA_CONFIG, load_calib
from .link import DEFAULT_PORT, FrameClient  # noqa: F401
from .omy_serial import open_reader
from .franka_retarget import OMY_JOINTS, OmyToFrankaRetarget


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["serial", "tcp"], default="serial",
                    help="serial = L100 on this machine (default); tcp = publisher.py on another host")
    ap.add_argument("--port", default="/dev/robotis_left", help="serial device (source=serial)")
    ap.add_argument("--baudrate", type=int, default=4_000_000)
    ap.add_argument("--gripper-open", type=float, default=48.2)
    ap.add_argument("--tcp-host", default="127.0.0.1")
    ap.add_argument("--tcp-port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--calib", default="", help="omy_calib.json; if given, also show mapped Franka joints")
    ap.add_argument("--hz", type=float, default=10.0)
    args = ap.parse_args()

    cfg = load_calib(args.calib) if args.calib else DEFAULT_OMY_TO_FRANKA_CONFIG
    rt = OmyToFrankaRetarget(cfg)
    client = open_reader(args.source, args.port, args.baudrate, args.tcp_host, args.tcp_port,
                         gripper_open_pos=args.gripper_open)
    d = math.degrees
    print("waiting for stream ...")
    try:
        while True:
            a, age = client.latest()
            if a is None or age > 0.5:
                print("\r  no data (stream stale)                                                    ", end="", flush=True)
                time.sleep(0.2)
                continue
            raw = "  ".join(f"j{i + 1}={d(a[f'{k}.pos']):+7.1f}" for i, k in enumerate(OMY_JOINTS))
            out = f"L100 raw (deg)   {raw}  grip={d(a['gripper.pos']):+6.1f}\n"
            if args.calib:
                q = rt.raw_target(a)
                fr = "  ".join(f"J{i + 1}={d(v):+7.1f}" for i, v in enumerate(q))
                g = "CLOSE" if rt.gripper_command(a["gripper.pos"]) < 0 else "open"
                out += f"Franka cmd (deg) {fr}  grip={g}\n"
            print("\033[2J\033[H" + out + "\n(Ctrl-C 離開)", end="", flush=True)
            time.sleep(1.0 / args.hz)
    except KeyboardInterrupt:
        print("\n")
    finally:
        client.close()


if __name__ == "__main__":
    main()
