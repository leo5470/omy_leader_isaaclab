"""L100 -> Franka calibration, lerobot-style. Reads the L100 directly (default) or a publisher.py stream (--source tcp).

    omy-leader-calib                # writes ./omy_calib.json   (or python -m omy_leader_isaaclab.franka_calib)

Two steps, Enter each time:

  1. RANGE   press Enter, then move EVERY joint to both of its limits (and squeeze the gripper
             fully, then release). A live table shows min / max per joint. Press Enter when done.
  2. READY   put the L100 in the Franka ready pose (what you'd do to imitate the robot's default
             pose on screen), hold still, press Enter.

Signs are NOT determined here: the file starts with the defaults (joint_2 inverted, measured on
2026-09-19); after teleop, report any joint that moves the wrong way and flip it in the JSON.
Output is consumed by teleop.py --calib omy_calib.json.
"""

from __future__ import annotations

if __package__ in (None, ""):  # allow `python omy_leader_isaaclab/<file>.py` as well as `python -m omy_leader_isaaclab.<file>`
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    __package__ = "omy_leader_isaaclab"

import argparse
import json
import math
import threading
import time
from dataclasses import replace

import numpy as np

from .franka_config import DEFAULT_OMY_TO_FRANKA_CONFIG, FRANKA_HOME
from .link import DEFAULT_PORT, FrameClient  # noqa: F401
from .omy_serial import open_reader
from .franka_retarget import OMY_JOINTS, OmyToFrankaRetarget

NAMES = (*OMY_JOINTS, "gripper")
# Known from the 2026-09-19 manual test: j1 forward-correct, j2 reads inverted vs URDF.
DEFAULT_SIGNS = {"joint_1": 1.0, "joint_2": -1.0, "joint_3": 1.0, "joint_4": 1.0, "joint_5": 1.0, "joint_6": 1.0}


def _wait_enter(msg: str) -> None:
    input(f"\n>>> {msg}\n    按 Enter 繼續: ")


def _read(client: FrameClient, n: int = 20) -> np.ndarray:
    vals = []
    t0 = time.time()
    while len(vals) < n and time.time() - t0 < 3.0:
        a, age = client.latest()
        if a is not None and age < 0.5:
            vals.append([a[f"{k}.pos"] for k in NAMES])
        time.sleep(0.01)
    if not vals:
        raise SystemExit("no L100 data - check --port (serial) or that publisher.py is streaming (--source tcp)")
    return np.mean(vals, axis=0)


def _record_ranges(client: FrameClient) -> tuple[np.ndarray, np.ndarray]:
    """Track min/max of every joint until the user presses Enter; live table like lerobot's."""
    stop = threading.Event()
    threading.Thread(target=lambda: (input(), stop.set()), daemon=True).start()
    lo = np.full(7, np.inf)
    hi = np.full(7, -np.inf)
    t_print = 0.0
    while not stop.is_set():
        a, age = client.latest()
        if a is not None and age < 0.5:
            v = np.array([a[f"{k}.pos"] for k in NAMES])
            lo = np.minimum(lo, v)
            hi = np.maximum(hi, v)
        if time.time() - t_print > 0.2:
            t_print = time.time()
            cells = "  ".join(f"{k[-1] if k != 'gripper' else 'g'}:[{math.degrees(l):+6.0f},{math.degrees(h):+6.0f}]" for k, l, h in zip(NAMES, lo, hi))
            print(f"\r    min/max (deg)  {cells}   ", end="", flush=True)
        time.sleep(0.02)
    print()
    return lo, hi


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["serial", "tcp"], default="serial",
                    help="serial = L100 on this machine (default); tcp = publisher.py on another host")
    ap.add_argument("--port", default="/dev/robotis_left", help="serial device (source=serial)")
    ap.add_argument("--baudrate", type=int, default=4_000_000)
    ap.add_argument("--gripper-open", type=float, default=48.2)
    ap.add_argument("--tcp-host", default="127.0.0.1")
    ap.add_argument("--tcp-port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--out", default="omy_calib.json")
    ap.add_argument("--gripper-only", action="store_true", help="redo only the gripper (squeeze / release) and update --out in place")
    ap.add_argument("--center", default="joint_1,joint_5", help="joints whose zero is the range midpoint instead of the READY reading (left/right symmetric)")
    args = ap.parse_args()

    client = open_reader(args.source, args.port, args.baudrate, args.tcp_host, args.tcp_port,
                         gripper_open_pos=args.gripper_open)
    try:
        _calibrate(args, client)
    finally:
        client.close()


def _calibrate(args, client):
    print("[calib] connecting to publisher ...")
    _read(client)
    print("[calib] OK, reading L100.")
    deg = lambda v: f"{math.degrees(v):+7.1f}°"  # noqa: E731

    if args.gripper_only:
        with open(args.out) as f:
            calib = json.load(f)
        _wait_enter("夾爪：【捏到底】不要放")
        g_closed = float(_read(client)[6])
        _wait_enter("夾爪：【放開】")
        g_open = float(_read(client)[6])
        print(f"    gripper open={deg(g_open)} closed={deg(g_closed)} travel={deg(g_closed - g_open)}")
        if abs(g_closed - g_open) < math.radians(2.0):
            raise SystemExit("    !! 行程 < 2°，沒讀到夾爪動作，檔案不更新")
        calib.update(gripper_open_rad=g_open, gripper_closed_rad=g_closed)
        with open(args.out, "w") as f:
            json.dump(calib, f, indent=2)
        print(f"[calib] gripper updated in {args.out}")
        return

    # ---- 1. range of motion ---------------------------------------------------------
    _wait_enter("步驟 1/2  範圍：按 Enter 後，把【每一個關節】都轉到兩端極限，夾爪也捏到底再放開。全部轉過後再按一次 Enter 結束")
    lo, hi = _record_ranges(client)
    for k, l, h in zip(NAMES, lo, hi):
        flag = "   !! 幾乎沒動" if h - l < math.radians(5) else ""
        print(f"    {k:8s} min {deg(l)}  max {deg(h)}  span {deg(h - l)}{flag}")

    # ---- 2. ready pose ---------------------------------------------------------------
    _wait_enter("步驟 2/2  READY：把 L100 擺成你要模仿 sim 裡 Franka 預設姿勢的樣子（夾爪放開），停住")
    rest = _read(client)
    print("    ready 讀值:", "  ".join(f"{k}={deg(v)}" for k, v in zip(NAMES, rest)))

    # ---- derived ------------------------------------------------------------------
    g_open = float(rest[6])  # released
    g_closed = float(lo[6] if abs(lo[6] - g_open) > abs(hi[6] - g_open) else hi[6])  # the far end
    signs = dict(DEFAULT_SIGNS)
    cfg = replace(DEFAULT_OMY_TO_FRANKA_CONFIG, omy_sign=signs, gripper_open_rad=g_open, gripper_closed_rad=g_closed)
    rest_action = {f"{k}.pos": float(v) for k, v in zip(NAMES, rest)}
    cfg = OmyToFrankaRetarget(cfg).auto_offset(rest_action, FRANKA_HOME)
    # left/right joints: zero at the midpoint of the measured range so both directions get equal travel
    zero = dict(cfg.omy_zero_rad)
    for k in [s.strip() for s in args.center.split(",") if s.strip()]:
        i = NAMES.index(k)
        mid = float((lo[i] + hi[i]) / 2.0)
        print(f"    {k}: zero READY {deg(zero[k])} -> range centre {deg(mid)} (READY now maps {deg(signs[k] * (rest[i] - mid))} off home)")
        zero[k] = mid
    cfg = replace(cfg, omy_zero_rad=zero)
    check = OmyToFrankaRetarget(cfg).raw_target(rest_action)

    calib = {
        "omy_sign": signs,
        "omy_zero_rad": cfg.omy_zero_rad,
        "gripper_closed_rad": g_closed,
        "gripper_open_rad": g_open,
        "range_min_rad": {k: float(v) for k, v in zip(NAMES, lo)},
        "range_max_rad": {k: float(v) for k, v in zip(NAMES, hi)},
        "ready_reading_rad": {k: float(v) for k, v in zip(NAMES, rest)},
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "omy_sign: flip a joint to -1 if it moves the wrong way in the sim, then restart teleop",
    }
    with open(args.out, "w") as f:
        json.dump(calib, f, indent=2)
    print(f"\n[calib] gripper open={deg(g_open)} closed={deg(g_closed)}")
    print("[calib] signs (defaults, adjust after teleop):", {k: int(v) for k, v in signs.items()})
    print("[calib] READY -> Franka:", np.round(check, 3), " target", np.round(FRANKA_HOME, 3))
    print(f"[calib] saved {args.out}")


if __name__ == "__main__":
    main()
