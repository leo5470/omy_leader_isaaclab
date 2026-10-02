"""Run on the host the L100 is plugged into (when that is NOT the sim host). Streams joint radians over TCP.

    python -m omy_leader_isaaclab.publisher --port /dev/robotis_left            # DynamixelSDK reader (default)
    python -m omy_leader_isaaclab.publisher --reader lerobot --port ...           # via lerobot_teleoperator_omy
    python -m omy_leader_isaaclab.publisher --print                              # just print readings

Then open a reverse tunnel so the sim host can reach us:   ssh -N -R 5555:localhost:5555 <sim host>
On the sim host run teleop.py / record_demos with source=tcp.
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

from .link import DEFAULT_PORT, FrameServer, pack


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/robotis_left", help="OMY U2D2 serial device")
    ap.add_argument("--baudrate", type=int, default=4_000_000)
    ap.add_argument("--fps", type=float, default=100.0)
    ap.add_argument("--tcp-host", default="127.0.0.1")
    ap.add_argument("--tcp-port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--print", action="store_true", help="print readings (deg) at 5 Hz, no TCP")
    ap.add_argument("--reader", choices=["serial", "lerobot"], default="serial",
                    help="serial = DynamixelSDK directly (default); lerobot = charlie8612/lerobot_teleoperator_omy plugin")
    ap.add_argument(
        "--units", choices=["normalized", "rad"], default="normalized",
        help="what the plugin's get_action() returns: 'normalized' = lerobot RANGE_M100_100 over one "
             "revolution (current omy_leader), 'rad' = already radians (older plugin builds)",
    )
    ap.add_argument(
        "--gripper-open", type=float, default=48.2,
        help="plugin gripper goal/rest position (normalized 0..100); the trigger springs back here and it "
             "reads as 0 rad. 60 = plugin default; 48.2 = the more-open rest the operator asked for",
    )
    args = ap.parse_args()

    if args.reader == "lerobot":
        from . import omy_motor_tables  # noqa: PLC0415

        added = omy_motor_tables.register()
        if added:
            print(f"[publisher] registered missing Dynamixel models in lerobot tables: {added}")
        from lerobot_teleoperator_omy import OmyLeader, OmyLeaderConfig  # noqa: PLC0415

        leader = OmyLeader(OmyLeaderConfig(port=args.port, baudrate=args.baudrate, gripper_open_pos=args.gripper_open))
        leader.connect()
        get_action, disconnect = leader.get_action, leader.disconnect
    else:
        from .omy_serial import OmySerialLeader  # noqa: PLC0415

        leader = OmySerialLeader(args.port, args.baudrate, gripper_open_pos=args.gripper_open)
        get_action, disconnect = leader.read, leader.close
        args.units = "rad"  # arm radians; gripper uses legacy half-angle units, 0 = released
    server = None if args.print else FrameServer(args.tcp_host, args.tcp_port)
    if server:
        print(f"[publisher] listening on {args.tcp_host}:{args.tcp_port}, streaming at {args.fps:.0f} Hz")

    # lerobot RANGE_M100_100 normalizes 0..4095 ticks (one revolution, centre 2048) to -100..100,
    # so 200 units == 2*pi. Extended-position multi-turn readings simply exceed +/-100 and stay linear.
    k = math.pi / 100.0

    def to_rad(a: dict[str, float]) -> dict[str, float]:
        if args.units == "rad":
            return a
        out = {key: v * k for key, v in a.items() if key != "gripper.pos"}
        # RANGE_0_100: legacy half-angle units, intentionally matching the direct reader.
        out["gripper.pos"] = (a["gripper.pos"] - args.gripper_open) * k  # 0 = released
        return out

    period = 1.0 / args.fps
    seq = 0
    t_next = time.perf_counter()
    t_report = time.time()
    try:
        while True:
            a = to_rad(get_action())
            if server:
                server.send(pack(seq, a))
            seq += 1
            if args.print or time.time() - t_report > 5.0:
                deg = "  ".join(f"{k[:-4]}={math.degrees(v):+7.1f}" for k, v in a.items())
                extra = f"  clients={server.n_clients}" if server else ""
                print(f"[{seq:7d}] {deg}{extra}")
                t_report = time.time()
                if args.print:
                    time.sleep(0.2)
                    continue
            t_next += period
            dt = t_next - time.perf_counter()
            if dt > 0:
                time.sleep(dt)
            else:
                t_next = time.perf_counter()
    except KeyboardInterrupt:
        pass
    finally:
        disconnect()


if __name__ == "__main__":
    main()
