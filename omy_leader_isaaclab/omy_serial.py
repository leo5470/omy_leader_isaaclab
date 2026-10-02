"""Minimal OMY-L100 reader on DynamixelSDK (Protocol 2.0, X-series), no lerobot dependency.

Motors: joint_1..3 XH540-W150 (ids 1-3), joint_4..6 XC330-T288 (ids 4-6), gripper XC330-T181 (id 7).
Arm motors are read torque-off in Extended Position mode; the gripper trigger is put in
current-based position mode with a small current limit so it springs back to ``gripper_open_pos``
(same behaviour as ``lerobot_teleoperator_omy``). Angles are returned in radians with the X-series
centre tick (2048) as zero for the arm. Gripper readings intentionally retain the
legacy publisher's HALF-angle units: (present - goal) * pi / 4095. They are zero
at rest and positive when squeezed; existing 10-degree mirror thresholds remain valid.
"""

from __future__ import annotations

import math
import threading
import time

ADDR_TORQUE_ENABLE = 64
ADDR_OPERATING_MODE = 11
ADDR_DRIVE_MODE = 10
ADDR_HOMING_OFFSET = 20
ADDR_CURRENT_LIMIT = 38
ADDR_GOAL_CURRENT = 102
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
MODE_EXTENDED_POSITION = 4
MODE_CURRENT_POSITION = 5
TICKS_PER_REV = 4096
CENTER = 2048
GRIPPER_RANGE_MAX = 4095
GRIPPER_DRIVE_MODE = 1
GRIPPER_HOMING_OFFSET = 100
TRIGGER_FIX_VERSION = 1

JOINT_IDS = {"joint_1": 1, "joint_2": 2, "joint_3": 3, "joint_4": 4, "joint_5": 5, "joint_6": 6, "gripper": 7}


def ticks_to_rad(ticks: int) -> float:
    return (ticks - CENTER) * (2.0 * math.pi / TICKS_PER_REV)


def gripper_goal_ticks(open_pos: float) -> int:
    """LeRobot RANGE_0_100; DynamixelMotorsBus.apply_drive_mode is False.

    Direction reversal happens in motor register Drive_Mode, not again in software.
    LeRobot truncates: 48.2 / 100 * 4095 = 1973.79 -> 1973 (not 3035).
    """
    if not math.isfinite(open_pos) or not 0 <= open_pos <= 100:
        raise ValueError("gripper_open_pos must be finite and in [0, 100]")
    return int(open_pos / 100 * GRIPPER_RANGE_MAX)


class OmySerialLeader:
    def __init__(
        self,
        port: str = "/dev/robotis_left",
        baudrate: int = 4_000_000,
        gripper_spring: bool = True,
        gripper_open_ticks: int | None = None,
        gripper_current_limit: int = 100,
        *,
        gripper_open_pos: float = 48.2,
    ):
        from dynamixel_sdk import GroupSyncRead, PacketHandler, PortHandler  # noqa: PLC0415

        goal = gripper_goal_ticks(gripper_open_pos)
        if gripper_open_ticks is not None:
            if not isinstance(gripper_open_ticks, int) or not 0 <= gripper_open_ticks <= GRIPPER_RANGE_MAX:
                raise ValueError("gripper_open_ticks must be an integer in [0, 4095]")
            goal = gripper_open_ticks
        self._gripper_open_ticks = goal
        self._lock = threading.RLock()
        self._closed = False
        self._port = PortHandler(port)
        if not self._port.openPort():
            raise ConnectionError(f"cannot open {port}")
        if not self._port.setBaudRate(baudrate):
            self._port.closePort()
            raise ConnectionError(f"cannot set baudrate {baudrate} on {port}")
        self._ph = PacketHandler(2.0)
        self._ids = list(JOINT_IDS.values())
        self._reader = GroupSyncRead(self._port, self._ph, ADDR_PRESENT_POSITION, 4)
        for i in self._ids:
            self._reader.addParam(i)
        try:
            self._configure(gripper_spring, gripper_current_limit)
        except BaseException:
            try:
                self.close()
            except Exception:
                pass
            raise

    def _configure(self, gripper_spring, gripper_current_limit):
        for i in self._ids:
            self._write1(i, ADDR_TORQUE_ENABLE, 0)
        for name, i in JOINT_IDS.items():
            if name != "gripper":
                self._write1(i, ADDR_OPERATING_MODE, MODE_EXTENDED_POSITION)
        g = JOINT_IDS["gripper"]
        # EEPROM writes require torque off. Avoid rewriting unchanged EEPROM.
        if self._read1(g, ADDR_DRIVE_MODE) != GRIPPER_DRIVE_MODE:
            self._write1(g, ADDR_DRIVE_MODE, GRIPPER_DRIVE_MODE)
        if self._read4(g, ADDR_HOMING_OFFSET) != GRIPPER_HOMING_OFFSET:
            self._write4(g, ADDR_HOMING_OFFSET, GRIPPER_HOMING_OFFSET)
        if gripper_spring:
            self._write1(g, ADDR_OPERATING_MODE, MODE_CURRENT_POSITION)
            self._write2(g, ADDR_CURRENT_LIMIT, gripper_current_limit)
            self._write2(g, ADDR_GOAL_CURRENT, gripper_current_limit)
            # Set the goal before enabling torque so no old goal can pull the trigger.
            self._write4(g, ADDR_GOAL_POSITION, self._gripper_open_ticks)
            self._write1(g, ADDR_TORQUE_ENABLE, 1)
            # XC330 can reset Goal_Position to Present_Position when torque is
            # enabled. Reassert and read back the requested spring rest afterward.
            self._write4(g, ADDR_GOAL_POSITION, self._gripper_open_ticks)
            if self._read4(g, ADDR_GOAL_POSITION) != self._gripper_open_ticks:
                raise ConnectionError("gripper goal readback does not match requested rest")

    # ---- packet helpers ------------------------------------------------------------
    def _check(self, res, err, what):
        if res != 0 or err != 0:
            raise ConnectionError(f"{what}: {self._ph.getTxRxResult(res)} {self._ph.getRxPacketError(err)}")

    def _write1(self, i, addr, v):
        self._check(*self._ph.write1ByteTxRx(self._port, i, addr, v), f"write1 id{i}@{addr}")

    def _write2(self, i, addr, v):
        self._check(*self._ph.write2ByteTxRx(self._port, i, addr, v), f"write2 id{i}@{addr}")

    def _write4(self, i, addr, v):
        self._check(*self._ph.write4ByteTxRx(self._port, i, addr, v), f"write4 id{i}@{addr}")

    def _read1(self, i, addr):
        value, result, error = self._ph.read1ByteTxRx(self._port, i, addr)
        self._check(result, error, f"read1 id{i}@{addr}")
        return value

    def _read4(self, i, addr):
        value, result, error = self._ph.read4ByteTxRx(self._port, i, addr)
        self._check(result, error, f"read4 id{i}@{addr}")
        return value - 2**32 if value >= 2**31 else value

    def gripper_status(self):
        """Read back the configuration and position for hardware verification."""
        with self._lock:
            g = JOINT_IDS["gripper"]
            return {
                "drive_mode": self._read1(g, ADDR_DRIVE_MODE),
                "homing_offset": self._read4(g, ADDR_HOMING_OFFSET),
                "operating_mode": self._read1(g, ADDR_OPERATING_MODE),
                "torque": self._read1(g, ADDR_TORQUE_ENABLE),
                "goal_ticks": self._read4(g, ADDR_GOAL_POSITION),
                "present_ticks": self._read4(g, ADDR_PRESENT_POSITION),
            }

    # ---- public -----------------------------------------------------------------
    def read(self) -> dict[str, float]:
        """Arm radians; gripper legacy half-angle units (0 released, + squeezed)."""
        with self._lock:
            if self._closed:
                raise ConnectionError("leader is closed")
            return self._read_positions()

    def _read_positions(self):
        if self._reader.txRxPacket() != 0:
            raise ConnectionError("sync read failed")
        out = {}
        for name, i in JOINT_IDS.items():
            v = self._reader.getData(i, ADDR_PRESENT_POSITION, 4)
            if v >= 2**31:  # signed 32-bit (extended position)
                v -= 2**32
            out[f"{name}.pos"] = (
                (v - self._gripper_open_ticks) * math.pi / GRIPPER_RANGE_MAX
                if name == "gripper" else ticks_to_rad(v)
            )
        return out

    def close(self):
        with self._lock:
            if self._closed:
                return
            try:
                self._write1(JOINT_IDS["gripper"], ADDR_TORQUE_ENABLE, 0)
            finally:
                self._port.closePort()
                self._closed = True


class SerialClient:
    """Background reader with the same ``latest()`` API as :class:`link.FrameClient`, so calibration
    and monitor tools can read the leader directly on the sim machine (the common single-PC setup)."""

    def __init__(self, port: str = "/dev/robotis_left", baudrate: int = 4_000_000, hz: float = 100.0,
                 gripper_open_pos: float = 48.2):
        import threading

        self._leader = OmySerialLeader(port, baudrate, gripper_open_pos=gripper_open_pos)
        self._buf: tuple[dict | None, float] = (None, 0.0)
        self._stop = threading.Event()
        self._period = 1.0 / hz
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            try:
                self._buf = (self._leader.read(), time.time())
            except Exception as e:  # noqa: BLE001
                print(f"[omy_serial] read error: {e}")
                time.sleep(0.5)
            time.sleep(self._period)

    def latest(self):
        a, t = self._buf
        return (None, float("inf")) if a is None else (a, time.time() - t)

    def close(self):
        self._stop.set()
        self._thread.join()
        self._leader.close()


def open_reader(source: str = "serial", port: str = "/dev/robotis_left", baudrate: int = 4_000_000,
                tcp_host: str = "127.0.0.1", tcp_port: int = 5555, gripper_open_pos: float = 48.2):
    """Return an object with ``latest()`` for either the local serial leader or a remote publisher."""
    if source == "tcp":
        from .link import FrameClient

        return FrameClient(tcp_host, tcp_port)
    return SerialClient(port, baudrate, gripper_open_pos=gripper_open_pos)


if __name__ == "__main__":  # quick check: python -m omy_leader_isaaclab.omy_serial /dev/robotis_left
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("port", nargs="?", default="/dev/robotis_left")
    parser.add_argument("--gripper-open", type=float, default=48.2)
    parser.add_argument("--seconds", type=float, default=0, help="0: run until Ctrl+C")
    args = parser.parse_args()
    l = OmySerialLeader(args.port, gripper_open_pos=args.gripper_open)
    try:
        print("[gripper]", l.gripper_status(), flush=True)
        print("Gripper units: legacy half-angle; release = 0, squeeze = positive", flush=True)
        started = time.monotonic()
        while not args.seconds or time.monotonic() - started < args.seconds:
            a = l.read()
            print("  ".join(f"{k[:-4]}={math.degrees(v):+7.2f}" for k, v in a.items()), flush=True)
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        l.close()
