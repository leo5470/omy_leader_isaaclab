"""Isaac Lab teleop device for the ROBOTIS OMY-L100 leader arm (targets: OMY 1:1 or Franka).

Plugs into Isaac Lab's device factory (``create_teleop_device``), so any script that builds its
device from ``env_cfg.teleop_devices`` (cyclo_lab's ``record_demos.py``, this repo's ``teleop.py``)
can use it by name:

    env_cfg.teleop_devices = DevicesCfg(devices={"omy_leader": OmyLeaderCfg(target="omy", source="serial")})
    record_demos.py --task <task> --teleop_device omy_leader

Output
    target="omy"    7-vector  [joint1..joint6 (rad), gripper +1 open / -1 close]     (mirror.py)
    target="franka" 8-vector  [J1..J7 (rad, J3 = 0), gripper +1 / -1]               (franka_retarget.py)
for envs whose arm action is an absolute ``JointPositionAction`` and gripper a
``BinaryJointPositionAction`` (see omy_env_cfg.py / franka_env_cfg.py).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch

from isaaclab.devices import DeviceBase, DeviceCfg


@dataclass
class OmyLeaderCfg(DeviceCfg):
    target: str = "omy"  # "omy" (same kinematics, 1:1) | "franka" (J3 locked, wrist remapped)
    source: str = "serial"  # "serial" (DynamixelSDK on this host) | "tcp" (publisher.py on another host)
    port: str = "/dev/robotis_left"
    baudrate: int = 4_000_000
    gripper_open_pos: float = 48.2  # plugin RANGE_0_100 spring rest position
    tcp_host: str = "127.0.0.1"
    tcp_port: int = 5555
    calib_path: str = ""  # omy_calib.json (signs / zeros / gripper / wrist mode); "" = defaults
    # when the stream comes up: hold the leader in the target's default pose, zeros are computed
    auto_zero: bool = False
    auto_zero_settle_s: float = 3.0
    stale_s: float = 0.5  # hold the last command if no fresh reading for this long
    dt: float = 1.0 / 50.0  # env control period, for the rate limiter
    vel_scale: float = 1.0
    hz: float = 100.0  # serial read rate
    callbacks: dict = field(default_factory=dict)
    # Isaac Lab >= 3.0 resolves the device class from this field (DeviceCfg declares it as a
    # dataclass field, so a class attribute would be shadowed); older versions use DEVICE_MAP.
    class_type: type | None = None

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = OmyLeaderDevice


class OmyLeaderDevice(DeviceBase):
    def __init__(self, cfg: OmyLeaderCfg, retargeters=None):
        super().__init__(retargeters)
        self.cfg = cfg
        self._callbacks: dict[str, Callable] = {}
        self._warned = False
        self._zeroed = not cfg.auto_zero
        self._t_up: float | None = None
        self._mapper = self._make_mapper()
        self._last = self._mapper_home()
        self._open_source()

    # ---- mapping ------------------------------------------------------------------------
    def _make_mapper(self):
        c = self.cfg
        if c.target == "franka":
            from dataclasses import replace

            from .franka_config import DEFAULT_OMY_TO_FRANKA_CONFIG, load_calib
            from .franka_retarget import OmyToFrankaRetarget

            fcfg = load_calib(c.calib_path) if c.calib_path else DEFAULT_OMY_TO_FRANKA_CONFIG
            return OmyToFrankaRetarget(replace(fcfg, dt=c.dt, vel_scale=c.vel_scale))
        if c.target == "omy":
            from .mirror import MirrorConfig, OmyMirror

            mcfg = MirrorConfig.from_json(c.calib_path) if c.calib_path else MirrorConfig()
            mcfg.dt, mcfg.vel_scale = c.dt, c.vel_scale
            return OmyMirror(mcfg)
        raise ValueError(f"unknown target {c.target!r} (omy | franka)")

    def _mapper_home(self) -> np.ndarray:
        if self.cfg.target == "franka":
            from .franka_config import FRANKA_HOME

            return np.concatenate([FRANKA_HOME, [1.0]])
        from .mirror import OMY_DEFAULT_Q

        return np.concatenate([OMY_DEFAULT_Q, [1.0]])

    def _auto_zero(self, a: dict[str, float]) -> None:
        if self.cfg.target == "franka":
            from .franka_retarget import OmyToFrankaRetarget

            self._mapper = OmyToFrankaRetarget(self._mapper.auto_offset(a))
            zeros = self._mapper.config.omy_zero_rad
        else:
            from .mirror import OmyMirror

            self._mapper = OmyMirror(self._mapper.auto_zero(a))
            zeros = self._mapper.cfg.zero_rad
        print("[omy_leader] auto-zero:", {k: round(v, 3) for k, v in zeros.items()})

    # ---- reader --------------------------------------------------------------------------
    def _open_source(self):
        c = self.cfg
        if c.source == "tcp":
            from .link import FrameClient

            self._client = FrameClient(c.tcp_host, c.tcp_port)
            self._latest = self._client.latest
        elif c.source == "serial":
            from .omy_serial import OmySerialLeader

            self._serial = OmySerialLeader(c.port, c.baudrate, gripper_open_pos=c.gripper_open_pos)
            self._buf: tuple[dict | None, float] = (None, 0.0)
            self._stop = threading.Event()
            self._serial_thread = threading.Thread(target=self._serial_loop, daemon=True)
            self._serial_thread.start()
            self._latest = lambda: (self._buf[0], time.time() - self._buf[1])
        else:
            raise ValueError(f"unknown source {c.source!r} (serial | tcp)")

    def _serial_loop(self):
        period = 1.0 / self.cfg.hz
        while not self._stop.is_set():
            try:
                self._buf = (self._serial.read(), time.time())
            except Exception as e:  # noqa: BLE001 - keep the loop alive across bus hiccups
                print(f"[omy_leader] serial read error: {e}")
                time.sleep(0.5)
            time.sleep(period)

    # ---- DeviceBase ----------------------------------------------------------------------
    def __str__(self) -> str:
        c = self.cfg
        wrist = "J5 = j5, J6 = 180deg - j4, J7 = 45deg - j6" if c.target == "franka" else "1:1"
        arm = "1:1, J3 locked, J4 = j3" if c.target == "franka" else "1:1"
        return (
            f"OMY-L100 leader ({c.source}) -> {c.target}\n"
            f"  arm    : {arm} (sign / zero from calib)\n"
            f"  wrist  : {wrist}\n"
            "  gripper: squeeze trigger -> close\n"
        )

    def reset(self) -> None:
        self._mapper.reset()

    def add_callback(self, key, func: Callable) -> None:
        self._callbacks[key] = func

    def _get_raw_data(self):
        return self._latest()

    def _tensor(self, a: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(a, dtype=torch.float32, device=self.cfg.sim_device)

    def advance(self) -> torch.Tensor:
        a, age = self._latest()
        if a is None or age > self.cfg.stale_s:
            if not self._warned:
                print(f"[omy_leader] no fresh leader data (age {age:.2f}s) - holding last command")
                self._warned = True
            return self._tensor(self._last)
        self._warned = False
        if not self._zeroed:
            if self._t_up is None:
                self._t_up = time.time()
                print(f"[omy_leader] hold the leader in the {self.cfg.target} default pose - zeroing in {self.cfg.auto_zero_settle_s:.0f}s")
            if time.time() - self._t_up < self.cfg.auto_zero_settle_s:
                return self._tensor(self._last)
            self._auto_zero(a)
            self._zeroed = True
        self._last = self._mapper.step(a)
        return self._tensor(self._last)

    def close(self):
        if self.cfg.source == "serial":
            self._stop.set()
            self._serial_thread.join()
            self._serial.close()
        else:
            self._client.close()


# Register with Isaac Lab's factory so env_cfg.teleop_devices can name this device.
try:
    from isaaclab.devices import teleop_device_factory as _f

    if hasattr(_f, "DEVICE_MAP"):
        _f.DEVICE_MAP[OmyLeaderCfg] = OmyLeaderDevice
except Exception:  # pragma: no cover
    pass
