# omy_leader_isaaclab

Use a **ROBOTIS OMY-L100 leader arm** to teleoperate robots in **NVIDIA Isaac Lab** — in joint
space, no IK, from a single pip-installable repo.

| target | mapping | env |
|---|---|---|
| **OMY-F3M** (ROBOTIS Lab / [cyclo_lab](https://github.com/ROBOTIS-GIT/robotis_lab)) | 1:1 joint mirror (same kinematics) | `Cyclo-Lift-Cube-OMY-Leader-v0`, `Cyclo-Pick-Place-Bottle-OMY-Leader-v0` |
| **Franka Panda** (Isaac Lab stack task) | arm 1:1 with J3 locked, wrist remapped (fixed per-joint map, optional exact ZYZ solve) | `Isaac-Stack-Cube-Franka-JointTeleop-v0` (+ `-Cam-v0` with cameras) |

The leader is exposed as an Isaac Lab **teleop device** (`OmyLeaderCfg` / `OmyLeaderDevice`) registered
in the official `create_teleop_device` factory, so scripts that build their device from
`env_cfg.teleop_devices` — e.g. cyclo_lab's `record_demos.py` — use it with `--teleop_device omy_leader`
and no code changes.

Why joint space: an end-effector teleop through differential IK lags, drifts and cannot control the
redundant joint; a leader arm gives you every joint directly.

## Install

On the machine running Isaac Sim / Isaac Lab (tested: Isaac Sim 5.1.0 with Isaac Lab v2.3.0 + robotis_lab,
and with Isaac Lab `main`):

```bash
# inside your Isaac Lab python environment
pip install git+https://github.com/leo5470/omy_leader_isaaclab@robolab-trigger-fix
```

Dependencies are only `numpy` and `dynamixel-sdk` (the L100 is read directly over Protocol 2.0, 4 Mbps).
Optional: `[lerobot]` to read the leader through [`lerobot_teleoperator_omy`](https://github.com/charlie8612/lerobot_teleoperator_omy)
instead, `[viewer]` for the laptop video viewer.

For the OMY targets install ROBOTIS Lab as well and add the 3-line hook from `scripts/cyclo_lab_hook.patch`
to `cyclo_lab/__init__.py` (this is what a `record_demos.py` invocation needs to see the device and envs).

## RoboLab trigger correction

This fork applies RoboLab's L100 serial trigger correction to upstream commit
`ed4608bd8c49133e3b9fc9d231bee651d8489b65`. No post-install patch is required.

The spring rest uses the LeRobot RANGE_0_100 conversion: `--gripper-open 48.2`
sets the goal to `int(48.2 / 100 * 4095) = 1973` ticks. The reader configures
`Drive_Mode=1` and `Homing_Offset=100` with torque disabled, writes the spring goal
before enabling torque, then reasserts and checks it afterward because the XC330
can reset the goal when torque is enabled. Background readers stop before cleanup;
cleanup disables trigger torque and closes the serial port.

Gripper samples preserve the legacy publisher's half-angle units:
`(present_position - goal_ticks) * pi / 4095`. Arm readings keep their existing
full-radian units. The publisher, calibration, monitor, and Isaac Lab device all
forward custom spring rest positions. Recalibrate gripper endpoints if they were
captured using the uncorrected serial reader.

The trigger regression tests use a fake Dynamixel bus and do not open hardware:

```bash
python -m pytest tests -q
```

## Quick start (leader plugged into the sim machine)

```bash
export ISAACLAB=~/IsaacLab
scripts/run_teleop.sh --target omy    --auto-zero             # OMY in cyclo_lab, with a display
scripts/run_teleop.sh --target franka --auto-zero --headless --stream   # Franka, video on port 5556
```

`--auto-zero`: when the stream comes up, hold the leader in the target's default pose (what you see on
screen) for 3 s; per-joint zeros are computed. Add `--port /dev/ttyUSB0` if your udev name differs.

Record demos with cyclo_lab's own script (device settings come from env vars, no new flags):

```bash
OMY_LEADER_SOURCE=serial OMY_LEADER_AUTO_ZERO=1 \
python scripts/imitation_learning/isaaclab_recorder/record_demos.py \
    --task Cyclo-Pick-Place-Bottle-OMY-Leader-v0 --teleop_device omy_leader --dataset_file datasets/omy.hdf5
```

## Advanced: leader on a different machine / sim on a headless server

The common setup is one PC with Isaac Sim and the L100 on USB — everything above is that. If your sim
runs on a headless server elsewhere:

```
[L100 host]  omy-leader-publisher --port /dev/robotis_left         # 72-byte TCP frames, 100 Hz
             ssh -N -R 5555:localhost:5555 <sim host>               # or relay through the laptop
[sim host]   scripts/run_teleop.sh --target franka --source tcp --headless --stream
[laptop]     SIM_HOST=<sim> L100_HOST=<l100 host> scripts/laptop_connect.sh   # tunnels + viewer window
```

The video is the sim's own camera rendered to JPEG over TCP (30 fps, a few ms over a LAN), which works
through plain ssh where Isaac's WebRTC livestream (UDP) cannot.

## Calibration

`--auto-zero` is usually enough. For a persistent calibration (signs, zeros, gripper endpoints, wrist mode):

```bash
omy-leader-calib                      # lerobot-style: sweep every joint to its limits, then hold the ready pose
omy-leader-monitor --calib omy_calib.json   # live raw angles + mapped Franka joints
scripts/run_teleop.sh --target franka --calib omy_calib.json
```

`examples/omy_calib_franka.json` is a real calibration of one L100 (signs j2/j4/j6 inverted relative to
the URDF axes; wrist mode `direct`, reset with the hand in line with the forearm). Flip a joint's
`omy_sign` if it moves the wrong way in the sim.

### Franka wrist mapping

The L100 is UR-like (yaw–pitch–pitch, then pitch–roll–spin) and the Franka with J3 = 0 is
yaw–pitch–pitch, roll–pitch–roll. Around the operator's working pose the joints correspond one to one,
so the default `direct` mode uses

```
J1 = j1   J2 = j2   J3 = 0   J4 = j3        J5 = j5   J6 = 180° − j4   J7 = 45° − j6
```

(with per-joint sign/zero from calibration). `wrist_mode: zyz` instead matches the tool orientation
exactly via a closed-form ZYZ decomposition (`franka_wrist.py`), at the cost of J5/J7 coupling near the
Franka wrist singularity (J6 = 0 or 180°).

## Layout

```
omy_leader_isaaclab/
  device.py           OmyLeaderCfg(target="omy"|"franka", source="serial"|"tcp") + OmyLeaderDevice(DeviceBase)
  omy_serial.py       DynamixelSDK reader (arm torque-off extended position, spring-loaded gripper trigger)
  link.py             TCP leader stream (publisher.py <-> device)
  mirror.py           L100 -> OMY 1:1 (sign / zero / rate limit / gripper hysteresis / auto-zero)
  franka_config.py    Franka mapping config, joint limits, calibration loader
  franka_retarget.py  L100 -> Franka 8-d action (J3 locked, wrist direct or ZYZ)
  franka_wrist.py     wrist geometry: direct map and closed-form ZYZ solve
  franka_calib.py     lerobot-style calibration -> omy_calib.json;  franka_monitor.py: live readout
  omy_env_cfg.py      cyclo_lab OMY tasks with absolute JointPositionAction + teleop_devices
  franka_env_cfg.py   Isaac Lab Franka stack task, same treatment (+ camera variant)
  teleop.py           standalone runner (--target, --source, --stream)
  publisher.py / fake_publisher.py / video.py / viewer.py
scripts/              run_teleop.sh, laptop_connect.sh, cyclo_lab_hook.patch
tests/                pure-numpy tests of both mappings (pytest)
```

## Status

- Franka target: driven with a real L100 (Isaac Lab `main`, Isaac Sim 5.1.0).
- OMY target: verified end to end with a simulated leader in cyclo_lab; the DynamixelSDK reader has
  been exercised only through the lerobot path so far — please open an issue if your L100 misbehaves.
- cyclo_lab's `record_demos.py` imports `omni.ui`, so it needs a display (not `--headless`).

## License

Apache-2.0.
