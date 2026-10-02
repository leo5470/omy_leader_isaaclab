"""Exercise real serial-reader configuration against a fake Dynamixel bus."""

import math
import sys
from types import SimpleNamespace

import pytest

from omy_leader_isaaclab import omy_serial as serial


@pytest.fixture
def bus(monkeypatch):
    state = SimpleNamespace(registers={}, writes=[], closed=False, fail_goal=False)

    class Port:
        def __init__(self, name):
            pass

        def openPort(self):
            return True

        def setBaudRate(self, baud):
            return True

        def closePort(self):
            state.closed = True

    class Packet:
        def __init__(self, protocol):
            pass

        def write(self, port, motor, address, value):
            if address in (10, 11, 20, 38):
                assert state.registers.get((motor, 64), 0) == 0
            if motor == 7 and address == 64 and value == 1:
                assert (7, 116) in state.registers
            if state.fail_goal and address == 116:
                return 1, 0
            state.writes.append((motor, address, value))
            state.registers[motor, address] = value
            if motor == 7 and address == 64 and value == 1:
                state.registers[7, 116] = state.registers.get((7, 132), 2788)
            return 0, 0

        write1ByteTxRx = write2ByteTxRx = write4ByteTxRx = write

        def read(self, port, motor, address):
            return state.registers.get((motor, address), 0), 0, 0

        read1ByteTxRx = read4ByteTxRx = read

        def getTxRxResult(self, result):
            return "communication failed"

        def getRxPacketError(self, error):
            return ""

    class Reader:
        def __init__(self, *args):
            pass

        def addParam(self, motor):
            return True

        def txRxPacket(self):
            return 0

        def getData(self, motor, address, length):
            return state.registers.get((motor, address), 2048)

    monkeypatch.setitem(sys.modules, "dynamixel_sdk", SimpleNamespace(
        PortHandler=Port, PacketHandler=Packet, GroupSyncRead=Reader))
    return state


def test_plugin_goal_and_register_sequence(bus):
    leader = serial.OmySerialLeader()
    assert bus.registers[7, 116] == 1973  # exact LeRobot truncation, not rounding
    assert bus.registers[7, 10] == 1
    assert bus.registers[7, 20] == 100
    assert bus.registers[7, 11] == 5
    assert bus.registers[7, 64] == 1
    leader.close()
    assert bus.registers[7, 64] == 0 and bus.closed
    leader.close()  # cleanup is idempotent


def test_custom_goal_and_legacy_units_leave_arm_unchanged(bus):
    leader = serial.OmySerialLeader(gripper_open_pos=60)
    goal = int(0.6 * 4095)
    assert bus.registers[7, 116] == goal
    bus.registers[7, 132] = goal
    bus.registers[1, 132] = 3072
    assert leader.read()["gripper.pos"] == 0
    assert leader.read()["joint_1.pos"] == pytest.approx(math.pi / 2)
    bus.registers[7, 132] = goal + 256
    assert leader.read()["gripper.pos"] == pytest.approx(256 * math.pi / 4095)
    leader.close()


def test_configuration_failure_turns_torque_off_and_closes(bus):
    bus.fail_goal = True
    with pytest.raises(ConnectionError):
        serial.OmySerialLeader()
    assert bus.registers[7, 64] == 0 and bus.closed


def test_unchanged_eeprom_is_not_rewritten(bus):
    bus.registers[7, 10] = 1
    bus.registers[7, 20] = 100
    leader = serial.OmySerialLeader()
    assert not any(m == 7 and address in (10, 20) for m, address, value in bus.writes)
    leader.close()


@pytest.mark.parametrize("value", [-1, 101, math.nan, math.inf])
def test_bad_rest_position_rejected_before_open(bus, value):
    with pytest.raises(ValueError):
        serial.OmySerialLeader(gripper_open_pos=value)
    assert not bus.writes


def test_publisher_forwards_custom_rest_position(monkeypatch):
    from omy_leader_isaaclab import publisher
    calls = []

    class Leader:
        def __init__(self, port, baud, *, gripper_open_pos):
            calls.append(gripper_open_pos)

        def read(self):
            raise KeyboardInterrupt

        def close(self):
            calls.append("closed")

    monkeypatch.setattr(serial, "OmySerialLeader", Leader)
    monkeypatch.setattr(sys, "argv", ["publisher", "--print", "--gripper-open", "55"])
    publisher.main()
    assert calls == [55.0, "closed"]
