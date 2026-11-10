"""Hermetic stand-ins for the Feetech serial stack.

A real ``FeetechMotorsBus`` is built (its SCS ``PortHandler`` constructor only stores the port name, nothing is
opened) and its four SDK objects are swapped for the ones below, so lerobot's own ``read``/``write``/``sync_read``/
``connect`` code runs unchanged against a scripted bus. Nothing here touches a serial port, a camera, or sleeps.
"""
from contextlib import contextmanager

from scservo_sdk import COMM_RX_TIMEOUT, COMM_SUCCESS

from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.motors.motors_bus import Motor, MotorCalibration, MotorNormMode

CTRL = FeetechMotorsBus.model_ctrl_table["sts3215"]
MODEL_NUMBER = FeetechMotorsBus.model_number_table["sts3215"]
TORQUE_ENABLE = CTRL["Torque_Enable"][0]
GOAL_POSITION = CTRL["Goal_Position"][0]
PRESENT_POSITION = CTRL["Present_Position"][0]
FW_MAJOR, FW_MINOR = 0, 1
OVERLOAD_BIT = 0x20

SO101_MOTORS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def _from_bytes(data):
    return sum(int(b) << (8 * i) for i, b in enumerate(data))


class Sim:
    """Scripted state shared by the fake port, packet handler and sync groups."""

    def __init__(self, ids):
        self.ids = list(ids)
        self.stalled = False            # every packet times out ("There is no status packet!")
        self.heal_on_reopen = True      # closing + reopening the port clears the stall (the CH343 case)
        self.ping_fail_rounds = 0       # pings fail for this many port openings (arm not powered yet)
        self.pings_fail = False
        self.sync_fail_count = 0        # the next N sync transactions time out whatever the stall state
        self.read_fail_addrs = set()    # single-register reads at these addresses time out
        self.overload_ids = set()       # motors whose status packets carry the overload error bit
        self.error_text = "[RxPacketError] Overload error!"
        self.firmware = {i: (3, 9) for i in self.ids}
        self.reg = {(i, PRESENT_POSITION): 2000 + i for i in self.ids}
        self.writes = []                # (id, addr, value) in order, single and sync writes alike
        self.port = FakePort(self)
        self.packet = FakePacket(self)
        self.reader = FakeSyncReader(self)
        self.writer = FakeSyncWriter(self)

    def comm(self):
        return COMM_RX_TIMEOUT if self.stalled else COMM_SUCCESS

    def error(self, id_):
        return OVERLOAD_BIT if id_ in self.overload_ids else 0

    def register_write(self, id_, addr, value):
        self.writes.append((id_, addr, value))
        self.reg[(id_, addr)] = value
        if addr == TORQUE_ENABLE:
            self.overload_ids.discard(id_)   # model: cycling torque clears the latched overload

    def torque_writes(self, id_):
        return [v for i, a, v in self.writes if i == id_ and a == TORQUE_ENABLE]

    def goal_writes(self, id_):
        return [v for i, a, v in self.writes if i == id_ and a == GOAL_POSITION]


class FakePort:
    def __init__(self, sim):
        self.sim = sim
        self.is_open = False
        self.is_using = False
        self.open_ok = True
        self.log = []
        self.timeouts = []

    @property
    def opens(self):
        return self.log.count("open")

    def openPort(self):
        self.log.append("open")
        if not self.open_ok:
            return False
        self.is_open = True
        if self.sim.heal_on_reopen:
            self.sim.stalled = False
        if self.sim.ping_fail_rounds > 0:
            self.sim.ping_fail_rounds -= 1
            self.sim.pings_fail = True
        else:
            self.sim.pings_fail = False
        return True

    def closePort(self):
        self.log.append("close")
        self.is_open = False

    def clearPort(self):
        self.log.append("clear")

    def setPacketTimeoutMillis(self, ms):
        self.timeouts.append(ms)

    def getBaudRate(self):
        return 1_000_000


class FakePacket:
    def __init__(self, sim):
        self.sim = sim

    def ping(self, port, id_):
        if self.sim.stalled or self.sim.pings_fail or id_ not in self.sim.ids:
            return 0, COMM_RX_TIMEOUT, 0
        return MODEL_NUMBER, COMM_SUCCESS, self.sim.error(id_)

    def _read(self, id_, addr):
        s = self.sim
        if s.stalled or addr in s.read_fail_addrs:
            return 0, COMM_RX_TIMEOUT, 0       # the SDK returns 0, not None, on a failed read
        if addr == FW_MAJOR:
            return s.firmware[id_][0], COMM_SUCCESS, s.error(id_)
        if addr == FW_MINOR:
            return s.firmware[id_][1], COMM_SUCCESS, s.error(id_)
        return s.reg.get((id_, addr), 0), COMM_SUCCESS, s.error(id_)

    def read1ByteTxRx(self, port, id_, addr):
        return self._read(id_, addr)

    read2ByteTxRx = read1ByteTxRx
    read4ByteTxRx = read1ByteTxRx

    def writeTxRx(self, port, id_, addr, length, data):
        s = self.sim
        if s.stalled:
            return COMM_RX_TIMEOUT, 0
        err = s.error(id_)
        s.register_write(id_, addr, _from_bytes(data))
        return COMM_SUCCESS, err

    def getTxRxResult(self, comm):
        if comm == COMM_SUCCESS:
            return "[TxRxResult] Communication success."
        return "[TxRxResult] There is no status packet!"

    def getRxPacketError(self, error):
        return self.sim.error_text if error else ""


class FakeSyncReader:
    def __init__(self, sim):
        self.sim = sim
        self.ids = []
        self.start_address = 0
        self.data_length = 0

    def clearParam(self):
        self.ids = []

    def addParam(self, id_):
        self.ids.append(id_)

    def txRxPacket(self):
        if self.sim.sync_fail_count > 0:
            self.sim.sync_fail_count -= 1
            return COMM_RX_TIMEOUT
        return self.sim.comm()

    def getData(self, id_, addr, length):
        return self.sim.reg.get((id_, addr), 0)


class FakeSyncWriter:
    def __init__(self, sim):
        self.sim = sim
        self.staged = []
        self.start_address = 0
        self.data_length = 0

    def clearParam(self):
        self.staged = []

    def addParam(self, id_, data):
        self.staged.append((id_, _from_bytes(data)))

    def txPacket(self):
        if self.sim.stalled:
            return COMM_RX_TIMEOUT
        for id_, value in self.staged:
            self.sim.register_write(id_, self.start_address, value)
        return COMM_SUCCESS


def make_bus(sim=None, port="/dev/tty.so101_follower", connected=True):
    """A FeetechMotorsBus with the SO-101 motor map whose SDK objects are the fakes above."""
    motors = {}
    for i, name in enumerate(SO101_MOTORS, start=1):
        mode = MotorNormMode.RANGE_0_100 if name == "gripper" else MotorNormMode.RANGE_M100_100
        motors[name] = Motor(i, "sts3215", mode)
    calibration = {
        n: MotorCalibration(id=m.id, drive_mode=0, homing_offset=0, range_min=0, range_max=4095)
        for n, m in motors.items()
    }
    bus = FeetechMotorsBus(port, motors, calibration)
    sim = sim or Sim([m.id for m in motors.values()])
    bus.port_handler = sim.port
    bus.packet_handler = sim.packet
    bus.sync_reader = sim.reader
    bus.sync_writer = sim.writer
    sim.port.is_open = connected
    return bus, sim


class RecordingBus:
    """Duck-typed bus for SOFollower.configure: records every write and whether torque was disabled around it."""

    def __init__(self, motors=SO101_MOTORS):
        self.motors = {m: None for m in motors}
        self.writes = []
        self.torque_blocks = 0
        self._in_block = False

    @contextmanager
    def torque_disabled(self, motors=None):
        self.torque_blocks += 1
        self._in_block = True
        try:
            yield
        finally:
            self._in_block = False

    def configure_motors(self):
        self.writes.append(("configure_motors", None, None, None, self._in_block))

    def write(self, data_name, motor, value, normalize=True, num_retry=0):
        self.writes.append((data_name, motor, value, normalize, self._in_block))

    def plugin_writes(self, motor="gripper"):
        """The writes the plugin adds: it is the only caller passing normalize=False."""
        return [(n, v, inblk) for n, m, v, norm, inblk in self.writes if m == motor and norm is False]
