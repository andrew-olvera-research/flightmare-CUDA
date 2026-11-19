"""Wire-level PX4 SITL contract regression tests (no firmware process needed)."""

import socket

import numpy as np
import pytest

pytest.importorskip("pymavlink")
from pymavlink.dialects.v20 import common as mav
from flightlib_cuda.px4_sitl import Link, rotor_actions


@pytest.fixture
def connection():
    with socket.create_server(("127.0.0.1", 0)) as server:
        with socket.create_connection(server.getsockname()) as peer:
            sock, _ = server.accept()
            with sock:
                yield Link(sock), peer


def test_motor_order_and_scaling():
    # FR, BL, FL, BR -> FL, FR, BR, BL. Action zero is Flightlib hover.
    np.testing.assert_allclose(rotor_actions([0, 0.25, 0.5, 1]), [0, -0.5, 0.5, -0.25])
    for bad in (-0.1, float("nan"), 1.2):
        with pytest.raises(ValueError):
            rotor_actions([0, 0, bad, 0])


@pytest.mark.parametrize("stamp,flags", [(4000, 0), (3000, 1), (5000, 1), (4000, 1)])
def test_actuator_barrier(connection, stamp, flags):
    link, peer = connection
    payload = mav.MAVLink_hil_actuator_controls_message(stamp, [0.5] * 16, 128, flags)
    peer.sendall(payload.pack(mav.MAVLink(None)))
    if stamp == 4000 and flags == 1:
        assert link.actuator(4000).controls[:4] == [0.5] * 4
    else:
        with pytest.raises(RuntimeError, match="lockstep violation"):
            link.actuator(4000)


def test_sensor_wire_axes_units_and_timestamp(connection):
    link, peer = connection
    state = np.zeros(13)
    state[:4] = [11, 2, 3, 1]
    state[7:10] = [1, -2, 3]
    link.send(100000, np.array([1, 2, 3, 4, 5, 6.0]), state, 25)
    decoder = mav.MAVLink(None)
    packets = []
    peer.settimeout(1)
    while len(packets) < 2:
        packets.extend(decoder.parse_buffer(peer.recv(8192)) or ())
    gps, imu = packets
    assert gps.time_usec == imu.time_usec == 100000
    assert [gps.vn, gps.ve, gps.vd] == [100, 200, -300]
    assert gps.alt == 491000 and gps.lat > 473977420 and gps.lon < 85455940
    assert [imu.xacc, imu.yacc, imu.zacc, imu.xgyro, imu.ygyro, imu.zgyro] == [
        4,
        -5,
        -6,
        1,
        -2,
        -3,
    ]
