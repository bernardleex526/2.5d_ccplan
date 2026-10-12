import json
from datetime import datetime
from pathlib import Path
import sys

import pytest

# Support the plan's direct pytest command without an installed ROS package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from m20_bridge import protocol


@pytest.fixture(autouse=True)
def fixed_time(monkeypatch):
    class FixedDateTime:
        @staticmethod
        def now():
            return datetime(2023, 1, 1)

    monkeypatch.setattr(protocol, 'datetime', FixedDateTime)


def expected_message(type, command, items):
    return {'PatrolDevice': {
        'Type': type,
        'Command': command,
        'Time': '2023-01-01 00:00:00',
        'Items': items,
    }}


def test_header_and_json_byte_length():
    # A multibyte body larger than one byte can represent distinguishes byte
    # length from character count and exposes swapped length-field bytes.
    payload = protocol.encode(100, 100, {'Description': '巡检' * 256})
    body = payload[16:]
    assert payload[:4] == bytes.fromhex('EB 90 EB 90')
    assert len(body) > 255
    assert payload[4] == len(body) & 0xff
    assert payload[5] == len(body) >> 8
    assert int.from_bytes(payload[4:6], 'little') == len(body)
    assert payload[6:9] == bytes.fromhex('01 00 01')
    assert payload[9:16] == b'\x00' * 7
    assert len(body) > len(body.decode('utf-8'))
    assert json.loads(body) == expected_message(
        100, 100, {'Description': '巡检' * 256})


@pytest.mark.parametrize('gait_param', [1, 14])
def test_allowed_gaits(gait_param):
    expected = expected_message(2, 23, {'GaitParam': gait_param})
    assert protocol.decode(protocol.encode(2, 23, {'GaitParam': gait_param})) == expected
    assert protocol.decode(protocol.gait(gait_param)) == expected


@pytest.mark.parametrize('gait_param', [0, 2, -1, 15, None, '1', 1.0, True])
def test_other_gaits_raise(gait_param):
    with pytest.raises(ValueError):
        protocol.encode(2, 23, {'GaitParam': gait_param})
    with pytest.raises(ValueError):
        protocol.gait(gait_param)


def test_missing_gait_raises():
    with pytest.raises(ValueError):
        protocol.encode(2, 23, {})


@pytest.mark.parametrize('value, expected', [(-2, -1), (-1, -1), (0, 0), (1, 1), (2, 1)])
@pytest.mark.parametrize('axis_name', ['X', 'Y', 'Yaw'])
def test_planar_axes_clamped(axis_name, value, expected):
    items = {name: 0 for name in ('X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw')}
    items[axis_name] = value
    decoded = protocol.decode(protocol.encode(2, 21, items))
    assert decoded['PatrolDevice']['Items'][axis_name] == expected
    assert items[axis_name] == value  # Encoding must not alter the caller's command.


@pytest.mark.parametrize('value', [-2, -1, 0, 1, 2])
def test_nonplanar_axes_fixed_zero(value):
    items = {'X': 0, 'Y': 0, 'Z': value, 'Roll': value, 'Pitch': value, 'Yaw': 0}
    decoded = protocol.decode(protocol.encode(2, 21, items))
    assert decoded['PatrolDevice']['Items'] == {
        'X': 0, 'Y': 0, 'Z': 0, 'Roll': 0, 'Pitch': 0, 'Yaw': 0,
    }
    assert items['Z'] == items['Roll'] == items['Pitch'] == value


def test_axis_constructor():
    decoded = protocol.decode(protocol.axis(x=2, y=-2, z=2, roll=-2, pitch=2, yaw=1))
    assert decoded == expected_message(2, 21, {
        'X': 1, 'Y': -1, 'Z': 0, 'Roll': 0, 'Pitch': 0, 'Yaw': 1,
    })


def test_heartbeat_constructor():
    assert protocol.decode(protocol.heartbeat()) == expected_message(100, 100, {})


@pytest.mark.parametrize('mode', [0, 1])
def test_usage_mode_constructor(mode):
    expected = expected_message(1101, 5, {'Mode': mode})
    assert protocol.decode(protocol.usage_mode(mode)) == expected
    assert protocol.decode(protocol.encode(1101, 5, {'Mode': mode})) == expected


@pytest.mark.parametrize('mode', [-1, 2, None, '0', True])
def test_invalid_usage_mode_raises(mode):
    with pytest.raises(ValueError):
        protocol.usage_mode(mode)
    with pytest.raises(ValueError):
        protocol.encode(1101, 5, {'Mode': mode})


@pytest.mark.parametrize('type, command, items', [
    (100, 100, {}),
    (1101, 5, {'Mode': 0}),
    (2, 23, {'GaitParam': 14}),
    (2, 21, {'X': 1, 'Y': -1, 'Z': 0, 'Roll': 0, 'Pitch': 0, 'Yaw': 0}),
    (1002, 1, {'Gait': 14, 'Description': '巡检'}),
])
def test_encode_decode_round_trip(type, command, items):
    expected = expected_message(type, command, items)
    payload = protocol.encode(type, command, items)
    assert json.loads(payload[16:]) == expected
    assert protocol.decode(payload) == expected


def test_decode_independent_manual_message():
    body = b'{"PatrolDevice":{"Type":100,"Command":100,"Time":"2023-01-01 00:00:00","Items":{}}}'
    # Build independently of encode using the literal appendix header layout.
    header = bytes.fromhex('EB 90 EB 90') + len(body).to_bytes(2, 'little')
    header += bytes.fromhex('01 00 01 00 00 00 00 00 00 00')
    assert protocol.decode(header + body) == expected_message(100, 100, {})


@pytest.mark.parametrize('payload', [
    b'',
    bytes.fromhex('EB 90 EB 90'),
    bytes.fromhex('EB 90 EB 90 02 00 01 00 01 00 00 00 00 00 00 00') + b'{',
    bytes.fromhex('EB 90 EB 90 02 00 01 00 01 00 00 00 00 00 00 00') + b'{}x',
    bytes.fromhex('EB 91 EB 90 02 00 01 00 01 00 00 00 00 00 00 00') + b'{}',
])
def test_decode_rejects_broken_framing(payload):
    with pytest.raises(ValueError):
        protocol.decode(payload)
