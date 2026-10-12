"""Pure Python M20 messages for execution-plan section 5.2 and task T04.

Body fields, local-time format, and command identifiers follow the manufacturer's
Lynx M20 Software Development Manual V0.0.4-0, sections 1.1.6 and 1.2.1-1.2.5:
https://www.deeprobotics.us/wp-content/uploads/2025/08/Lynx-M20-Series-Software-Development-Manual-V0.0.4-0.pdf

The plan's EB 90 EB 90 synchronization bytes and fixed bytes 6-15 are reproduced
verbatim from the appendix contract, even where the manual's header table
conflicts. Their semantics remain plan open item P15; no message-ID progression
or undocumented header fields are introduced. decode returns the full JSON
object, including the PatrolDevice envelope, without changing received values.
"""

from datetime import datetime
import json


_SYNC = bytes.fromhex('EB 90 EB 90')
_HEADER_TAIL = bytes.fromhex('01 00 01 00 00 00 00 00 00 00')
_AXES = ('X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw')
_NONPLANAR_AXES = ('Z', 'Roll', 'Pitch')


def encode(type, command, items) -> bytes:
    """Encode one command with a local YYYY-MM-DD HH:MM:SS sending time.

    Basic/Stair gait commands accept integer GaitParam 1/14 only. Usage mode
    accepts integer Mode 0/1. Axis commands clamp each ratio and fix Z/Roll/Pitch
    to zero, as required for planar hill patrol. Caller items are never modified.
    """
    items = dict(items)
    if (type, command) == (2, 23):
        value = items.get('GaitParam')
        if not isinstance(value, int) or isinstance(value, bool) or value not in (1, 14):
            raise ValueError('GaitParam must be integer 1 (Basic) or 14 (Stair)')
    elif (type, command) == (1101, 5):
        value = items.get('Mode')
        if not isinstance(value, int) or isinstance(value, bool) or value not in (0, 1):
            raise ValueError('Mode must be integer 0 (regular) or 1 (navigation)')
    elif (type, command) == (2, 21):
        for name in _AXES:
            items[name] = max(-1, min(1, items[name]))
        for name in _NONPLANAR_AXES:
            items[name] = 0

    body = json.dumps({'PatrolDevice': {
        'Type': type,
        'Command': command,
        'Time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Items': items,
    }}, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8')
    return _SYNC + len(body).to_bytes(2, 'little') + _HEADER_TAIL + body


def decode(payload) -> dict:
    """Decode one complete plan-contract datagram to its JSON dictionary."""
    if len(payload) < 16:
        raise ValueError('M20 datagram is shorter than its 16-byte header')
    if payload[:4] != _SYNC or payload[6:16] != _HEADER_TAIL:
        raise ValueError('M20 datagram header does not match the section 5.2 contract')
    if int.from_bytes(payload[4:6], 'little') != len(payload) - 16:
        raise ValueError('M20 JSON byte length does not match the header')
    return json.loads(payload[16:].decode('utf-8'))


def heartbeat() -> bytes:
    """Construct the manual's Type=100/Command=100 empty-items heartbeat."""
    return encode(100, 100, {})


def usage_mode(mode) -> bytes:
    """Construct Type=1101/Command=5: mode 0 regular, mode 1 navigation."""
    return encode(1101, 5, {'Mode': mode})


def gait(gait_param) -> bytes:
    """Construct Type=2/Command=23: gait 1 Basic, gait 14 Stair."""
    return encode(2, 23, {'GaitParam': gait_param})


def axis(x=0, y=0, z=0, roll=0, pitch=0, yaw=0) -> bytes:
    """Construct Type=2/Command=21 with clamped planar command ratios."""
    return encode(2, 21, {
        'X': x, 'Y': y, 'Z': z, 'Roll': roll, 'Pitch': pitch, 'Yaw': yaw,
    })
