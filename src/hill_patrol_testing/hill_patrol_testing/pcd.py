"""Minimal PCD reader (ascii / binary / binary_compressed) for x,y,z fields, numpy only.

binary_compressed is PCL's layout: header, then uint32 compressed size, uint32
uncompressed size, then an LZF block holding the fields column-major (all x, then
all y, ...), each column `size * count * points` bytes.
"""
import numpy as np

_TYPES = {('F', 4): '<f4', ('F', 8): '<f8', ('I', 1): 'i1', ('I', 2): '<i2', ('I', 4): '<i4',
          ('I', 8): '<i8', ('U', 1): 'u1', ('U', 2): '<u2', ('U', 4): '<u4', ('U', 8): '<u8'}


def lzf_decompress(data, expected):
    out = bytearray(expected)
    i = o = 0
    n = len(data)
    while i < n:
        ctrl = data[i]
        i += 1
        if ctrl < 32:
            length = ctrl + 1
            out[o:o + length] = data[i:i + length]
            i += length
            o += length
        else:
            length = ctrl >> 5
            ref = o - ((ctrl & 0x1f) << 8) - 1
            if length == 7:
                length += data[i]
                i += 1
            ref -= data[i]
            i += 1
            length += 2
            if ref < 0:
                raise ValueError('corrupt LZF stream')
            for k in range(length):  # overlapping copies are allowed by LZF
                out[o + k] = out[ref + k]
            o += length
    if o != expected:
        raise ValueError(f'LZF size mismatch: {o} != {expected}')
    return bytes(out)


def read_xyz(path):
    """Return an (N, 3) float64 array of finite x,y,z points."""
    with open(path, 'rb') as stream:
        raw = stream.read()
    header, offset = {}, 0
    while True:
        end = raw.index(b'\n', offset)
        line = raw[offset:end].decode('ascii').strip()
        offset = end + 1
        if not line or line.startswith('#'):
            continue
        key, *values = line.split()
        header[key.upper()] = values
        if key.upper() == 'DATA':
            break
    fields = header['FIELDS']
    sizes = [int(s) for s in header['SIZE']]
    types = header['TYPE']
    counts = [int(c) for c in header.get('COUNT', ['1'] * len(fields))]
    points = int(header['POINTS'][0])
    kind = header['DATA'][0]
    dtype = np.dtype([(f, _TYPES[(t, s)], (c,)) if c > 1 else (f, _TYPES[(t, s)])
                      for f, t, s, c in zip(fields, types, sizes, counts)])
    if kind == 'ascii':
        table = np.loadtxt(raw[offset:].decode('ascii').splitlines(), ndmin=2)
        columns = {}
        col = 0
        for f, c in zip(fields, counts):
            columns[f] = table[:, col]
            col += c
        xyz = np.stack([columns['x'], columns['y'], columns['z']], 1)
    elif kind == 'binary':
        data = np.frombuffer(raw, dtype=dtype, count=points, offset=offset)
        xyz = np.stack([data['x'], data['y'], data['z']], 1).astype(float)
    elif kind == 'binary_compressed':
        compressed, size = np.frombuffer(raw, dtype='<u4', count=2, offset=offset)
        block = lzf_decompress(raw[offset + 8:offset + 8 + int(compressed)], int(size))
        columns, start = {}, 0
        for f, t, s, c in zip(fields, types, sizes, counts):
            length = s * c * points
            columns[f] = np.frombuffer(block, dtype=_TYPES[(t, s)], count=c * points, offset=start)
            start += length
        xyz = np.stack([columns['x'], columns['y'], columns['z']], 1).astype(float)
    else:
        raise ValueError(f'unsupported PCD DATA {kind}')
    return xyz[np.all(np.isfinite(xyz), axis=1)]
