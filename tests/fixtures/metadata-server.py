#!/usr/bin/env python3
"""Immutable executable for owned metadata subprocess tests."""
import os
import pathlib
import sys
import time

folder = pathlib.Path(__file__).parent
mode = (folder / 'mode').read_text()
(folder / 'pid').write_text(str(os.getpid()))
assert os.read(0, 1) == b'', 'metadata stdin must be closed'
if mode == 'success':
    os.write(1, b'out\x00\xff')
    os.write(2, b'err\x00\xfe')
elif mode == 'nonzero':
    print('version: 9999 (must-not-be-accepted)', flush=True)
    print('fixture-private-output', file=sys.stderr, flush=True)
    sys.exit(7)
elif mode == 'large-both':
    sys.stdout.buffer.write(b'o' * 65536)
    sys.stdout.flush()
    sys.stderr.buffer.write(b'e' * 65536)
    sys.stderr.flush()
elif mode.startswith('flood-'):
    stream = sys.stdout if mode == 'flood-stdout' else sys.stderr
    stream.buffer.write(b'x' * 300000)
    stream.flush()
    time.sleep(120)
elif mode == 'closed-pipes-hang':
    os.close(1)
    os.close(2)
    time.sleep(120)
elif mode == 'hang':
    time.sleep(120)
elif mode.startswith('version-'):
    stream = sys.stdout if mode == 'version-stdout' else sys.stderr
    print('backend noise', file=stream)
    print('version: 9013 (test-hash)', file=stream)
elif mode == 'devices':
    print('Available devices:\n  CUDA0: Fixture GPU (8192 MiB, 7000 MiB free)')
    print('backend noise', file=sys.stderr)
else:
    raise AssertionError('unknown fixture mode')
