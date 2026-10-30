"""The device test cases (see devtest.py)."""
from fractions import Fraction

import numpy as np

from devtest import (ApiCase, Buf, Case, Local, Val, CL_MAP_READ, CL_MAP_WRITE, CL_MEM_READ_ONLY,
                     CL_MEM_READ_WRITE, CL_MEM_USE_HOST_PTR, NULL_ID)

K = 'tests/kernels/'
P = 'tests/probe/'

I32_MIN, I32_MAX = -2 ** 31, 2 ** 31 - 1


def nonzero(lo, hi):
    def f(rng, shape):
        v = rng.integers(lo, hi, shape, endpoint=True)
        return np.where(v == 0, 1, v)
    return f


def with_specials(specials, lo, hi):
    def f(rng, shape):
        v = rng.uniform(lo, hi, shape).astype(np.float32).reshape(-1)
        s = np.asarray(specials, np.float32)
        v[:len(s)] = s
        return v.reshape(shape)
    return f


CONVERT_SPECIALS = [np.nan, np.inf, -np.inf, 1e10, -1e10, 0.5, 1.5, 2.5, -0.5, -1.5, 255.5, 256.0,
                    -1.0, 0.0, -0.0, 2147483520.0, -2147483648.0, 3e9, 254.5, 1e-40]


def convert_check(inputs, ref, dev):
    # Out-of-range float -> int conversions without _sat are implementation-defined
    # (6.2.3.3) and single precision denormals may be flushed to zero (7.2).
    v = inputs[0].astype(np.float64)
    in_range = (v > I32_MIN - 1.0) & (v < I32_MAX + 1.0)
    errs = []

    def cmp(what, want, got, rows):
        for j in np.nonzero(rows & (want != got))[0][:4]:
            errs.append('%s[%d] (v=%r) = %r, want %r' % (what, j, inputs[0][j], got[j], want[j]))
    i32, i32_ref = dev[1].reshape(-1, 2), ref[1].reshape(-1, 2)
    cmp('convert_int_sat', i32_ref[:, 0], i32[:, 0], np.ones(len(v), bool))
    cmp('convert_int_rte', i32_ref[:, 1], i32[:, 1], in_range)
    cmp('convert_uchar_sat', ref[2], dev[2], np.ones(len(v), bool))
    back, back_ref = dev[3], ref[3].copy()
    tiny = np.abs(back_ref) < np.finfo(np.float32).tiny
    back_ref[tiny & (back == 0)] = 0
    cmp('back', back_ref.view(np.uint32), back.view(np.uint32), in_range)
    return errs


def atomics_check(n):
    def check(inputs, ref, dev):
        errs = []
        u0, u = inputs[1], dev[1]
        if not 0 <= int(u[2]) < n:
            errs.append('u[2] = %d after atomic_xchg, want an id' % u[2])
        if not 0 <= int(u[3]) < n:
            errs.append('u[3] = %d after atomic_cmpxchg, want an id' % u[3])
        for j in (0, 1, 4, 5, 6):
            if u[j] != ref[1][j]:
                errs.append('u[%d] = 0x%08x, want 0x%08x' % (j, u[j], ref[1][j]))
        if u0[7:].tobytes() != u[7:].tobytes():
            errs.append('u[7..] changed')
        return errs
    return check


def atomic_offsets_check(n):
    def check(inputs, ref, dev):
        errs = []
        u0, s0 = inputs[0].astype(np.int64), inputs[1]
        u, s, o = dev[0].astype(np.int64), dev[1], dev[2].reshape(n, 8).astype(np.int64)
        mask = 0xffffffff

        def perm(col, start, step, what):
            got = sorted(int(x) & mask for x in o[:, col])
            want = sorted((start + step * k) & mask for k in range(n))
            if got != want:
                errs.append('%s returned %s..., want a permutation of %s...' % (
                    what, got[:4], want[:4]))
        perm(0, u0[3], 1, 'atomic_inc(&u[3])')
        perm(1, u0[5], 2, 'atomic_add(u + 5, 2)')
        perm(2, int(s0[1]['b'][2]), -1, 'atomic_dec(&s[1].b[2])')
        perm(7, u0[13], -3, 'atomic_sub(&u[13], 3)')
        for j, want in ((3, u0[3] + n), (5, u0[5] + 2 * n), (13, u0[13] - 3 * n)):
            if (u[j] & mask) != (want & mask):
                errs.append('u[%d] = %d, want %d' % (j, u[j], want & mask))
        if (int(s[1]['b'][2]) & mask) != ((int(s0[1]['b'][2]) - n) & mask):
            errs.append('s[1].b[2] = %d' % s[1]['b'][2])
        if s[0]['c'] != max(int(s0[0]['c']), n - 1 - 4):
            errs.append('s[0].c = %d after atomic_max' % s[0]['c'])
        if s[2]['c'] != min(int(s0[2]['c']), -4):
            errs.append('s[2].c = %d after atomic_min' % s[2]['c'])
        for j in range(4):
            if not (u[7 + j] == 0 or (0 <= u[7 + j] < n and u[7 + j] & 3 == j)):
                errs.append('u[%d] = %d after atomic_cmpxchg' % (7 + j, u[7 + j]))
        swapped = sorted([int(x) for x in o[:, 5]] + [int(u[12])])
        if swapped != sorted([int(u0[12])] + list(range(n))):
            errs.append('atomic_xchg(&u[12]) values are not the ids and the old value')
        return errs
    return check


def locals_check(n):
    def check(inputs, ref, dev):
        # o[4i + 2] adds lp[0], which the kernel never writes
        r = ref[0].reshape(-1, 4)[:n]
        d = dev[0].reshape(-1, 4)[:n]
        bad = (r[:, [0, 1, 3]] != d[:, [0, 1, 3]]).any(axis=1)
        return ['o[%d..]: device %s, expected %s' % (4 * i, d[i], r[i])
                for i in np.nonzero(bad)[0][:6]]
    return check


def shapes_check(n):
    def check(inputs, ref, dev):
        u0, u, o = inputs[0], dev[0], dev[2]
        errs = []
        if u[3] != u0[3] + n:
            errs.append('u[3] = %d, want %d' % (u[3], u0[3] + n))
        if not u0[3] <= o[0] < u0[3] + n:
            errs.append('o[0] = %d' % o[0])
        return errs
    return check


PARAMS = np.dtype([('a', '<i4'), ('b', '<f4'), ('c', '<i2')], align=True)
S = np.dtype([('a', '<u4'), ('b', '<u4', 3), ('c', '<i4')], align=True)

N = 1000
CASES = [
    Case('vadd', K + 'vadd.cl', 'vadd', (N,), [Buf('i4', N), Buf('i4', N), Buf('i4', N, 'zeros')]),
    Case('vadd_256k', K + 'vadd.cl', 'vadd', (1 << 18,),
         [Buf('i4', 1 << 18), Buf('i4', 1 << 18), Buf('i4', 1 << 18, 'zeros')]),
    Case('vadd_offset', K + 'vadd.cl', 'vadd', (300,), [Buf('i4', N), Buf('i4', N),
         Buf('i4', N, 'zeros')], offset=(123,)),
    Case('copy', P + 'copy.cl', 'copy', (N,), [Buf('u4', N, 'zeros'), Buf('u4', N),
         Buf('u4', 4, 'zeros')]),
    Case('workitem_1d', K + 'workitem.cl', 'workitem', (37,), [Buf('u4', 8 * 64, 'zeros')]),
    Case('workitem_2d', K + 'workitem.cl', 'workitem', (6, 4), [Buf('u4', 8 * 40, 'zeros')],
         offset=(2, 1)),
    Case('workitem_3d', K + 'workitem.cl', 'workitem', (5, 3, 4), [Buf('u4', 8 * 40, 'zeros')],
         offset=(1, 2, 3)),
    Case('saxpy', K + 'float_math.cl', 'saxpy', (N,), [Val(np.float32(2.5)), Buf('f4', N),
         Buf('f4', N)], ulp=1, atol=3.1e-5),	# mad() may round a * x: an ulp of |a * x| <= 250
    Case('fdiv_sqrt', K + 'float_math.cl', 'fdiv_sqrt', (N,), [Buf('f4', N, lo=0.0, hi=100.0),
         Buf('f4', N, 'zeros')], ulp=8, atol=1e-6),
    Case('transcendental', K + 'builtins.cl', 'transcendental', (N,),
         [Buf('f4', N, lo=-10.0, hi=10.0), Buf('f4', 8 * N, 'zeros')], ulp=16, atol=1e-5),
    Case('intops', K + 'intops.cl', 'intops', (N,), [Buf('i4', N, lo=I32_MIN + 1),
         Buf('i4', N, nonzero(-100000, 100000)), Buf('i4', 6 * N, 'zeros')]),
    Case('branches', K + 'control.cl', 'branches', (N,), [Buf('i4', N, lo=-40, hi=60),
         Buf('i4', N, 'zeros'), Val(np.int32(40))]),
    Case('private_array', K + 'private_array.cl', 'private_array', (200,),
         [Buf('i4', 16 * 200, lo=-1000, hi=1000), Buf('i4', 200, 'zeros')]),
    Case('misc', K + 'misc.cl', 'misc', (N,), [Buf('i4', N, 'zeros'),
         Val(np.array((5, 2.5, -3), PARAMS)), Buf('i4', 4, values=[10, 20, 30, 40])]),
    Case('convert', K + 'convert.cl', 'convert', (N,),
         [Buf('f4', N, with_specials(CONVERT_SPECIALS, -3e9, 3e9)), Buf('i4', 2 * N, 'zeros'),
          Buf('u1', N, 'zeros'), Buf('f4', N, 'zeros')], check=convert_check, skip_compare=(1, 2, 3)),
    Case('convert_small', K + 'convert.cl', 'convert', (N,),
         [Buf('f4', N, lo=-300.0, hi=300.0), Buf('i4', 2 * N, 'zeros'),
          Buf('u1', N, 'zeros'), Buf('f4', N, 'zeros')]),
    Case('vec4', K + 'vectors.cl', 'vec4', (N,), [Buf('(4,)f4', N), Buf('(4,)f4', N),
         Buf('(4,)f4', N, 'zeros'), Buf('f4', N, 'zeros')], ulp=4, atol=1e-3),
    Case('vload_store', K + 'vectors.cl', 'vload_store', (N,), [Buf('f4', 3 * N),
         Buf('f4', 3 * N + 5, 'zeros')]),
    Case('bytes', K + 'bytes.cl', 'bytes', (333,), [Buf('u1', 333), Buf('u1', 337, 'zeros'),
         Buf('i2', 335, 'zeros')]),
    Case('char4ops', K + 'bytes.cl', 'char4ops', (N,), [Buf('(4,)i1', N),
         Buf('(4,)i1', N, 'zeros')]),
    Case('atomics', K + 'atomics.cl', 'atomics', (N,),
         [Buf('i4', 4, values=[0, 0, I32_MIN, I32_MAX]),
          Buf('u4', 8, values=[0, 1000, 0, 0, 0xffffffff, 0, 0, 77]),
          Buf('i4', N, lo=-1000, hi=1000)], check=atomics_check(N), skip_compare=(1,)),
    Case('atomic_offsets', K + 'atomic_offsets.cl', 'atomic_offsets', (200,),
         [Buf('u4', 16, values=[0, 0, 0, 100, 0, 7, 0, 0, 0, 0, 0, 0, 55, 1000, 0, 0]),
          Buf(S, 4, 'zeros'), Buf('i4', 8 * 200, 'zeros')],
         check=atomic_offsets_check(200), skip_compare=(0, 1, 2)),
    Case('atomic_stress', K + 'atomics.cl', 'atomic_stress', (N,),
         [Buf('i4', 3, 'zeros'), Buf('u4', 16, 'zeros'), Buf('u4', N, lo=0, hi=1000),
          Val(np.int32(16))]),
    Case('pointers', K + 'pointers.cl', 'pointers', (64,), [Buf('i4', 16, lo=-1000, hi=1000),
         Buf('(5,)i4', 4, lo=-1000, hi=1000), Buf('i4', 8 * 64, 'zeros')]),	# 4 S structs
    Case('alias', K + 'pointers.cl', 'alias', (N,), [Buf('i4', N, lo=-1000, hi=1000),
         Buf('i4', 2 * N, 'zeros'), Val(np.int32(0))]),
    Case('locals', K + 'kernel_locals.cl', 'loc', (64,), [Buf('u4', 4 * 64, 'zeros'), Local(16)],
         check=locals_check(64), skip_compare=(0,)),
    Case('shapes', K + 'kernel_locals.cl', 'shapes', (100,), [Buf('u4', 8, values=[0, 0, 0, 5]),
         Buf(S, 2, 'zeros'), Buf('i4', 4, 'zeros')], check=shapes_check(100),
         skip_compare=(0, 2)),
    Case('reduce', K + 'local_reduce.cl', 'reduce', (N,), [Buf('i4', N), Buf('i4', N, 'zeros'),
         Local(4)]),
    Case('local_atomic', K + 'local_reduce.cl', 'local_atomic', (N,), [Buf('i4', N, 'zeros')]),
]


# fp_ops rows 0..: infinities, NaN, overflow, signed zeros and a result below FLT_MIN
FP_A = [np.inf, -np.inf, np.nan, np.inf, 3e38, -3e38, 1e38, 0.0, -0.0, 2e-38, 1.5, -2.5]
FP_B = [1.0, np.inf, 2.0, -np.inf, 3e38, 3e38, 10.0, -0.0, -0.0, 0.5, 0.0, 0.0]


def f32_rtz(x):
    """An exact value (a Fraction) rounded toward zero to float32; beyond FLT_MAX
    that is +-FLT_MAX."""
    m = abs(x)
    f = np.float32(min(float(m), float(np.finfo(np.float32).max)))
    while Fraction(float(f)) > m:
        f = np.nextafter(f, np.float32(0))
    with np.errstate(over='ignore'):
        while np.isfinite(np.nextafter(f, np.float32(np.inf))) and \
                Fraction(float(np.nextafter(f, np.float32(np.inf)))) <= m:
            f = np.nextafter(f, np.float32(np.inf))
    return -f if x < 0 else f


def fp_ops_check(inputs, ref, dev):
    # The device rounds toward zero (CL_FP_ROUND_TO_ZERO), so each result is
    # compared with the exact value rounded that way instead of pocl's rte.
    a, b = inputs[0], inputs[1]
    out = dev[2].reshape(-1, 4)
    ops = (('a + b', lambda x, y: x + y), ('a - b', lambda x, y: x - y),
           ('a * b', lambda x, y: x * y),
           ('(int)a + 0.5f * b', lambda x, y: int(x) + Fraction(1, 2) * y))
    errs = []
    for k, (what, op) in enumerate(ops):
        wrong = []
        for j in range(len(a)):
            x, y = float(a[j]), float(b[j])
            if k == 3 and not (np.isfinite(x) and abs(x) < 2.0 ** 31):
                continue		# (int) of these is undefined
            if np.isfinite(x) and np.isfinite(y):
                e = op(Fraction(x), Fraction(y))
                # an exact zero keeps IEEE's sign, which float64 gives
                want = f32_rtz(e) if e else np.float32(op(np.float64(x), np.float64(y)))
            else:
                with np.errstate(invalid='ignore', over='ignore'):
                    want = np.float32(op(np.float64(x), np.float64(y)))
            got = out[j, k]
            if np.isnan(want) and np.isnan(got):
                continue
            if want.view(np.uint32) == got.view(np.uint32):
                continue
            if abs(want) < np.finfo(np.float32).tiny and got == 0:
                continue		# below FLT_MIN may be flushed (no CL_FP_DENORM)
            wrong.append('%s with a=%r, b=%r: %r, want %r (rtz)' % (what, a[j], b[j], got, want))
        errs += wrong[:4]
        if len(wrong) > 4:
            errs.append('%s: %d wrong' % (what, len(wrong)))
    return errs

# API scripts: (s, rng) adds the calls; outputs and return codes must match pocl.
CL_COMPLETE = 0


def ints(rng, n, lo=-1000, hi=1000):
    return rng.integers(lo, hi, n, endpoint=True).astype(np.int32).tobytes()


def api_buffers(s, rng):
    s.buffer(0, ints(rng, 300))
    s.buffer(1, bytes(1200))
    s.add('write', id=1, offset=40, size=400, data=ints(rng, 100), blocking=1)
    s.add('copy', src=0, dst=1, src_offset=8, dst_offset=600, size=512)
    s.add('copy', src=0, dst=0, src_offset=0, dst_offset=800, size=400)
    s.add('copy', src=0, dst=0, src_offset=0, dst_offset=100, size=400)	# overlaps
    s.add('copy', src=0, dst=1, src_offset=1100, dst_offset=0, size=200)	# out of range
    s.add('read', id=1, offset=1000, size=400, blocking=1)			# out of range
    s.add('write', id=0, offset=3, size=5, data=b'abcde', blocking=0)	# unaligned
    s.read(0, 1200)
    s.read(1, 1200, blocking=0)
    s.add('finish')


def api_rect(s, rng):
    s.buffer(0, ints(rng, 1024))
    s.buffer(1, bytes(4096))
    host = ints(rng, 512)
    # 2D and 3D regions, default and explicit pitches
    s.add('write_rect', id=1, bo0=5, bo1=2, bo2=1, ho0=3, ho1=1, ho2=0, r0=37, r1=6, r2=3,
          brow=64, bslice=1024, hrow=48, hslice=0, host_size=2048, data=host, blocking=1)
    s.add('read_rect', id=0, bo0=12, bo1=3, bo2=0, ho0=1, ho1=2, ho2=1, r0=20, r1=7, r2=2,
          brow=100, bslice=900, hrow=0, hslice=0, host_size=600, data=bytes(range(256)) * 2
          + bytes(88), blocking=1)
    s.add('copy_rect', src=0, dst=1, so0=4, so1=1, so2=1, do0=0, do1=0, do2=2, r0=60, r1=5, r2=2,
          srow=80, sslice=800, drow=0, dslice=0)
    s.add('copy_rect', src=1, dst=1, so0=0, so1=0, so2=0, do0=16, do1=0, do2=0, r0=16, r1=4, r2=1,
          srow=32, sslice=0, drow=32, dslice=0)	# rows interleave without sharing bytes
    s.add('copy_rect', src=1, dst=1, so0=0, so1=0, so2=0, do0=8, do1=1, do2=0, r0=16, r1=4, r2=1,
          srow=32, sslice=0, drow=32, dslice=0)	# overlaps
    s.add('read_rect', id=0, bo0=0, bo1=0, bo2=0, ho0=0, ho1=0, ho2=0, r0=64, r1=64, r2=2,
          brow=0, bslice=0, hrow=0, hslice=0, host_size=8192, data=None, blocking=1)  # too big
    s.read(1, 4096)


def api_map(s, rng):
    s.buffer(0, ints(rng, 256))
    s.buffer(1, ints(rng, 256), flags=CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR)
    for b in (0, 1):
        s.add('map', id=b, flags=CL_MAP_READ, offset=64, size=256, data=None, blocking=1)
        s.add('map', id=b, flags=CL_MAP_WRITE, offset=128, size=100, data=ints(rng, 25),
              blocking=0)
        s.add('map', id=b, flags=CL_MAP_READ | CL_MAP_WRITE, offset=0, size=1024,
              data=ints(rng, 256), blocking=1)
        s.add('map', id=b, flags=CL_MAP_READ, offset=1000, size=100, data=None, blocking=1)
    s.read(0, 1024)
    s.read(1, 1024)


def api_subbuffer(s, rng):
    s.buffer(0, ints(rng, 1024))
    s.add('subbuffer', id=1, parent=0, flags=0, origin=256, size=512)
    s.add('subbuffer', id=2, parent=0, flags=CL_MEM_READ_ONLY, origin=1024, size=1000)
    s.add('subbuffer', id=3, parent=0, flags=0, origin=100, size=64)		# misaligned
    # out of range (past the harness's guard bytes too)
    s.add('subbuffer', id=4, parent=0, flags=0, origin=3840, size=1024)
    s.add('write', id=1, offset=8, size=40, data=ints(rng, 10), blocking=1)
    s.add('copy', src=2, dst=1, src_offset=0, dst_offset=100, size=300)
    s.add('copy', src=0, dst=1, src_offset=300, dst_offset=0, size=40)	# overlaps via the parent
    s.read(2, 1000)
    s.read(0, 4096)


def api_events(s, rng):
    s.buffer(0, bytes(512))
    s.add('user_event', event=0)
    s.add('write', id=0, offset=0, size=256, data=ints(rng, 64), blocking=0, wait=0, event=1)
    s.add('status', slot=1, compare=False)		# queued or submitted
    s.add('status', slot=0)				# CL_SUBMITTED
    s.add('read', id=0, offset=0, size=512, blocking=0, wait=1, event=2)
    s.add('set_status', slot=0, status=CL_COMPLETE)
    s.add('wait', slot=2)
    s.add('status', slot=1)
    s.add('status', slot=2)
    s.add('set_status', slot=0, status=CL_COMPLETE)	# twice
    s.add('finish')


def api_kernel_events(s, rng):
    """A kernel waits for a user event; its arguments are the ones set when it was enqueued."""
    s.buffer(0, ints(rng, 100))
    s.buffer(1, bytes(400))
    s.buffer(2, bytes(400))
    s.kernel(0, 'scale')
    s.add('arg_mem', kid=0, index=0, id=0)
    s.add('arg_mem', kid=0, index=1, id=1)
    s.add('arg_value', kid=0, index=2, data=np.int32(3).tobytes(), size=4)
    s.add('user_event', event=0)
    s.ndrange(0, (100,), wait=0, event=1)
    s.add('arg_mem', kid=0, index=1, id=2)
    s.add('arg_value', kid=0, index=2, data=np.int32(-5).tobytes(), size=4)
    s.ndrange(0, (50,), offset=(25,))
    s.add('set_status', slot=0, status=CL_COMPLETE)
    s.add('wait', slot=1)
    s.add('status', slot=1)
    s.read(1, 400)
    s.read(2, 400)


def api_kernel_errors(s, rng):
    s.buffer(0, ints(rng, 64))
    s.kernel(0, 'scale')
    s.kernel(1, 'no_such_kernel')
    s.ndrange(0, (64,))						# arguments not set
    s.add('arg_mem', kid=0, index=0, id=0)
    s.add('arg_mem', kid=0, index=1, id=0)
    s.add('arg_value', kid=0, index=2, data=bytes(8), size=8)	# wrong size
    s.add('arg_value', kid=0, index=3, data=bytes(4), size=4)	# no such argument
    s.add('arg_local', kid=0, index=0, size=16)			# not a __local argument
    s.add('arg_value', kid=0, index=2, data=np.int32(2).tobytes(), size=4)
    s.ndrange(0, (64,), local=0)
    s.ndrange(0, (0,), compare=False)		# 1.1: invalid; pocl takes it (2.1 and later)
    s.read(0, 256)


def api_repeat(s, rng):
    """One kernel, many dispatches over the same buffer, then a chain of two kernels."""
    s.buffer(0, ints(rng, 333))
    s.buffer(1, bytes(333 * 4))
    s.kernel(0, 'add_one')
    s.kernel(1, 'scale')
    s.add('arg_mem', kid=0, index=0, id=0)
    for _ in range(40):
        s.ndrange(0, (333,))
    s.add('arg_mem', kid=1, index=0, id=0)
    s.add('arg_mem', kid=1, index=1, id=1)
    s.add('arg_value', kid=1, index=2, data=np.int32(7).tobytes(), size=4)
    s.ndrange(1, (333,))
    s.add('arg_mem', kid=0, index=0, id=1)
    s.ndrange(0, (333,))
    s.read(0, 333 * 4)
    s.read(1, 333 * 4)


def api_null_arg(s, rng):
    s.buffer(0, bytes(64 * 4))
    s.buffer(1, np.array([7], np.int32).tobytes())
    s.kernel(0, 'null_arg')
    s.add('arg_mem', kid=0, index=0, id=0)
    s.add('arg_mem', kid=0, index=1, id=NULL_ID)
    s.ndrange(0, (32,))
    s.add('arg_mem', kid=0, index=1, id=1)
    s.ndrange(0, (32,), offset=(32,))
    s.read(0, 256)


VALUE_P = np.dtype([('a', '<i4'), ('b', '<f4'), ('c', '<i2')], align=True)


def api_values(s, rng):
    s.buffer(0, bytes(32 * 4))
    s.kernel(0, 'values')
    vals = [np.int8(-77), np.uint8(201), np.int16(-30000), np.uint16(65000),
            np.array([-1, 2, -128, 127], np.int8), np.array([255, 9], np.uint8),
            np.array([-32768, 1234], np.int16), np.array([1.5, -2.25], np.float32),
            np.array([3.0, -0.0, 1e20, 0.0], np.float32),	# float3 takes 4 floats
            np.array([np.inf, -7.5, 1e-30, 42.0], np.float32),
            np.array((-9, 0.125, -2), VALUE_P), np.uint32(0xdeadbeef)]
    s.add('arg_mem', kid=0, index=0, id=0)
    for i, v in enumerate(vals):
        b = np.asarray(v).tobytes()
        s.add('arg_value', kid=0, index=i + 1, data=b, size=len(b))
    s.ndrange(0, (1,))
    s.read(0, 32 * 4)
    s.buffer(1, bytes(32 * 4))
    s.kernel(1, 'wide_values')
    s.add('arg_mem', kid=1, index=0, id=1)
    b = np.arange(-4, 4, dtype=np.int32).tobytes()
    s.add('arg_value', kid=1, index=1, data=b, size=len(b))
    b = (np.arange(16, dtype=np.float32) * -1.75).tobytes()
    s.add('arg_value', kid=1, index=2, data=b, size=len(b))
    s.add('arg_value', kid=1, index=3, data=np.uint32(77).tobytes(), size=4)
    s.ndrange(1, (1,))
    s.read(1, 32 * 4)


def api_scratch(s, rng):
    """512 bytes of private and __local memory a work-item, over more than one scene."""
    nx, ny = 300, 120
    s.buffer(0, ints(rng, 1024))
    s.buffer(1, bytes(nx * ny * 4))
    s.kernel(0, 'scratch')
    s.add('arg_mem', kid=0, index=0, id=0)
    s.add('arg_mem', kid=0, index=1, id=1)
    s.add('arg_local', kid=0, index=2, size=256)
    s.ndrange(0, (nx, ny))
    s.read(1, nx * ny * 4)


A = K + 'api.cl'
CASES += [
    ApiCase('api_buffers', None, api_buffers),
    ApiCase('api_rect', None, api_rect),
    ApiCase('api_map', None, api_map),
    ApiCase('api_subbuffer', None, api_subbuffer),
    ApiCase('api_events', None, api_events),
    ApiCase('api_kernel_events', A, api_kernel_events),
    ApiCase('api_kernel_errors', A, api_kernel_errors),
    ApiCase('api_repeat', A, api_repeat),
    ApiCase('api_null_arg', A, api_null_arg),
    ApiCase('api_values', A, api_values),
    ApiCase('api_scratch', A, api_scratch),
    Case('fp_ops', A, 'fp_ops', (N,), [Buf('f4', N, with_specials(FP_A, -1000.0, 1000.0)),
         Buf('f4', N, with_specials(FP_B, -1000.0, 1000.0)), Buf('f4', 4 * N, 'zeros')],
         check=fp_ops_check, skip_compare=(2,)),
]
